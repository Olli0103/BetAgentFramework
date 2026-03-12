"""Ironclad Alias Resolution — the name normalization gatekeeper.

Every team/player name that enters the database MUST pass through this resolver.
It normalizes text, applies sport-specific heuristics (tennis "Lastname I." patterns),
performs fuzzy matching against known aliases, and auto-registers unknown names.

Resolution order:
  1. Exact match (case-insensitive, accent-stripped) against alias cache
  2. Tennis abbreviated name heuristic ("Sinner J." → "Jannik Sinner")
  3. Fuzzy match (SequenceMatcher ≥ threshold) against canonical names
  4. Auto-register as new canonical name (first encounter)

Golden Rule: NO raw string ever reaches the DB unresolved.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from difflib import SequenceMatcher

from sqlalchemy import select
from sqlalchemy.orm import Session

from bet_agent.db.models import Sport, TeamAlias

logger = logging.getLogger(__name__)

# Minimum similarity ratio for fuzzy matching.
# 0.88 is high enough to catch "Bayern Munich" vs "Bayern München"
# but low enough to avoid false positives like "Sinner" vs "Skinner".
FUZZY_THRESHOLD = 0.88


# ── Text normalization ──────────────────────────────────────────────


def normalize_text(name: str) -> str:
    """Strip, collapse whitespace, remove diacritics (é→e, ö→o, ü→u).

    Preserves original casing — use make_key() for case-insensitive comparison.
    """
    name = " ".join(name.strip().split())
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def make_key(name: str) -> str:
    """Lowercase comparison key with diacritics removed."""
    return normalize_text(name).lower()


# ── Tennis name heuristics ──────────────────────────────────────────


_RE_LASTNAME_INITIAL = re.compile(
    r"^(.+?)\s+([A-Z])\.?(?:\s?([A-Z])\.?)?$"
)  # "Sinner J." or "Sinner J.J." or "Sinner J"

_RE_INITIAL_LASTNAME = re.compile(
    r"^([A-Z])\.?\s+(.+)$"
)  # "J. Sinner" or "J Sinner"

_RE_LASTNAME_COMMA_FIRST = re.compile(
    r"^(.+?),\s*(.+)$"
)  # "Sinner, Jannik"


def is_abbreviated_tennis_name(name: str) -> bool:
    """Check if name looks like an abbreviated tennis format."""
    s = name.strip()
    return bool(
        _RE_LASTNAME_INITIAL.match(s)
        or _RE_INITIAL_LASTNAME.match(s)
        or _RE_LASTNAME_COMMA_FIRST.match(s)
    )


def try_match_abbreviated(abbreviated: str, full_name: str) -> bool:
    """Check if an abbreviated name matches a full name.

    Handles:
      - "Sinner J." matches "Jannik Sinner"
      - "J. Sinner" matches "Jannik Sinner"
      - "Sinner, Jannik" matches "Jannik Sinner"
      - "Sinner, J." matches "Jannik Sinner"
    """
    abbr = abbreviated.strip()
    full_parts = full_name.strip().split()
    if len(full_parts) < 2:
        return False

    full_first = full_parts[0]
    full_last = full_parts[-1]
    full_first_key = make_key(full_first)
    full_last_key = make_key(full_last)

    # Pattern 3 first (comma has highest priority to avoid regex overlap):
    # "Lastname, Firstname" or "Lastname, F."
    m = _RE_LASTNAME_COMMA_FIRST.match(abbr)
    if m:
        lastname = m.group(1).strip()
        firstname_or_initial = m.group(2).strip().rstrip(".")
        if make_key(lastname) != full_last_key:
            return False
        # Full first name match
        if make_key(firstname_or_initial) == full_first_key:
            return True
        # Initial match
        if len(firstname_or_initial) == 1:
            return full_first[0].upper() == firstname_or_initial[0].upper()
        return False

    # Pattern 1: "Lastname F." or "Lastname F.G."
    m = _RE_LASTNAME_INITIAL.match(abbr)
    if m:
        lastname = m.group(1).strip()
        initial = m.group(2)
        return (
            make_key(lastname) == full_last_key
            and full_first[0].upper() == initial
        )

    # Pattern 2: "F. Lastname"
    m = _RE_INITIAL_LASTNAME.match(abbr)
    if m:
        initial = m.group(1)
        lastname = m.group(2).strip()
        return (
            make_key(lastname) == full_last_key
            and full_first[0].upper() == initial
        )

    return False


# ── The Resolver ────────────────────────────────────────────────────


class IroncladAliasResolver:
    """Sport-aware name resolution with fuzzy matching and auto-registration.

    Thread-safety: NOT thread-safe. One resolver per session per sport.
    """

    def __init__(
        self,
        session: Session,
        sport: Sport,
        *,
        fuzzy_threshold: float = FUZZY_THRESHOLD,
        auto_register: bool = True,
    ) -> None:
        self._session = session
        self._sport = sport
        self._fuzzy_threshold = fuzzy_threshold
        self._auto_register = auto_register
        # key → canonical_name
        self._cache: dict[str, str] = {}
        # canonical_key → canonical_name (preserves original casing)
        self._canonicals: dict[str, str] = {}
        # Resolution tracking — confidence metrics
        self._exact_hits: int = 0
        self._fuzzy_hits: int = 0
        self._auto_registered: int = 0
        self._unresolved: list[str] = []  # names that fell through (auto_register=False)
        self._load_cache()

    def _load_cache(self) -> None:
        """Load all aliases for this sport (+ global/unscoped) into memory."""
        query = select(TeamAlias).where(
            (TeamAlias.sport == self._sport) | (TeamAlias.sport.is_(None))
        )
        rows = self._session.execute(query).scalars().all()
        for row in rows:
            key = make_key(row.alias)
            self._cache[key] = row.canonical_name
            canon_key = make_key(row.canonical_name)
            self._canonicals[canon_key] = row.canonical_name

        logger.info(
            "Loaded %d aliases for %s (%d canonical names)",
            len(self._cache), self._sport.value, len(self._canonicals),
        )

    def resolve(self, raw_name: str) -> str:
        """Resolve a raw team/player name to its canonical form.

        GUARANTEE: This method never returns a raw, unnormalized string.
        Every name is either matched to an existing canonical name or
        auto-registered as a new canonical.
        """
        if not raw_name or not raw_name.strip():
            return raw_name

        normalized = normalize_text(raw_name)
        key = normalized.lower()

        # 1. Exact match in alias cache
        if key in self._cache:
            self._exact_hits += 1
            return self._cache[key]

        # 2. Exact match against canonical names (canonical → itself)
        if key in self._canonicals:
            self._exact_hits += 1
            return self._canonicals[key]

        # 3. Tennis abbreviated name heuristic
        if self._sport == Sport.TENNIS and is_abbreviated_tennis_name(normalized):
            for canon_name in self._canonicals.values():
                if try_match_abbreviated(normalized, canon_name):
                    self._register_alias(canon_name, raw_name.strip())
                    logger.info(
                        "Tennis abbreviation matched: '%s' → '%s'",
                        raw_name, canon_name,
                    )
                    return canon_name

        # 4. Fuzzy match against all known canonical names
        best_match, best_ratio = self._fuzzy_match(key)
        if best_match and best_ratio >= self._fuzzy_threshold:
            self._fuzzy_hits += 1
            self._register_alias(best_match, raw_name.strip())
            logger.info(
                "Fuzzy matched '%s' → '%s' (%.1f%%)",
                raw_name, best_match, best_ratio * 100,
            )
            return best_match

        # 5. Auto-register as new canonical name
        if self._auto_register:
            self._auto_registered += 1
            canonical = normalized  # accent-stripped, case-preserved
            self._register_canonical(canonical, raw_name.strip())
            logger.info(
                "New canonical registered: '%s' [%s]",
                canonical, self._sport.value,
            )
            return canonical

        # Fallback (auto_register=False): return normalized form
        self._unresolved.append(raw_name.strip())
        logger.warning(
            "UNRESOLVED alias (strict mode): '%s' [%s]",
            raw_name, self._sport.value,
        )
        return normalized

    def _fuzzy_match(self, key: str) -> tuple[str | None, float]:
        """Find best fuzzy match among canonical names."""
        best_name: str | None = None
        best_ratio = 0.0

        for canon_key, canon_name in self._canonicals.items():
            ratio = SequenceMatcher(None, key, canon_key).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_name = canon_name

        return best_name, best_ratio

    def _register_alias(self, canonical: str, alias: str) -> None:
        """Register a new alias → canonical mapping (persists to DB)."""
        alias_key = make_key(alias)
        if alias_key in self._cache:
            return

        existing = self._session.execute(
            select(TeamAlias).where(
                TeamAlias.alias == alias,
                (TeamAlias.sport == self._sport) | (TeamAlias.sport.is_(None)),
            )
        ).scalar_one_or_none()

        if not existing:
            self._session.add(TeamAlias(
                canonical_name=canonical,
                alias=alias,
                source=f"auto_{self._sport.value}",
                sport=self._sport,
            ))

        self._cache[alias_key] = canonical

    def _register_canonical(self, canonical: str, raw_alias: str) -> None:
        """Register a new canonical name (first encounter in this sport)."""
        canon_key = make_key(canonical)
        self._canonicals[canon_key] = canonical
        self._cache[canon_key] = canonical

        # Persist canonical as its own alias (self-referencing)
        existing = self._session.execute(
            select(TeamAlias).where(
                TeamAlias.alias == canonical,
                TeamAlias.sport == self._sport,
            )
        ).scalar_one_or_none()

        if not existing:
            self._session.add(TeamAlias(
                canonical_name=canonical,
                alias=canonical,
                source=f"auto_{self._sport.value}",
                sport=self._sport,
            ))

        # If raw_alias differs from canonical (e.g. had accents), register it too
        raw_key = make_key(raw_alias)
        if raw_key != canon_key:
            self._cache[raw_key] = canonical
            existing2 = self._session.execute(
                select(TeamAlias).where(
                    TeamAlias.alias == raw_alias,
                    TeamAlias.sport == self._sport,
                )
            ).scalar_one_or_none()

            if not existing2:
                self._session.add(TeamAlias(
                    canonical_name=canonical,
                    alias=raw_alias,
                    source=f"auto_{self._sport.value}",
                    sport=self._sport,
                ))

    @property
    def stats(self) -> dict[str, int | str | list[str]]:
        """Return resolver statistics for logging/debugging."""
        total = self._exact_hits + self._fuzzy_hits + self._auto_registered + len(self._unresolved)
        return {
            "sport": self._sport.value,
            "aliases_cached": len(self._cache),
            "canonical_names": len(self._canonicals),
            "exact_hits": self._exact_hits,
            "fuzzy_hits": self._fuzzy_hits,
            "auto_registered": self._auto_registered,
            "unresolved_count": len(self._unresolved),
            "unresolved_names": self._unresolved[:20],  # cap for logging
            "total_lookups": total,
        }
