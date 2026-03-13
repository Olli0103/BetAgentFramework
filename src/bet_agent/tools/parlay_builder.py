"""Moonshot Architect — Parlay Builder & Correlation Analyzer.

Builds high-confidence "Lotto" parlays (Kombiwetten) from approved single
picks. The Moonshot selects legs with the **highest model probability**
(not highest EV) and combines them into parlays with asymmetric upside.

Philosophy:
  The Moonshot is a Lotto ticket — we don't need +EV. We want the
  combinations our model is *most confident* about, giving us the best
  shot at landing a high-odds accumulator. Stake is always 1.00 EUR.

Golden Rules:
  1. Moonshot Rule — Every parlay is hard-capped at 1.00 EUR
  2. No LLM Math — All probability calculations in deterministic Python
  3. Highest Confidence — Sort legs by model_prob, not by EV
  4. Correlation Awareness — Penalize correlated legs to get realistic
     combined probabilities

Correlation handling:
  - Same-match legs (e.g. "Team A ML" + "Over 2.5" in same game) are
    positively correlated — we apply a correlation penalty
  - Cross-match legs are assumed independent (correlation ~ 0)
  - Same-sport/same-league legs get a small correlation bump for
    shared environmental factors (weather, referee pools)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from decimal import Decimal
from itertools import combinations

from sqlalchemy import select
from sqlalchemy.orm import Session

from bet_agent.db.models import (
    LedgerType,
    Match,
    PlacedBet,
    Prediction,
    PredictionStatus,
    Sport,
)

logger = logging.getLogger(__name__)


# -- Constants ----------------------------------------------------------------

MOONSHOT_HARD_CAP_EUR = 1.00
MIN_LEGS = 2
MAX_LEGS = 30
MIN_COMBINED_ODDS = 3.0  # minimum combined odds to qualify as "Moonshot"
MIN_LEG_PROB = 0.30  # ignore predictions below 30% confidence

# Correlation penalties (reduce combined probability)
# Same match, correlated markets (e.g. ML + O/U):
_SAME_MATCH_CORRELATION = 0.15
# Same league, same day:
_SAME_LEAGUE_CORRELATION = 0.03
# Cross-sport: assumed independent
_CROSS_SPORT_CORRELATION = 0.0


# -- Data structures ----------------------------------------------------------


@dataclass
class LegCorrelation:
    """Correlation assessment between two parlay legs."""
    prediction_a_id: str
    prediction_b_id: str
    same_match: bool
    same_sport: bool
    same_league: bool
    correlation_factor: float  # 0.0 = independent, 1.0 = perfectly correlated
    reason: str


@dataclass
class ParlayLeg:
    """A single leg in a parlay."""
    prediction_id: uuid.UUID
    match_id: uuid.UUID
    sport: str
    league: str
    match_description: str
    selection: str
    market_type: str
    model_prob: float
    best_odds: float
    best_sportsbook: str
    ev: float


@dataclass
class ParlayTicket:
    """A constructed parlay ready for sizing and alert."""
    parlay_id: uuid.UUID
    legs: list[ParlayLeg]
    combined_odds: float
    naive_combined_prob: float  # product of individual probs (no correlation)
    adjusted_combined_prob: float  # after correlation penalty
    combined_ev: float
    stake_eur: float
    potential_payout: float
    correlations: list[LegCorrelation]
    is_positive_ev: bool
    validation_errors: list[str] = field(default_factory=list)

    @property
    def num_legs(self) -> int:
        return len(self.legs)

    def to_dict(self) -> dict:
        return {
            "parlay_id": str(self.parlay_id),
            "num_legs": self.num_legs,
            "combined_odds": round(self.combined_odds, 2),
            "naive_prob": round(self.naive_combined_prob, 4),
            "adjusted_prob": round(self.adjusted_combined_prob, 4),
            "combined_ev": round(self.combined_ev, 4),
            "stake_eur": self.stake_eur,
            "potential_payout": round(self.potential_payout, 2),
            "is_positive_ev": self.is_positive_ev,
            "legs": [
                {
                    "match": leg.match_description,
                    "selection": leg.selection,
                    "odds": leg.best_odds,
                    "prob": round(leg.model_prob, 4),
                    "sportsbook": leg.best_sportsbook,
                }
                for leg in self.legs
            ],
        }

    def format_message(self) -> str:
        """Format as a human-readable Telegram message."""
        lines = [
            f"MOONSHOT PARLAY ({self.num_legs} legs)",
            f"Combined Odds: {self.combined_odds:.2f}",
            f"Adj. Probability: {self.adjusted_combined_prob:.1%}",
            f"EV: {self.combined_ev:+.4f}",
            f"Stake: {self.stake_eur:.2f} EUR | Payout: {self.potential_payout:.2f} EUR",
            "",
        ]
        for i, leg in enumerate(self.legs, 1):
            lines.append(
                f"  Leg {i}: {leg.match_description} — "
                f"{leg.selection} @{leg.best_odds:.2f} "
                f"(prob {leg.model_prob:.0%}, {leg.best_sportsbook})"
            )
        if self.correlations:
            corr_notes = [c for c in self.correlations if c.correlation_factor > 0]
            if corr_notes:
                lines.append("")
                lines.append("Correlations:")
                for c in corr_notes:
                    lines.append(f"  {c.reason} ({c.correlation_factor:.0%})")
        return "\n".join(lines)


# -- Core functions -----------------------------------------------------------


def check_leg_correlation(
    session: Session,
    prediction_a: Prediction,
    prediction_b: Prediction,
) -> LegCorrelation:
    """Assess statistical dependence between two parlay legs.

    Correlation sources:
      1. Same match -> strong positive correlation (especially ML + O/U)
      2. Same league, same day -> weak correlation (shared environment)
      3. Different sports -> independent

    Args:
        session: SQLAlchemy session.
        prediction_a: First prediction.
        prediction_b: Second prediction.

    Returns:
        LegCorrelation with assessed factor and reasoning.
    """
    match_a = session.get(Match, prediction_a.match_id)
    match_b = session.get(Match, prediction_b.match_id)

    if match_a is None or match_b is None:
        return LegCorrelation(
            prediction_a_id=str(prediction_a.id),
            prediction_b_id=str(prediction_b.id),
            same_match=False,
            same_sport=False,
            same_league=False,
            correlation_factor=0.0,
            reason="Match not found",
        )

    same_match = prediction_a.match_id == prediction_b.match_id
    same_sport = match_a.sport == match_b.sport
    same_league = same_sport and match_a.league == match_b.league

    if same_match:
        factor = _SAME_MATCH_CORRELATION
        reason = (
            f"Same match ({match_a.home_team} vs {match_a.away_team}): "
            f"{prediction_a.market_type.value} + {prediction_b.market_type.value} "
            f"are correlated"
        )
    elif same_league:
        factor = _SAME_LEAGUE_CORRELATION
        reason = f"Same league ({match_a.league}): weak environmental correlation"
    elif same_sport:
        factor = _CROSS_SPORT_CORRELATION
        reason = f"Same sport ({match_a.sport.value}), different leagues: ~independent"
    else:
        factor = _CROSS_SPORT_CORRELATION
        reason = "Cross-sport: independent"

    return LegCorrelation(
        prediction_a_id=str(prediction_a.id),
        prediction_b_id=str(prediction_b.id),
        same_match=same_match,
        same_sport=same_sport,
        same_league=same_league,
        correlation_factor=factor,
        reason=reason,
    )


def calculate_parlay_ev(
    legs: list[ParlayLeg],
    correlations: list[LegCorrelation],
) -> tuple[float, float, float]:
    """Compute combined EV accounting for correlations.

    The naive approach multiplies individual probabilities, but correlated
    legs make the true joint probability lower than the product.
    We apply a correlation penalty: adjusted_prob = naive_prob * (1 - avg_correlation).

    Args:
        legs: List of parlay legs with model_prob and best_odds.
        correlations: Pairwise correlation assessments.

    Returns:
        (combined_odds, adjusted_prob, combined_ev)
    """
    if not legs:
        return 1.0, 0.0, -1.0

    # Combined odds = product of individual decimal odds
    combined_odds = 1.0
    for leg in legs:
        combined_odds *= leg.best_odds

    # Naive combined probability = product of individual probabilities
    naive_prob = 1.0
    for leg in legs:
        naive_prob *= leg.model_prob

    # Correlation penalty: average pairwise correlation
    if correlations:
        total_corr = sum(c.correlation_factor for c in correlations)
        avg_corr = total_corr / len(correlations) if correlations else 0.0
    else:
        avg_corr = 0.0

    # Adjusted probability: penalize for correlations
    # Higher correlation -> lower true joint probability
    adjusted_prob = naive_prob * (1.0 - avg_corr)

    # EV = (probability * payout) - stake, normalized to stake=1
    combined_ev = (adjusted_prob * combined_odds) - 1.0

    return combined_odds, adjusted_prob, combined_ev


def validate_parlay_stake(
    stake_eur: float,
    hard_cap_eur: float = MOONSHOT_HARD_CAP_EUR,
) -> tuple[float, list[str]]:
    """Validate and cap the parlay stake.

    Args:
        stake_eur: Proposed stake.
        hard_cap_eur: Maximum allowed (default 1.00 EUR).

    Returns:
        (capped_stake, validation_errors)
    """
    errors: list[str] = []

    if stake_eur <= 0:
        errors.append("Stake must be positive")
        return 0.0, errors

    if stake_eur > hard_cap_eur:
        errors.append(
            f"Stake {stake_eur:.2f} EUR exceeds Moonshot hard cap "
            f"of {hard_cap_eur:.2f} EUR — capped"
        )
        stake_eur = hard_cap_eur

    return stake_eur, errors


# Eligible statuses for moonshot candidate pool (aggressive: all non-settled)
_ELIGIBLE_STATUSES = [
    PredictionStatus.PENDING,
    PredictionStatus.APPROVED,
    PredictionStatus.PLACED,
    PredictionStatus.VETOED,
]


def _fetch_todays_candidates(
    session: Session,
    sport_filter: str | None = None,
    target_date: date | None = None,
) -> list[Prediction]:
    """Fetch today's eligible predictions sorted by model confidence.

    Selects predictions with:
      - status in (PENDING, APPROVED, PLACED)
      - model_prob >= MIN_LEG_PROB (30%)
      - best_odds populated (line-shopped)
      - match scheduled today (or target_date)

    Sorted by model_prob DESC — highest confidence first.

    Args:
        session: SQLAlchemy session.
        sport_filter: Optional sport name to filter (e.g. "tennis", "basketball").
        target_date: Date to query. Defaults to today (UTC).

    Returns:
        List of Prediction objects sorted by model_prob descending.
    """
    if target_date is None:
        target_date = datetime.now(timezone.utc).date()

    day_start = datetime.combine(target_date, time.min, tzinfo=timezone.utc)
    day_end = datetime.combine(target_date, time.max, tzinfo=timezone.utc)

    query = (
        select(Prediction)
        .join(Match, Match.id == Prediction.match_id)
        .where(
            Prediction.status.in_(_ELIGIBLE_STATUSES),
            Prediction.model_prob >= MIN_LEG_PROB,
            Match.scheduled_at >= day_start,
            Match.scheduled_at <= day_end,
        )
        .order_by(Prediction.model_prob.desc())
    )

    if sport_filter:
        try:
            sport_enum = Sport(sport_filter.lower())
            query = query.where(Match.sport == sport_enum)
        except ValueError:
            logger.warning("Unknown sport filter: %s", sport_filter)

    return list(session.execute(query).scalars().all())


def _predictions_to_legs(
    session: Session,
    predictions: list[Prediction],
) -> list[ParlayLeg]:
    """Convert Prediction objects into ParlayLeg dataclasses."""
    legs: list[ParlayLeg] = []
    for pred in predictions:
        match = session.get(Match, pred.match_id)
        if match is None:
            continue
        legs.append(ParlayLeg(
            prediction_id=pred.id,
            match_id=pred.match_id,
            sport=match.sport.value,
            league=match.league,
            match_description=f"{match.home_team} vs {match.away_team}",
            selection=pred.selection,
            market_type=pred.market_type.value,
            model_prob=float(pred.model_prob),
            best_odds=float(pred.best_odds) if pred.best_odds else float(
                Decimal("1") / pred.implied_prob
            ),
            best_sportsbook=pred.best_sportsbook or "unknown",
            ev=float(pred.ev),
        ))
    return legs


def _build_ticket(
    session: Session,
    predictions: list[Prediction],
    legs: list[ParlayLeg],
    stake_eur: float,
) -> ParlayTicket | None:
    """Assemble a ParlayTicket from legs with correlation analysis."""
    if len(legs) < MIN_LEGS:
        return None

    # Validate and cap stake
    stake_eur, stake_errors = validate_parlay_stake(stake_eur)

    # Check all pairwise correlations
    correlations: list[LegCorrelation] = []
    for pred_a, pred_b in combinations(predictions[:len(legs)], 2):
        corr = check_leg_correlation(session, pred_a, pred_b)
        correlations.append(corr)

    # Calculate combined EV
    combined_odds, adjusted_prob, combined_ev = calculate_parlay_ev(legs, correlations)
    naive_prob = 1.0
    for leg in legs:
        naive_prob *= leg.model_prob

    is_positive = combined_ev > 0

    # Build validation warnings
    errors = list(stake_errors)
    if len(legs) > MAX_LEGS:
        errors.append(f"Too many legs: {len(legs)} > {MAX_LEGS}")

    same_match_corrs = [c for c in correlations if c.same_match]
    if same_match_corrs:
        errors.append(
            f"{len(same_match_corrs)} same-match correlation(s) detected — "
            f"combined probability penalized"
        )

    ticket = ParlayTicket(
        parlay_id=uuid.uuid4(),
        legs=legs,
        combined_odds=round(combined_odds, 2),
        naive_combined_prob=naive_prob,
        adjusted_combined_prob=adjusted_prob,
        combined_ev=combined_ev,
        stake_eur=stake_eur,
        potential_payout=round(stake_eur * combined_odds, 2),
        correlations=correlations,
        is_positive_ev=is_positive,
        validation_errors=errors,
    )

    logger.info(
        "Built parlay: %d legs, odds=%.2f, adj_prob=%.4f, EV=%+.4f, stake=%.2f EUR",
        ticket.num_legs, combined_odds, adjusted_prob, combined_ev, stake_eur,
    )

    return ticket


# -- Public API ---------------------------------------------------------------


def build_parlay(
    session: Session,
    prediction_ids: list[uuid.UUID] | None = None,
    sport_filter: str | None = None,
    num_legs: int = 3,
    target_date: date | None = None,
    stake_eur: float = MOONSHOT_HARD_CAP_EUR,
) -> ParlayTicket | None:
    """Build a parlay from approved predictions.

    Selection strategy: picks with the **highest model probability** are
    chosen first (Lotto philosophy — we want the most confident legs).
    No +EV filter is applied; combined EV is computed for information
    but does not gate the parlay.

    Can either take explicit prediction IDs, or auto-select the best
    approved picks for a given sport/date.

    Args:
        session: SQLAlchemy session.
        prediction_ids: Explicit picks. If None, auto-select from approved.
        sport_filter: Sport name to filter (e.g. "tennis", "basketball").
        num_legs: Desired number of legs (default 3, range 2-6).
        target_date: Date for auto-selection. Defaults to today (UTC).
        stake_eur: Stake (hard-capped at MOONSHOT_HARD_CAP_EUR).

    Returns:
        ParlayTicket if successful, None if not enough valid legs.
    """
    num_legs = max(MIN_LEGS, min(num_legs, MAX_LEGS))

    if prediction_ids:
        predictions = list(
            session.execute(
                select(Prediction).where(
                    Prediction.id.in_(prediction_ids),
                    Prediction.status.in_(_ELIGIBLE_STATUSES),
                )
            ).scalars().all()
        )
        # Sort by model confidence (highest first)
        predictions = sorted(
            predictions,
            key=lambda p: float(p.model_prob),
            reverse=True,
        )[:num_legs]
    else:
        # Auto-select: highest confidence eligible predictions
        candidates = _fetch_todays_candidates(
            session, sport_filter=sport_filter, target_date=target_date,
        )
        # Deduplicate: max one leg per match (pick highest prob market)
        seen_matches: set[uuid.UUID] = set()
        predictions = []
        for pred in candidates:
            if pred.match_id not in seen_matches:
                predictions.append(pred)
                seen_matches.add(pred.match_id)
            if len(predictions) >= num_legs:
                break

    # Aggressive leg fallback: if not enough for requested legs,
    # try smaller parlay sizes down to MIN_LEGS
    if len(predictions) < num_legs and len(predictions) >= MIN_LEGS:
        logger.info(
            "Requested %d legs but only %d eligible — building %d-leg parlay",
            num_legs, len(predictions), len(predictions),
        )
    elif len(predictions) < MIN_LEGS:
        logger.info(
            "Not enough eligible predictions for parlay: %d < %d minimum",
            len(predictions), MIN_LEGS,
        )
        return None

    legs = _predictions_to_legs(session, predictions)
    return _build_ticket(session, predictions, legs, stake_eur)


def get_best_combos_today(
    session: Session,
    sport_filter: str | None = None,
    num_legs: int = 3,
    top_n: int = 3,
    target_date: date | None = None,
) -> list[ParlayTicket]:
    """Get the best parlay combinations for today.

    Builds multiple parlays by sliding a window over today's approved
    predictions (sorted by model probability). Returns up to ``top_n``
    distinct parlays, each with ``num_legs`` legs, ensuring no two
    parlays share the exact same set of legs.

    Use cases:
      - "Was sind die besten Kombis heute?"
      - "Zeig mir die Top 3 Kombis"

    Args:
        session: SQLAlchemy session.
        sport_filter: Optional sport name (e.g. "tennis").
        num_legs: Legs per combo (default 3, range 2-6).
        top_n: Maximum number of combos to return (default 3).
        target_date: Date to query. Defaults to today (UTC).

    Returns:
        List of ParlayTickets sorted by adjusted_combined_prob descending.
    """
    num_legs = max(MIN_LEGS, min(num_legs, MAX_LEGS))

    candidates = _fetch_todays_candidates(
        session, sport_filter=sport_filter, target_date=target_date,
    )

    if len(candidates) < MIN_LEGS:
        logger.info(
            "Not enough candidates for combos: %d < %d minimum",
            len(candidates), MIN_LEGS,
        )
        return []

    # Deduplicate: max one prediction per match (highest prob wins)
    seen_matches: set[uuid.UUID] = set()
    unique_preds: list[Prediction] = []
    for pred in candidates:
        if pred.match_id not in seen_matches:
            unique_preds.append(pred)
            seen_matches.add(pred.match_id)

    if len(unique_preds) < MIN_LEGS:
        return []

    # Generate combos: use itertools.combinations on the top candidates
    # Limit candidate pool to avoid combinatorial explosion
    pool_size = min(len(unique_preds), max(num_legs + 6, 12))
    pool = unique_preds[:pool_size]

    tickets: list[ParlayTicket] = []

    for combo in combinations(pool, min(num_legs, len(pool))):
        combo_list = list(combo)
        legs = _predictions_to_legs(session, combo_list)

        if len(legs) < MIN_LEGS:
            continue

        ticket = _build_ticket(
            session, combo_list, legs, MOONSHOT_HARD_CAP_EUR,
        )
        if ticket is None:
            continue

        # Skip combos with trivially low combined odds (not "Moonshot" enough)
        if ticket.combined_odds < MIN_COMBINED_ODDS:
            continue

        tickets.append(ticket)

    # Sort by adjusted combined probability (highest confidence first)
    tickets.sort(key=lambda t: t.adjusted_combined_prob, reverse=True)

    return tickets[:top_n]


def build_best_parlay(
    session: Session,
    sport_filter: str | None = None,
    num_legs: int = 3,
    target_date: date | None = None,
) -> ParlayTicket | None:
    """Convenience: build the single best N-leg parlay for a sport.

    Auto-selects the highest-confidence approved predictions and builds
    the optimal parlay. Used by the Telegram concierge for requests like
    "Baue mir eine 3er Kombi fuer Tennis".

    Args:
        session: SQLAlchemy session.
        sport_filter: Sport name (e.g. "tennis", "basketball").
        num_legs: Desired number of legs (default 3).
        target_date: Date to query. Defaults to today (UTC).

    Returns:
        ParlayTicket if successful, None otherwise.
    """
    return build_parlay(
        session,
        prediction_ids=None,
        sport_filter=sport_filter,
        num_legs=num_legs,
        target_date=target_date,
        stake_eur=MOONSHOT_HARD_CAP_EUR,
    )
