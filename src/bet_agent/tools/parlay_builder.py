"""Moonshot Architect — Parlay Builder & Correlation Analyzer.

Builds smart +EV parlays (Kombiwetten) from approved single picks.
Analyzes statistical correlation between legs to avoid naive independence
assumptions, and enforces the 1.00 EUR hard cap on every parlay ticket.

Golden Rules:
  1. Moonshot Rule — Every parlay is hard-capped at 1.00 EUR
  2. No LLM Math — All EV/probability calculations in deterministic Python
  3. +EV Required — Never build a parlay just for high odds

Correlation handling:
  - Same-match legs (e.g. "Team A ML" + "Over 2.5" in same game) are
    positively correlated — we apply a correlation penalty
  - Cross-match legs are assumed independent (correlation ≈ 0)
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


# ── Constants ────────────────────────────────────────────────────────────

MOONSHOT_HARD_CAP_EUR = 1.00
MIN_LEGS = 2
MAX_LEGS = 6

# Correlation penalties (reduce combined probability)
# Same match, correlated markets (e.g. ML + O/U):
_SAME_MATCH_CORRELATION = 0.15
# Same league, same day:
_SAME_LEAGUE_CORRELATION = 0.03
# Cross-sport: assumed independent
_CROSS_SPORT_CORRELATION = 0.0


# ── Data structures ─────────────────────────────────────────────────────


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
            f"🎰 MOONSHOT PARLAY ({self.num_legs} legs)",
            f"Combined Odds: {self.combined_odds:.2f}",
            f"Adj. Probability: {self.adjusted_combined_prob:.1%}",
            f"EV: {self.combined_ev:+.4f}",
            f"Stake: {self.stake_eur:.2f} EUR | Payout: {self.potential_payout:.2f} EUR",
            "",
        ]
        for i, leg in enumerate(self.legs, 1):
            lines.append(
                f"  Leg {i}: {leg.match_description} — "
                f"{leg.selection} @{leg.best_odds:.2f} ({leg.best_sportsbook})"
            )
        if self.correlations:
            corr_notes = [c for c in self.correlations if c.correlation_factor > 0]
            if corr_notes:
                lines.append("")
                lines.append("⚠️ Correlations:")
                for c in corr_notes:
                    lines.append(f"  {c.reason} ({c.correlation_factor:.0%})")
        return "\n".join(lines)


# ── Core functions ───────────────────────────────────────────────────────


def check_leg_correlation(
    session: Session,
    prediction_a: Prediction,
    prediction_b: Prediction,
) -> LegCorrelation:
    """Assess statistical dependence between two parlay legs.

    Correlation sources:
      1. Same match → strong positive correlation (especially ML + O/U)
      2. Same league, same day → weak correlation (shared environment)
      3. Different sports → independent

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
    We apply a correlation penalty: adjusted_prob = naive_prob × (1 - avg_correlation).

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
    # Higher correlation → lower true joint probability
    adjusted_prob = naive_prob * (1.0 - avg_corr)

    # EV = (probability × payout) - stake, normalized to stake=1
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


def build_parlay(
    session: Session,
    prediction_ids: list[uuid.UUID] | None = None,
    sport_filter: str | None = None,
    max_legs: int = MAX_LEGS,
    min_ev: float = 0.0,
    stake_eur: float = MOONSHOT_HARD_CAP_EUR,
) -> ParlayTicket | None:
    """Build a parlay from approved predictions.

    Can either take explicit prediction IDs, or auto-select the best
    approved picks for a given sport.

    Args:
        session: SQLAlchemy session.
        prediction_ids: Explicit picks. If None, auto-select from approved.
        sport_filter: Sport name to filter (e.g. "tennis").
        max_legs: Maximum number of legs (default 6).
        min_ev: Minimum combined EV threshold (default 0.0).
        stake_eur: Stake (hard-capped at MOONSHOT_HARD_CAP_EUR).

    Returns:
        ParlayTicket if successful, None if no valid parlay can be built.
    """
    # Validate and cap stake
    stake_eur, stake_errors = validate_parlay_stake(stake_eur)

    # Get predictions
    if prediction_ids:
        predictions = list(
            session.execute(
                select(Prediction).where(
                    Prediction.id.in_(prediction_ids),
                    Prediction.status == PredictionStatus.APPROVED,
                )
            ).scalars().all()
        )
    else:
        # Auto-select: best approved predictions by EV
        query = (
            select(Prediction)
            .join(Match, Match.id == Prediction.match_id)
            .where(
                Prediction.status == PredictionStatus.APPROVED,
                Prediction.ev > 0,
            )
            .order_by(Prediction.ev.desc())
        )
        if sport_filter:
            try:
                sport_enum = Sport(sport_filter)
                query = query.where(Match.sport == sport_enum)
            except ValueError:
                logger.warning("Unknown sport filter: %s", sport_filter)

        # Filter to today's matches
        today_start = datetime.combine(date.today(), time.min, tzinfo=timezone.utc)
        today_end = datetime.combine(date.today(), time.max, tzinfo=timezone.utc)
        query = query.where(
            Match.scheduled_at >= today_start,
            Match.scheduled_at <= today_end,
        )

        predictions = list(session.execute(query).scalars().all())

    if len(predictions) < MIN_LEGS:
        logger.info(
            "Not enough approved predictions for parlay: %d < %d minimum",
            len(predictions), MIN_LEGS,
        )
        return None

    # Limit to max_legs (take highest EV)
    predictions = sorted(predictions, key=lambda p: float(p.ev), reverse=True)[:max_legs]

    # Build legs
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

    if len(legs) < MIN_LEGS:
        logger.info("Not enough valid legs: %d < %d", len(legs), MIN_LEGS)
        return None

    # Check all pairwise correlations
    correlations: list[LegCorrelation] = []
    for pred_a, pred_b in combinations(predictions, 2):
        corr = check_leg_correlation(session, pred_a, pred_b)
        correlations.append(corr)

    # Calculate combined EV
    combined_odds, adjusted_prob, combined_ev = calculate_parlay_ev(legs, correlations)
    naive_prob = 1.0
    for leg in legs:
        naive_prob *= leg.model_prob

    is_positive = combined_ev >= min_ev

    # Build validation errors
    errors = list(stake_errors)
    if not is_positive:
        errors.append(f"Combined EV {combined_ev:.4f} is below threshold {min_ev}")
    if len(legs) > MAX_LEGS:
        errors.append(f"Too many legs: {len(legs)} > {MAX_LEGS}")

    # Warn about same-match correlations
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


def build_best_parlay(
    session: Session,
    sport_filter: str | None = None,
    num_legs: int = 3,
) -> ParlayTicket | None:
    """Convenience: build the best N-leg parlay for a sport.

    Auto-selects the highest-EV approved predictions and builds
    the optimal parlay. Used by the Telegram concierge for
    "Baue mir eine 3er Kombi für Tennis" requests.

    Args:
        session: SQLAlchemy session.
        sport_filter: Sport name (e.g. "tennis").
        num_legs: Desired number of legs.

    Returns:
        ParlayTicket if successful, None otherwise.
    """
    return build_parlay(
        session,
        prediction_ids=None,
        sport_filter=sport_filter,
        max_legs=num_legs,
        stake_eur=MOONSHOT_HARD_CAP_EUR,
    )
