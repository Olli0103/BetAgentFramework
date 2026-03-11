"""Daily Prediction Runner — generates predictions for today's matches.

Called by the Master Agent every morning after the Scout's crawl.
For each NOT_STARTED match, generates ML or analytical predictions
and stores them in the predictions table.

Pipeline:
  1. Query matches table for today's NOT_STARTED matches
  2. For each match: try ML model → fall back to analytical model
  3. Filter for +EV opportunities
  4. Store predictions (idempotent: upsert on unique constraint)

Golden Rule #1: NO LLM MATH.  Delegates to XGBoost or analytical models.
Golden Rule #2: STATEFUL MEMORY.  All predictions go to PostgreSQL.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timezone
from decimal import Decimal
from pathlib import Path
from typing import Sequence

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from bet_agent.db.models import (
    MarketType,
    Match,
    MatchState,
    OddsMarket,
    Prediction,
    PredictionStatus,
    Sport,
)

logger = logging.getLogger(__name__)


def run_daily_predictions(
    session: Session,
    prediction_date: date | None = None,
    model_dir: Path | None = None,
    min_ev: float = 0.0,
) -> list[Prediction]:
    """Generate predictions for all of today's upcoming matches.

    Args:
        session: SQLAlchemy session (caller manages transaction).
        prediction_date: Date to generate predictions for (defaults to today).
        model_dir: Directory containing trained ML models.
        min_ev: Minimum EV threshold to store a prediction (default: store all).

    Returns:
        List of Prediction objects created/updated.
    """
    if prediction_date is None:
        prediction_date = date.today()

    # Query today's NOT_STARTED matches
    day_start = datetime.combine(prediction_date, time.min, tzinfo=timezone.utc)
    day_end = datetime.combine(prediction_date, time.max, tzinfo=timezone.utc)

    matches: Sequence[Match] = session.execute(
        select(Match)
        .where(
            Match.match_state == MatchState.NOT_STARTED,
            Match.scheduled_at >= day_start,
            Match.scheduled_at <= day_end,
        )
        .order_by(Match.scheduled_at)
    ).scalars().all()

    logger.info(
        "Found %d NOT_STARTED matches for %s", len(matches), prediction_date,
    )

    # Bulk-load all odds for today's matches (eliminates N+1 query)
    match_ids = [m.id for m in matches]
    bulk_odds = _bulk_load_odds(session, match_ids) if match_ids else {}

    all_predictions: list[Prediction] = []

    for match in matches:
        try:
            odds_map = bulk_odds.get(match.id, {})
            preds = _predict_match(session, match, prediction_date, model_dir, odds_map)
            for pred in preds:
                if pred.ev >= Decimal(str(min_ev)):
                    _upsert_prediction(session, pred)
                    all_predictions.append(pred)
        except Exception as exc:
            logger.warning(
                "Failed to predict %s vs %s: %s",
                match.home_team, match.away_team, exc,
            )

    session.flush()
    logger.info(
        "Generated %d predictions for %s (%d matches)",
        len(all_predictions), prediction_date, len(matches),
    )
    return all_predictions


def _predict_match(
    session: Session,
    match: Match,
    prediction_date: date,
    model_dir: Path | None,
    odds_map: dict[str, float] | None = None,
) -> list[Prediction]:
    """Generate all market predictions for a single match."""
    predictions: list[Prediction] = []

    # Use pre-loaded odds or fall back to per-match query
    if odds_map is None:
        odds_map = _get_match_odds(session, match)

    # Try ML prediction first
    ml_preds = _try_ml_prediction(session, match, prediction_date, odds_map, model_dir)
    if ml_preds:
        predictions.extend(ml_preds)
    else:
        # Fall back to analytical model
        analytical_preds = _try_analytical_prediction(session, match, prediction_date, odds_map)
        predictions.extend(analytical_preds)

    return predictions


# ── ML Prediction ────────────────────────────────────────────────────


def _try_ml_prediction(
    session: Session,
    match: Match,
    prediction_date: date,
    odds_map: dict[str, float],
    model_dir: Path | None,
) -> list[Prediction]:
    """Try to generate predictions using trained XGBoost model."""
    from bet_agent.ml.trainer import find_latest_model

    artifact = find_latest_model(match.sport, "match_winner", model_dir)
    if artifact is None:
        return []

    from bet_agent.tools.ev_calculator import calculate_ml_pre_match_ev

    odds_home = odds_map.get("home")
    odds_draw = odds_map.get("draw")
    odds_away = odds_map.get("away")

    if odds_home is None:
        return []

    ev_results = calculate_ml_pre_match_ev(
        session,
        match.sport.value,
        match.home_team,
        match.away_team,
        prediction_date,
        odds_home,
        odds_draw,
        odds_away,
        model_dir,
    )

    predictions: list[Prediction] = []
    selection_map = {"home": "home", "draw": "draw", "away": "away"}

    for market, ev_result in ev_results.items():
        predictions.append(Prediction(
            match_id=match.id,
            model_name=artifact.model_name,
            market_type=MarketType.MATCH_WINNER,
            selection=selection_map[market],
            model_prob=Decimal(str(ev_result.updated_prob)),
            implied_prob=Decimal(str(ev_result.implied_prob)),
            prob_edge=Decimal(str(ev_result.prob_edge)),
            ev=Decimal(str(ev_result.ev)),
            model_source=ev_result.model_source,
            status=PredictionStatus.PENDING,
        ))

    # Also try over/under model
    ou_preds = _try_ml_over_under(session, match, prediction_date, odds_map, model_dir)
    predictions.extend(ou_preds)

    return predictions


def _try_ml_over_under(
    session: Session,
    match: Match,
    prediction_date: date,
    odds_map: dict[str, float],
    model_dir: Path | None,
) -> list[Prediction]:
    """Try over/under prediction using XGBoost regressor."""
    from bet_agent.ml.trainer import find_latest_model, predict_total
    from bet_agent.tools.feature_factory import build_feature_vector, get_feature_names

    import numpy as np

    artifact = find_latest_model(match.sport, "over_under", model_dir)
    if artifact is None:
        return []

    ou_line = odds_map.get("ou_line")
    odds_over = odds_map.get("over")
    odds_under = odds_map.get("under")
    if ou_line is None or odds_over is None:
        return []

    fv = build_feature_vector(session, match.sport, match.home_team, match.away_team, prediction_date)
    feature_names = get_feature_names(match.sport)
    X = np.zeros((1, len(feature_names)))
    for j, fname in enumerate(feature_names):
        X[0, j] = fv.features.get(fname, 0.0)

    expected_total = predict_total(artifact.file_path, X)

    # Crude probability: if expected_total > line, favor over
    # Use a simple normal approximation with std ≈ 15% of expected
    from scipy.stats import norm
    std = max(expected_total * 0.15, 1.0)
    p_over = float(1.0 - norm.cdf(ou_line, loc=expected_total, scale=std))
    p_under = 1.0 - p_over

    predictions: list[Prediction] = []

    if odds_over > 1.0:
        implied = 1.0 / odds_over
        edge = p_over - implied
        ev = (p_over * (odds_over - 1.0)) - (1.0 - p_over)
        predictions.append(Prediction(
            match_id=match.id,
            model_name=artifact.model_name,
            market_type=MarketType.OVER_UNDER,
            selection=f"over_{ou_line}",
            model_prob=Decimal(str(round(p_over, 6))),
            implied_prob=Decimal(str(round(implied, 6))),
            prob_edge=Decimal(str(round(edge, 6))),
            ev=Decimal(str(round(ev, 6))),
            model_source="xgboost",
            status=PredictionStatus.PENDING,
        ))

    if odds_under is not None and odds_under > 1.0:
        implied = 1.0 / odds_under
        edge = p_under - implied
        ev = (p_under * (odds_under - 1.0)) - (1.0 - p_under)
        predictions.append(Prediction(
            match_id=match.id,
            model_name=artifact.model_name,
            market_type=MarketType.OVER_UNDER,
            selection=f"under_{ou_line}",
            model_prob=Decimal(str(round(p_under, 6))),
            implied_prob=Decimal(str(round(implied, 6))),
            prob_edge=Decimal(str(round(edge, 6))),
            ev=Decimal(str(round(ev, 6))),
            model_source="xgboost",
            status=PredictionStatus.PENDING,
        ))

    return predictions


# ── Analytical Prediction ────────────────────────────────────────────


def _try_analytical_prediction(
    session: Session,
    match: Match,
    prediction_date: date,
    odds_map: dict[str, float],
) -> list[Prediction]:
    """Generate predictions using analytical probability models + param estimator."""
    from bet_agent.tools.param_estimator import estimate_params
    from bet_agent.tools.prob_models.registry import get_model

    try:
        model = get_model(match.sport.value)
    except KeyError:
        logger.warning("No analytical model for sport %s", match.sport.value)
        return []

    params = estimate_params(session, match.sport, match.home_team, match.away_team, prediction_date)
    model_name = f"analytical_{match.sport.value}"

    predictions: list[Prediction] = []

    # Match winner probabilities
    try:
        probs = model.match_outcome_probs(**params.params)
    except TypeError:
        logger.warning("Param mismatch for %s model: %s", match.sport.value, params.params)
        return []

    odds_home = odds_map.get("home")
    odds_draw = odds_map.get("draw")
    odds_away = odds_map.get("away")

    for selection, prob_key, odds_val in [
        ("home", "home", odds_home),
        ("draw", "draw", odds_draw),
        ("away", "away", odds_away),
    ]:
        if odds_val is None or odds_val <= 1.0:
            continue
        prob = probs.get(prob_key, 0.0)
        if prob <= 0.0:
            continue

        implied = 1.0 / odds_val
        edge = prob - implied
        ev = (prob * (odds_val - 1.0)) - (1.0 - prob)

        predictions.append(Prediction(
            match_id=match.id,
            model_name=model_name,
            market_type=MarketType.MATCH_WINNER,
            selection=selection,
            model_prob=Decimal(str(round(prob, 6))),
            implied_prob=Decimal(str(round(implied, 6))),
            prob_edge=Decimal(str(round(edge, 6))),
            ev=Decimal(str(round(ev, 6))),
            model_source="analytical",
            status=PredictionStatus.PENDING,
        ))

    # Over/under analytical prediction
    ou_line = odds_map.get("ou_line")
    odds_over = odds_map.get("over")
    if ou_line is not None and odds_over is not None and odds_over > 1.0:
        try:
            p_over = model.over_under_prob(**params.params, line=ou_line)
            implied = 1.0 / odds_over
            edge = p_over - implied
            ev = (p_over * (odds_over - 1.0)) - (1.0 - p_over)

            predictions.append(Prediction(
                match_id=match.id,
                model_name=model_name,
                market_type=MarketType.OVER_UNDER,
                selection=f"over_{ou_line}",
                model_prob=Decimal(str(round(p_over, 6))),
                implied_prob=Decimal(str(round(implied, 6))),
                prob_edge=Decimal(str(round(edge, 6))),
                ev=Decimal(str(round(ev, 6))),
                model_source="analytical",
                status=PredictionStatus.PENDING,
            ))
        except (TypeError, ValueError):
            pass  # Model doesn't support these params for O/U

    return predictions


# ── Helpers ──────────────────────────────────────────────────────────


def _bulk_load_odds(
    session: Session,
    match_ids: list,
) -> dict[object, dict[str, float]]:
    """Bulk-load pre-match odds for multiple matches in a single query.

    Returns a dict keyed by match_id → odds_map (same format as _get_match_odds).
    This eliminates the N+1 query pattern when generating daily predictions.
    """
    from collections import defaultdict

    if not match_ids:
        return {}

    all_odds = session.execute(
        select(OddsMarket)
        .where(
            OddsMarket.match_id.in_(match_ids),
            OddsMarket.is_live == False,
        )
        .order_by(OddsMarket.scraped_at.desc())
    ).scalars().all()

    # Group by match_id
    by_match: dict[object, list[OddsMarket]] = defaultdict(list)
    for row in all_odds:
        by_match[row.match_id].append(row)

    # Build odds_map per match (same logic as _get_match_odds)
    result: dict[object, dict[str, float]] = {}
    for mid, rows in by_match.items():
        odds_map: dict[str, float] = {}
        for row in rows:
            sel = row.selection.lower()
            odds = float(row.odds_decimal)

            if row.market_type == MarketType.MATCH_WINNER:
                if sel == "home" and "home" not in odds_map:
                    odds_map["home"] = odds
                elif sel == "draw" and "draw" not in odds_map:
                    odds_map["draw"] = odds
                elif sel == "away" and "away" not in odds_map:
                    odds_map["away"] = odds
            elif row.market_type == MarketType.OVER_UNDER:
                if sel.startswith("over") and "over" not in odds_map:
                    odds_map["over"] = odds
                    parts = sel.split("_", 1)
                    if len(parts) > 1:
                        try:
                            odds_map["ou_line"] = float(parts[1])
                        except ValueError:
                            pass
                elif sel.startswith("under") and "under" not in odds_map:
                    odds_map["under"] = odds
        result[mid] = odds_map

    return result


def _get_match_odds(session: Session, match: Match) -> dict[str, float]:
    """Get the best available odds for a match from odds_markets.

    Returns a dict like:
        {"home": 2.10, "draw": 3.50, "away": 3.80, "ou_line": 2.5, "over": 1.90, "under": 1.95}
    """
    odds_rows = session.execute(
        select(OddsMarket)
        .where(
            OddsMarket.match_id == match.id,
            OddsMarket.is_live == False,
        )
        .order_by(OddsMarket.scraped_at.desc())
    ).scalars().all()

    result: dict[str, float] = {}

    for row in odds_rows:
        sel = row.selection.lower()
        odds = float(row.odds_decimal)

        if row.market_type == MarketType.MATCH_WINNER:
            if sel == "home" and "home" not in result:
                result["home"] = odds
            elif sel == "draw" and "draw" not in result:
                result["draw"] = odds
            elif sel == "away" and "away" not in result:
                result["away"] = odds
        elif row.market_type == MarketType.OVER_UNDER:
            if sel.startswith("over") and "over" not in result:
                result["over"] = odds
                # Extract line from selection like "over_2.5"
                parts = sel.split("_", 1)
                if len(parts) > 1:
                    try:
                        result["ou_line"] = float(parts[1])
                    except ValueError:
                        pass
            elif sel.startswith("under") and "under" not in result:
                result["under"] = odds

    return result


def _upsert_prediction(session: Session, pred: Prediction) -> None:
    """Insert or update a prediction (idempotent)."""
    existing = session.execute(
        select(Prediction).where(
            Prediction.match_id == pred.match_id,
            Prediction.model_name == pred.model_name,
            Prediction.market_type == pred.market_type,
            Prediction.selection == pred.selection,
        )
    ).scalar_one_or_none()

    if existing:
        # Update probabilities but don't regress pipeline status
        existing.model_prob = pred.model_prob
        existing.implied_prob = pred.implied_prob
        existing.prob_edge = pred.prob_edge
        existing.ev = pred.ev
        existing.model_source = pred.model_source
    else:
        session.add(pred)


# ── Query Helpers (for other agents) ─────────────────────────────────


def get_positive_ev_predictions(
    session: Session,
    prediction_date: date | None = None,
    sport: Sport | None = None,
    min_ev: float = 0.0,
    status: PredictionStatus = PredictionStatus.PENDING,
) -> list[Prediction]:
    """Query predictions with positive EV for a given date.

    Used by the Master Agent to find actionable opportunities.
    """
    if prediction_date is None:
        prediction_date = date.today()

    day_start = datetime.combine(prediction_date, time.min, tzinfo=timezone.utc)
    day_end = datetime.combine(prediction_date, time.max, tzinfo=timezone.utc)

    query = (
        select(Prediction)
        .join(Match)
        .where(
            Prediction.status == status,
            Prediction.ev > Decimal(str(min_ev)),
            Match.scheduled_at >= day_start,
            Match.scheduled_at <= day_end,
        )
        .order_by(Prediction.ev.desc())
    )

    if sport is not None:
        query = query.where(Match.sport == sport)

    return list(session.execute(query).scalars().all())


def update_prediction_status(
    session: Session,
    prediction_id,
    new_status: PredictionStatus,
) -> None:
    """Update a prediction's pipeline status (e.g., approved → placed)."""
    pred = session.get(Prediction, prediction_id)
    if pred:
        pred.status = new_status
