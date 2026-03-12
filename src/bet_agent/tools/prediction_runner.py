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

# ── Probability clipping to prevent saturation (prob=0 or prob=1) ────
_PROB_MIN = 1e-6
_PROB_MAX = 1.0 - 1e-6


def _clip_prob(p: float) -> float:
    """Clip probability to safe range [1e-6, 1-1e-6]."""
    return max(_PROB_MIN, min(p, _PROB_MAX))


# ── Minimum team name length to reject obvious abbreviations ────────
_MIN_TEAM_NAME_LEN = 4


def _is_abbreviation(name: str) -> bool:
    """Detect if a name is likely an abbreviation (all uppercase, < 5 chars)."""
    stripped = name.strip()
    if len(stripped) < _MIN_TEAM_NAME_LEN:
        return True
    # Pure uppercase abbreviation like "CHA", "BOS", "MEM" (3-4 chars, all caps)
    if len(stripped) <= 4 and stripped.isupper():
        return True
    return False


def validate_fixture(match: Match, odds_map: dict) -> tuple[bool, str]:
    """Validate a fixture is ready for prediction.

    Returns (is_valid, reason). Rejected fixtures are logged, not predicted.

    Checks:
      1. Team name quality (not abbreviations)
      2. Valid scheduled_at
      3. League populated
      4. Odds completeness (match_winner home odds required)
      5. Odds range sanity (> 1.0)
    """
    # 1. Team name quality — reject obvious abbreviations
    if _is_abbreviation(match.home_team):
        return False, f"home_team looks like abbreviation: '{match.home_team}'"
    if _is_abbreviation(match.away_team):
        return False, f"away_team looks like abbreviation: '{match.away_team}'"

    # 2. Valid scheduled_at
    if match.scheduled_at is None:
        return False, "missing scheduled_at"

    # 3. League populated
    if not match.league or not match.league.strip():
        return False, "missing league"

    # 4. Odds completeness — need at least match_winner home odds
    mw = odds_map.get("match_winner", {})
    if not mw or mw.get("home") is None:
        return False, "no match_winner odds available"

    # 5. Odds range sanity (should be decimal > 1.0 after normalization)
    for sel, val in mw.items():
        if val is not None and val <= 1.0:
            return False, f"invalid odds for {sel}: {val}"

    return True, ""


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
    rejected_count = 0

    for match in matches:
        try:
            odds_map = bulk_odds.get(match.id, {})

            # ── Fixture validity gate ─────────────────────────────
            is_valid, reject_reason = validate_fixture(match, odds_map)
            if not is_valid:
                logger.warning(
                    "Fixture rejected: %s vs %s — %s",
                    match.home_team, match.away_team, reject_reason,
                )
                rejected_count += 1
                continue

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
        "Generated %d predictions for %s (%d matches, %d rejected by fixture gate)",
        len(all_predictions), prediction_date, len(matches), rejected_count,
    )
    return all_predictions


def _predict_match(
    session: Session,
    match: Match,
    prediction_date: date,
    model_dir: Path | None,
    odds_map: dict | None = None,
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
    odds_map: dict,
    model_dir: Path | None,
) -> list[Prediction]:
    """Try to generate predictions using trained XGBoost model."""
    from bet_agent.ml.trainer import find_latest_model

    artifact = find_latest_model(match.sport, "match_winner", model_dir)
    if artifact is None:
        return []

    from bet_agent.tools.ev_calculator import calculate_ml_pre_match_ev

    mw_odds = odds_map.get("match_winner", {})
    odds_home = mw_odds.get("home")
    odds_draw = mw_odds.get("draw")
    odds_away = mw_odds.get("away")

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
            model_prob=Decimal(str(_clip_prob(ev_result.updated_prob))),
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
    odds_map: dict,
    model_dir: Path | None,
) -> list[Prediction]:
    """Try over/under prediction using XGBoost regressor.

    Iterates ALL available O/U lines and generates both Over AND Under
    predictions for each. The expected total from the ML model is used
    with a normal approximation to derive P(over) and P(under) per line.
    """
    from bet_agent.ml.trainer import find_latest_model, predict_total
    from bet_agent.tools.feature_factory import build_feature_vector, get_feature_names

    import numpy as np

    artifact = find_latest_model(match.sport, "over_under", model_dir)
    if artifact is None:
        return []

    ou_markets = odds_map.get("over_under", {})
    if not ou_markets:
        return []

    fv = build_feature_vector(session, match.sport, match.home_team, match.away_team, prediction_date)
    # Use feature names from the trained model artifact (not static fallback)
    # to ensure the feature vector matches what the model was trained on.
    feature_names = artifact.feature_names or get_feature_names(match.sport)
    X = np.zeros((1, len(feature_names)))
    for j, fname in enumerate(feature_names):
        X[0, j] = fv.features.get(fname, 0.0)

    expected_total = predict_total(artifact.file_path, X)

    from scipy.stats import norm
    std = max(expected_total * 0.15, 1.0)

    predictions: list[Prediction] = []

    for line, line_odds in ou_markets.items():
        p_over = float(1.0 - norm.cdf(line, loc=expected_total, scale=std))
        p_under = 1.0 - p_over

        # Over prediction
        odds_over = line_odds.get("over")
        if odds_over is not None and odds_over > 1.0:
            implied = 1.0 / odds_over
            edge = p_over - implied
            ev = (p_over * (odds_over - 1.0)) - (1.0 - p_over)
            predictions.append(Prediction(
                match_id=match.id,
                model_name=artifact.model_name,
                market_type=MarketType.OVER_UNDER,
                selection=f"over_{line}",
                model_prob=Decimal(str(round(_clip_prob(p_over), 6))),
                implied_prob=Decimal(str(round(implied, 6))),
                prob_edge=Decimal(str(round(edge, 6))),
                ev=Decimal(str(round(ev, 6))),
                model_source="xgboost",
                status=PredictionStatus.PENDING,
            ))

        # Under prediction
        odds_under = line_odds.get("under")
        if odds_under is not None and odds_under > 1.0:
            implied = 1.0 / odds_under
            edge = p_under - implied
            ev = (p_under * (odds_under - 1.0)) - (1.0 - p_under)
            predictions.append(Prediction(
                match_id=match.id,
                model_name=artifact.model_name,
                market_type=MarketType.OVER_UNDER,
                selection=f"under_{line}",
                model_prob=Decimal(str(round(_clip_prob(p_under), 6))),
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
    odds_map: dict,
) -> list[Prediction]:
    """Generate predictions using analytical probability models + param estimator.

    Iterates ALL available O/U lines and generates both Over AND Under
    predictions for each, using the analytical model's over_under_prob().
    """
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

    mw_odds = odds_map.get("match_winner", {})
    odds_home = mw_odds.get("home")
    odds_draw = mw_odds.get("draw")
    odds_away = mw_odds.get("away")

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
            model_prob=Decimal(str(round(_clip_prob(prob), 6))),
            implied_prob=Decimal(str(round(implied, 6))),
            prob_edge=Decimal(str(round(edge, 6))),
            ev=Decimal(str(round(ev, 6))),
            model_source="analytical",
            status=PredictionStatus.PENDING,
        ))

    # Over/Under analytical predictions — iterate ALL available lines
    ou_markets = odds_map.get("over_under", {})
    for line, line_odds in ou_markets.items():
        try:
            p_over = model.over_under_prob(**params.params, line=line)
            p_under = 1.0 - p_over

            # Over prediction
            odds_over = line_odds.get("over")
            if odds_over is not None and odds_over > 1.0:
                implied = 1.0 / odds_over
                edge = p_over - implied
                ev = (p_over * (odds_over - 1.0)) - (1.0 - p_over)

                predictions.append(Prediction(
                    match_id=match.id,
                    model_name=model_name,
                    market_type=MarketType.OVER_UNDER,
                    selection=f"over_{line}",
                    model_prob=Decimal(str(round(_clip_prob(p_over), 6))),
                    implied_prob=Decimal(str(round(implied, 6))),
                    prob_edge=Decimal(str(round(edge, 6))),
                    ev=Decimal(str(round(ev, 6))),
                    model_source="analytical",
                    status=PredictionStatus.PENDING,
                ))

            # Under prediction (the missing link!)
            odds_under = line_odds.get("under")
            if odds_under is not None and odds_under > 1.0:
                implied = 1.0 / odds_under
                edge = p_under - implied
                ev = (p_under * (odds_under - 1.0)) - (1.0 - p_under)

                predictions.append(Prediction(
                    match_id=match.id,
                    model_name=model_name,
                    market_type=MarketType.OVER_UNDER,
                    selection=f"under_{line}",
                    model_prob=Decimal(str(round(_clip_prob(p_under), 6))),
                    implied_prob=Decimal(str(round(implied, 6))),
                    prob_edge=Decimal(str(round(edge, 6))),
                    ev=Decimal(str(round(ev, 6))),
                    model_source="analytical",
                    status=PredictionStatus.PENDING,
                ))
        except (TypeError, ValueError):
            continue  # Model doesn't support O/U for these params

    return predictions


# ── Helpers ──────────────────────────────────────────────────────────


def _bulk_load_odds(
    session: Session,
    match_ids: list,
) -> dict[object, dict]:
    """Bulk-load pre-match odds for multiple matches in a single query.

    Returns a dict keyed by match_id → nested odds_map:
        {
            "match_winner": {"home": 2.10, "draw": 3.50, "away": 3.80},
            "over_under": {2.5: {"over": 1.85, "under": 1.95}, ...}
        }

    On PostgreSQL: uses DISTINCT ON to fetch only the latest odds per
    (match, market_type, selection) — loads O(matches) rows, not O(scrapes).
    On SQLite (tests): falls back to full load + Python-side dedup.
    """
    from collections import defaultdict

    if not match_ids:
        return {}

    # Detect dialect for DISTINCT ON support (PostgreSQL only)
    bind = session.get_bind()
    is_postgres = bind.dialect.name == "postgresql" if bind else False

    if is_postgres:
        # PostgreSQL: DISTINCT ON returns exactly 1 row per (match, market, selection)
        all_odds = session.execute(
            select(OddsMarket)
            .distinct(OddsMarket.match_id, OddsMarket.market_type, OddsMarket.selection)
            .where(
                OddsMarket.match_id.in_(match_ids),
                OddsMarket.is_live == False,
            )
            .order_by(
                OddsMarket.match_id,
                OddsMarket.market_type,
                OddsMarket.selection,
                OddsMarket.scraped_at.desc(),
            )
        ).scalars().all()
    else:
        # SQLite: load all, dedup in Python (fine for small test datasets)
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

    # Build nested odds_map per match:
    # {
    #   "match_winner": {"home": 2.10, "draw": 3.50, "away": 3.80},
    #   "over_under": {2.5: {"over": 1.85, "under": 1.95}, 3.5: {"over": ...}}
    # }
    # Dedup logic: first seen = latest due to ORDER BY scraped_at DESC.
    result: dict[object, dict] = {}
    for mid, rows in by_match.items():
        odds_map: dict = {"match_winner": {}, "over_under": {}}
        for row in rows:
            sel = row.selection.lower()
            raw_odds = float(row.odds_decimal)
            # Guard: reject invalid odds at read time (should not happen with CHECK constraint)
            if raw_odds <= 1.0:
                logger.warning("Skipping invalid odds %.4f for %s", raw_odds, sel)
                continue
            odds = raw_odds

            if row.market_type == MarketType.MATCH_WINNER:
                if sel in ("home", "draw", "away") and sel not in odds_map["match_winner"]:
                    odds_map["match_winner"][sel] = odds

            elif row.market_type == MarketType.OVER_UNDER:
                parts = sel.split("_", 1)
                if len(parts) == 2:
                    direction = parts[0]  # "over" or "under"
                    try:
                        line = float(parts[1])
                    except ValueError:
                        continue
                    if line not in odds_map["over_under"]:
                        odds_map["over_under"][line] = {}
                    if direction not in odds_map["over_under"][line]:
                        odds_map["over_under"][line][direction] = odds

        result[mid] = odds_map

    return result


def _get_match_odds(session: Session, match: Match) -> dict:
    """Get the best available odds for a match from odds_markets.

    Returns a nested dict:
        {
            "match_winner": {"home": 2.10, "draw": 3.50, "away": 3.80},
            "over_under": {
                2.5: {"over": 1.90, "under": 1.95},
                3.5: {"over": 2.40, "under": 1.55},
            }
        }
    """
    odds_rows = session.execute(
        select(OddsMarket)
        .where(
            OddsMarket.match_id == match.id,
            OddsMarket.is_live == False,
        )
        .order_by(OddsMarket.scraped_at.desc())
    ).scalars().all()

    result: dict = {"match_winner": {}, "over_under": {}}

    for row in odds_rows:
        sel = row.selection.lower()
        odds = float(row.odds_decimal)

        if row.market_type == MarketType.MATCH_WINNER:
            if sel in ("home", "draw", "away") and sel not in result["match_winner"]:
                result["match_winner"][sel] = odds

        elif row.market_type == MarketType.OVER_UNDER:
            parts = sel.split("_", 1)
            if len(parts) == 2:
                direction = parts[0]
                try:
                    line = float(parts[1])
                except ValueError:
                    continue
                if line not in result["over_under"]:
                    result["over_under"][line] = {}
                if direction not in result["over_under"][line]:
                    result["over_under"][line][direction] = odds

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


def get_todays_actionable_predictions(
    session: Session,
    prediction_date: date | None = None,
) -> list[Prediction]:
    """Query today's APPROVED and PLACED predictions for daily summary.

    Unlike get_positive_ev_predictions (which defaults to PENDING),
    this returns predictions that have already passed the veto+sizing pipeline
    and are ready for or have been executed.
    """
    if prediction_date is None:
        prediction_date = date.today()

    day_start = datetime.combine(prediction_date, time.min, tzinfo=timezone.utc)
    day_end = datetime.combine(prediction_date, time.max, tzinfo=timezone.utc)

    query = (
        select(Prediction)
        .join(Match)
        .where(
            Prediction.status.in_([PredictionStatus.APPROVED, PredictionStatus.PLACED]),
            Prediction.ev > Decimal("0"),
            Match.scheduled_at >= day_start,
            Match.scheduled_at <= day_end,
        )
        .order_by(Prediction.ev.desc())
    )

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
