"""Forecast/outcome pairing and error metrics.

Pure functions: no database, no I/O. The subtle part of performance tracking is
*which* prediction an outcome belongs to, so that logic lives here where it can
be tested without infrastructure.
"""

from dataclasses import dataclass
from datetime import datetime
from math import sqrt
from typing import Iterable

# (recorded_at, target_ts, prediction)
PredictionRow = tuple[datetime, datetime | None, float]


@dataclass(frozen=True)
class Forecast:
    recorded_at: datetime
    value: float


def choose_forecast(rows: Iterable[PredictionRow]) -> dict[datetime, float]:
    """Pick the latest prediction recorded at or before each target time.

    A prediction issued after its target time is not a forecast and is ignored.
    """
    best: dict[datetime, Forecast] = {}
    for recorded_at, target_ts, value in rows:
        if target_ts is None or recorded_at > target_ts:
            continue
        current = best.get(target_ts)
        if current is None or recorded_at > current.recorded_at:
            best[target_ts] = Forecast(recorded_at=recorded_at, value=value)
    return {target: forecast.value for target, forecast in best.items()}


def summarize_performance(
    rows: Iterable[PredictionRow],
    outcomes: dict[datetime, float],
) -> dict:
    """Compute MAE, RMSE, sample count, and coverage for paired forecasts.

    `mae` and `rmse` are None when nothing paired: reporting 0.0 would read as a
    perfect model rather than as missing evidence.
    """
    rows = list(rows)
    predicted_targets = {target for _recorded, target, _value in rows if target is not None}
    forecasts = choose_forecast(rows)
    pairs = [
        (value, outcomes[target])
        for target, value in forecasts.items()
        if target in outcomes
    ]

    if pairs:
        errors = [predicted - actual for predicted, actual in pairs]
        mae: float | None = sum(abs(error) for error in errors) / len(errors)
        rmse: float | None = sqrt(sum(error * error for error in errors) / len(errors))
    else:
        mae = rmse = None

    coverage = len(pairs) / len(predicted_targets) if predicted_targets else 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "samples": len(pairs),
        "coverage": coverage,
        "predicted_targets": len(predicted_targets),
    }
