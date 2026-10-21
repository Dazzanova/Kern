from typing import Any, Iterable

import pandas as pd

RANGE_WARN, RANGE_FAIL = 0.05, 0.20
NULL_WARN_DELTA, NULL_FAIL_DELTA = 0.01, 0.05
Z_WARN, Z_FAIL = 3.0, 5.0

QUANTILES = (0.01, 0.25, 0.50, 0.75, 0.99)


def profile_reference(
    df: pd.DataFrame, feature_columns: Iterable[str]
) -> dict[str, dict[str, float]]:
    profile: dict[str, dict[str, float]] = {}

    for column in feature_columns:
        series = pd.to_numeric(df[column], errors="coerce")
        quantiles = series.quantile(list(QUANTILES))
        profile[column] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "null_rate": float(series.isna().mean()),
            "p01": float(quantiles.loc[0.01]),
            "p25": float(quantiles.loc[0.25]),
            "p50": float(quantiles.loc[0.50]),
            "p75": float(quantiles.loc[0.75]),
            "p99": float(quantiles.loc[0.99]),
        }

    return profile


def _status(value: float, warn: float, fail: float) -> str:
    if value > fail:
        return "fail"
    if value > warn:
        return "warn"
    return "pass"


def evaluate_feature(
    feature: str, reference: dict[str, float], values: list[float | None]
) -> list[dict[str, Any]]:
    observed = pd.Series(values, dtype="float64")
    present = observed.dropna()

    # An empty observation window means the feature stopped arriving entirely,
    # so treat it as fully null rather than reporting nothing.
    null_rate = 1.0 if observed.empty else float(observed.isna().mean())
    null_check = {
        "feature": feature,
        "check_type": "null_rate",
        "expected": reference["null_rate"],
        "observed": null_rate,
        "status": _status(
            null_rate - reference["null_rate"], NULL_WARN_DELTA, NULL_FAIL_DELTA
        ),
    }

    if present.empty:
        return [null_check]

    range_fraction = float(
        ((present < reference["p01"]) | (present > reference["p99"])).mean()
    )

    std = reference["std"]
    if std and std > 0:
        z_score = abs(float(present.mean()) - reference["mean"]) / std
    else:
        z_score = 0.0

    return [
        {
            "feature": feature,
            "check_type": "range",
            "expected": 0.0,
            "observed": range_fraction,
            "status": _status(range_fraction, RANGE_WARN, RANGE_FAIL),
        },
        null_check,
        {
            "feature": feature,
            "check_type": "mean_shift",
            "expected": reference["mean"],
            "observed": float(present.mean()),
            "status": _status(z_score, Z_WARN, Z_FAIL),
        },
    ]


def evaluate_features(
    reference_profile: dict[str, dict[str, float]],
    observations: dict[str, list[float | None]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for feature, reference in reference_profile.items():
        results.extend(evaluate_feature(feature, reference, observations.get(feature, [])))

    return results
