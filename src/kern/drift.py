"""Distribution drift statistics.

Pure functions: no database, no I/O. The thresholds are module constants so they
are easy to inspect and tune in one place, matching the approach in `quality.py`.
"""

import numpy as np
from scipy.stats import ks_2samp

PSI_WARN, PSI_FAIL = 0.10, 0.25
KS_WARN, KS_FAIL = 0.05, 0.01
PSI_EPSILON = 1e-6

# Deterministic functions of the target timestamp. Over a short window these take
# a single value (every 03:00 request has hour=3, every April request has month=4),
# so comparing them with the training distribution measures the clock rather than
# input drift and would alarm permanently. They are excluded from drift checks.
TIME_DERIVED_FEATURES = frozenset(
    {
        "hour",
        "day_of_week",
        "is_weekend",
        "day_of_year",
        "month",
        "time_index_hours",
    }
)


def psi(reference, observed, bins: int = 10) -> float:
    """Population Stability Index between a reference and an observed sample.

    Bin edges come from the reference deciles, with infinite outer edges so
    observations outside the training range are counted rather than dropped.
    """
    observed = np.asarray(observed, dtype=float)
    if observed.size == 0:
        raise ValueError("observed sample is empty")

    reference = np.asarray(reference, dtype=float)
    edges = np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    # Duplicate quantiles (a heavily skewed feature) would produce zero-width
    # bins; collapsing them keeps the histogram well defined.
    edges = np.concatenate(([-np.inf], np.unique(edges), [np.inf]))

    reference_counts, _ = np.histogram(reference, bins=edges)
    observed_counts, _ = np.histogram(observed, bins=edges)

    reference_share = np.clip(reference_counts / max(reference_counts.sum(), 1), PSI_EPSILON, None)
    observed_share = np.clip(observed_counts / max(observed_counts.sum(), 1), PSI_EPSILON, None)

    return float(np.sum((observed_share - reference_share) * np.log(observed_share / reference_share)))


def ks(reference, observed) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov statistic and p-value."""
    observed = np.asarray(observed, dtype=float)
    if observed.size == 0:
        raise ValueError("observed sample is empty")

    result = ks_2samp(np.asarray(reference, dtype=float), observed)
    return float(result.statistic), float(result.pvalue)


def psi_status(value: float) -> str:
    if value > PSI_FAIL:
        return "fail"
    if value > PSI_WARN:
        return "warn"
    return "pass"


def ks_status(p_value: float) -> str:
    if p_value < KS_FAIL:
        return "fail"
    if p_value < KS_WARN:
        return "warn"
    return "pass"


def overall_status(checks: list[dict]) -> str:
    """Worst PSI verdict across features.

    PSI drives the verdict; KS is supporting context. At high traffic volume KS
    flags almost any difference as significant, so letting it decide would make
    the status a function of sample size rather than of drift.
    """
    order = {"pass": 0, "warn": 1, "fail": 2}
    statuses = [check["status"] for check in checks if check["method"] == "psi" and check["status"] in order]
    if not statuses:
        return "insufficient_data"
    return max(statuses, key=lambda status: order[status])
