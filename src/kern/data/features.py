import pandas as pd


REFERENCE_TIME = pd.Timestamp("2025-01-01")

FEATURE_COLUMNS = [
    "temperature_c",
    "humidity_pct",
    "hour",
    "day_of_week",
    "is_weekend",
    "day_of_year",
    "month",
    "time_index_hours",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    timestamps = pd.to_datetime(df["timestamp"])

    features = pd.DataFrame(
        {
            "temperature_c": df["temperature_c"].astype(float),
            "humidity_pct": df["humidity_pct"].astype(float),
            "hour": timestamps.dt.hour,
            "day_of_week": timestamps.dt.dayofweek,
            "is_weekend": (timestamps.dt.dayofweek >= 5).astype(int),
            "day_of_year": timestamps.dt.dayofyear,
            "month": timestamps.dt.month,
            "time_index_hours": (
                (timestamps - REFERENCE_TIME).dt.total_seconds() / 3600
            ),
        }
    )

    return features[FEATURE_COLUMNS]
