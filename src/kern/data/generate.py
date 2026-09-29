from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"


def generate_dataset(start: str, periods: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(
        start=start,
        periods=periods,
        freq="h",
    )

    hour = timestamps.hour.to_numpy()
    day_of_week = timestamps.dayofweek.to_numpy()
    day_of_year = timestamps.dayofyear.to_numpy()

    is_weekend = (day_of_week >= 5).astype(int)

    temperature = (
        22
        + 7 * np.sin(2 * np.pi * (hour - 14) / 24)
        + 4 * np.sin(2 * np.pi * day_of_year / 365)
        + rng.normal(0, 1.2, periods)
    )

    humidity = 60 - 0.4 * (temperature - 22) + rng.normal(0, 4, periods)

    daily_cycle = 100 * np.sin(2 * np.pi * (hour - 7) / 24) + 60 * np.sin(
        2 * np.pi * (hour - 18) / 24
    )

    weekly_cycle = np.where(is_weekend == 1, -80, 50)

    temperature_effect = 5 * (temperature - 20) ** 2

    trend = np.linspace(0, 120, periods)

    noise = rng.normal(0, 25, periods)

    load = 900 + daily_cycle + weekly_cycle + temperature_effect + trend + noise

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature_c": temperature,
            "humidity_pct": humidity,
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "load_mw": load,
        }
    )


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    train = generate_dataset(
        start="2025-01-01",
        periods=24 * 90,
        seed=42,
    )

    test = generate_dataset(
        start="2025-04-01",
        periods=24 * 30,
        seed=43,
    )

    train.to_csv(DATA_DIR / "train.csv", index=False)
    test.to_csv(DATA_DIR / "test.csv", index=False)

    print(f"Train: {train.shape}")
    print(f"Test:  {test.shape}")
    print(train.head())


if __name__ == "__main__":
    main()
