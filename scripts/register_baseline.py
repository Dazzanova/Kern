"""Register and promote the Phase 1 baseline artifact.

Requires: docker-compose up -d, and models/baseline.json plus data/train.csv
present locally (both are gitignored; regenerate with the training pipeline).
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from kern.config import get_settings  # noqa: E402
from kern.data.features import build_features  # noqa: E402
from kern.quality import profile_reference, reference_sample  # noqa: E402
from kern.registry import promote, register  # noqa: E402
from kern.storage.db import apply_migrations, create_pool  # noqa: E402
from kern.storage.objects import ObjectStore  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error  # noqa: E402
import numpy as np  # noqa: E402


def main() -> None:
    settings = get_settings()
    artifact = ROOT / "models" / "baseline.json"
    if not artifact.exists():
        raise SystemExit(f"missing artifact: {artifact}")

    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")

    features = build_features(train)
    model = XGBRegressor()
    model.load_model(artifact)
    predictions = model.predict(build_features(test))

    metrics = {
        "mae": float(mean_absolute_error(test["load_mw"], predictions)),
        "rmse": float(np.sqrt(mean_squared_error(test["load_mw"], predictions))),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
    }

    pool = create_pool(settings.database_url)
    apply_migrations(pool)
    store = ObjectStore(settings)
    store.ensure_bucket()

    record = register(
        pool,
        store,
        name=settings.model_name,
        version="v1",
        artifact_path=artifact,
        metrics=metrics,
        feature_schema=list(features.columns),
        reference_profile=profile_reference(features, features.columns),
        reference_sample=reference_sample(
            features, features.columns, limit=settings.reference_sample_size
        ),
    )
    promote(pool, name=settings.model_name, version=record.version)

    print(f"registered {record.name} {record.version} ({record.artifact_sha256[:12]})")
    print(f"metrics: {metrics}")

    pool.close()


if __name__ == "__main__":
    main()
