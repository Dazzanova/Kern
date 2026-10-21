"""Backfill `reference_sample` for model versions registered before Phase 3b.

The sample is derived metadata, not artifact identity: the digest and the stored
artifact are untouched, so existing versions do not need re-registering.

Requires: docker-compose up -d and data/train.csv (gitignored).
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from psycopg.types.json import Jsonb  # noqa: E402

from kern.config import get_settings  # noqa: E402
from kern.data.features import build_features  # noqa: E402
from kern.quality import reference_sample  # noqa: E402
from kern.storage.db import apply_migrations, create_pool  # noqa: E402


def main() -> None:
    settings = get_settings()
    train = pd.read_csv(ROOT / "data" / "train.csv")
    features = build_features(train)

    pool = create_pool(settings.database_url)
    apply_migrations(pool)

    with pool.connection() as conn:
        pending = conn.execute(
            "SELECT name, version, feature_schema FROM model_versions "
            "WHERE reference_sample IS NULL"
        ).fetchall()

        if not pending:
            print("every version already has a reference sample")
        for name, version, schema in pending:
            missing = [column for column in schema if column not in features.columns]
            if missing:
                print(f"skipped {name} {version}: training data lacks {missing}")
                continue

            sample = reference_sample(
                features, schema, limit=settings.reference_sample_size
            )
            conn.execute(
                "UPDATE model_versions SET reference_sample = %s "
                "WHERE name = %s AND version = %s",
                (Jsonb(sample), name, version),
            )
            print(f"backfilled {name} {version} ({len(schema)} features)")

    pool.close()


if __name__ == "__main__":
    main()
