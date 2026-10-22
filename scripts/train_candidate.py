"""Train a candidate on the training split and register it, without promoting it.

The holdout is `data/test.csv` (April, after the January-March training window). Promotion
is a separate, policy-checked operation:

    uv run python scripts/train_candidate.py --version cand-1
    curl -X POST localhost:8000/models/load-forecast/cand-1/promote

Artifacts are written under `models/candidates/` (gitignored).
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from kern.config import get_settings  # noqa: E402
from kern.registry import compute_digest, register  # noqa: E402
from kern.storage.db import apply_migrations, create_pool  # noqa: E402
from kern.storage.objects import ObjectStore  # noqa: E402
from kern.training.candidate import build_candidate  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        default=datetime.now(timezone.utc).strftime("cand-%Y%m%dT%H%M%S"),
        help="registry version name (default: a UTC timestamp)",
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "candidates")
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--reference-sample-size", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    settings = get_settings()

    train_df = pd.read_csv(args.data_dir / "train.csv")
    test_df = pd.read_csv(args.data_dir / "test.csv")

    overrides = {
        key: value
        for key, value in {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
        }.items()
        if value is not None
    }

    candidate = build_candidate(
        train_df,
        test_df,
        reference_sample_size=args.reference_sample_size
        or settings.reference_sample_size,
        **overrides,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = args.output_dir / f"{args.version}.json"
    candidate.model.save_model(artifact)

    pool = create_pool(settings.database_url)
    try:
        apply_migrations(pool)
        store = ObjectStore(settings)
        store.ensure_bucket()

        digest = compute_digest(artifact.read_bytes())
        record = register(
            pool,
            store,
            name=settings.model_name,
            version=args.version,
            artifact_path=artifact,
            metrics=candidate.metrics,
            feature_schema=candidate.feature_schema,
            reference_profile=candidate.reference_profile,
            reference_sample=candidate.reference_sample,
        )
    finally:
        pool.close()

    if record.artifact_sha256 != digest:
        raise SystemExit(
            f"{args.version} is already registered with a different artifact; "
            "choose a new --version"
        )

    print(f"registered {record.name} {record.version} ({record.artifact_sha256[:12]})")
    print(
        f"holdout MAE {candidate.metrics['mae']:.4f}  "
        f"RMSE {candidate.metrics['rmse']:.4f}  "
        f"rows {candidate.metrics['train_rows']}/{candidate.metrics['test_rows']}"
    )
    print(f"artifact: {artifact}")
    print(f"promote:  POST /models/{record.name}/{record.version}/promote")


if __name__ == "__main__":
    main()
