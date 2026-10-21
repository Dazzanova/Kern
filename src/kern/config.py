from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KERN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql://kern:kern@localhost:5432/kern"
    test_database_url: str | None = None
    s3_endpoint: str = "http://localhost:9000"
    s3_bucket: str = "kern-models"
    s3_access_key: str = "kern"
    s3_secret_key: str = "kern-secret"
    telemetry_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    feature_sample_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    telemetry_queue_size: int = Field(default=10_000, gt=0)
    model_name: str = "load-forecast"
    # Empty disables auth on the promote and outcome-ingestion endpoints.
    api_token: str = ""
    reference_sample_size: int = Field(default=1000, gt=0)
    drift_min_samples: int = Field(default=100, gt=0)
    # Seconds between scheduled detection cycles; 0 disables the loop.
    evaluation_interval: int = Field(default=300, ge=0)
    # Days of telemetry, checks, and resolved incidents to keep; 0 disables pruning.
    retention_days: int = Field(default=30, ge=0)
    performance_min_samples: int = Field(default=20, ge=1)
    performance_mae_warn: float = Field(default=90.0, gt=0)
    performance_mae_fail: float = Field(default=120.0, gt=0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
