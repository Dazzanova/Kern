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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
