import pytest
from pydantic import ValidationError

from kern.config import Settings


def test_defaults_match_compose_stack(monkeypatch):
    # conftest sets these for the test suite; clear them to observe true defaults.
    for name in ("KERN_DATABASE_URL", "KERN_TEST_DATABASE_URL", "KERN_FEATURE_SAMPLE_RATE"):
        monkeypatch.delenv(name, raising=False)

    settings = Settings(_env_file=None)

    assert settings.database_url == "postgresql://kern:kern@localhost:5432/kern"
    assert settings.s3_bucket == "kern-models"
    assert settings.model_name == "load-forecast"
    assert settings.feature_sample_rate == 0.05


def test_reads_prefixed_environment_variables(monkeypatch):
    monkeypatch.setenv("KERN_MODEL_NAME", "other-model")
    monkeypatch.setenv("KERN_TELEMETRY_QUEUE_SIZE", "5")

    settings = Settings(_env_file=None)

    assert settings.model_name == "other-model"
    assert settings.telemetry_queue_size == 5


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_rejects_out_of_range_sample_rate(monkeypatch, value):
    monkeypatch.setenv("KERN_FEATURE_SAMPLE_RATE", str(value))

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
