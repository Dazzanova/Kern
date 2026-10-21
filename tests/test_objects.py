import pytest

from kern.config import Settings
from kern.storage.objects import ObjectStore

pytestmark = pytest.mark.db


@pytest.fixture(scope="module")
def store():
    settings = Settings(_env_file=None)
    instance = ObjectStore(settings)
    instance.ensure_bucket()
    return instance


def test_round_trip_of_bytes(store):
    store.put_if_absent("tests/round-trip.bin", b"kern-payload")

    assert store.get("tests/round-trip.bin") == b"kern-payload"


def test_put_if_absent_does_not_overwrite(store):
    store.put_if_absent("tests/immutable.bin", b"first")
    store.put_if_absent("tests/immutable.bin", b"second")

    assert store.get("tests/immutable.bin") == b"first"


def test_missing_key_raises(store):
    with pytest.raises(KeyError):
        store.get("tests/never-written.bin")
