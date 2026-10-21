"""Window parsing shared by the report endpoints and the evaluation cycle."""

from datetime import timedelta

_WINDOW_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
MAX_WINDOW = timedelta(days=365)


def parse_window(value: str) -> timedelta:
    unit = value[-1:]
    amount = value[:-1]
    # Bound the digit count before int()/timedelta arithmetic: an absurdly long
    # amount would otherwise overflow C-level arithmetic and surface as a 500.
    if (
        unit not in _WINDOW_UNITS
        or not amount.isdigit()
        or len(amount) > 6
        or int(amount) == 0
    ):
        raise ValueError(f"invalid window: {value!r}")
    try:
        delta = timedelta(seconds=int(amount) * _WINDOW_UNITS[unit])
    except OverflowError as error:
        raise ValueError(f"invalid window: {value!r}") from error
    if delta > MAX_WINDOW:
        raise ValueError(f"window exceeds {MAX_WINDOW.days} days")
    return delta
