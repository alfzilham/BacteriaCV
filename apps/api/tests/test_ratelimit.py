"""Tests for the in-memory sliding-window rate limiter.

Expected values are literal numbers worked out by hand, not derived from the
limiter's own constants, so a wrong window or an off-by-one cannot hide.
"""

from __future__ import annotations

import pytest

from app.core.ratelimit import (
    SlidingWindowLimiter,
    client_key,
    limit_from_env,
    trust_forwarded_from_env,
)


class FakeClock:
    """A clock the test advances by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_accepts_up_to_the_limit_then_refuses() -> None:
    """With a limit of 2, the third request inside the window is refused."""
    clock = FakeClock()
    limiter = SlidingWindowLimiter(2, window=60.0, clock=clock)

    assert limiter.hit("a") == 0
    assert limiter.hit("a") == 0
    assert limiter.hit("a") == 60


def test_retry_after_counts_down_to_the_oldest_hit() -> None:
    """The wait shrinks as time passes and is a whole number of seconds."""
    clock = FakeClock()
    limiter = SlidingWindowLimiter(1, window=60.0, clock=clock)

    assert limiter.hit("a") == 0
    clock.now += 20.5

    assert limiter.hit("a") == 40


def test_accepts_again_once_the_window_has_passed() -> None:
    """A hit exactly one window old no longer counts."""
    clock = FakeClock()
    limiter = SlidingWindowLimiter(1, window=60.0, clock=clock)

    assert limiter.hit("a") == 0
    clock.now += 59.0
    assert limiter.hit("a") == 1
    clock.now += 1.0

    assert limiter.hit("a") == 0


def test_refused_requests_are_not_recorded() -> None:
    """Hammering while blocked must not extend the block."""
    clock = FakeClock()
    limiter = SlidingWindowLimiter(1, window=60.0, clock=clock)

    assert limiter.hit("a") == 0
    clock.now += 30.0
    assert limiter.hit("a") == 30
    assert limiter.hit("a") == 30
    clock.now += 30.0

    assert limiter.hit("a") == 0


def test_keys_are_counted_separately() -> None:
    """One client reaching its limit does not block another."""
    limiter = SlidingWindowLimiter(1, clock=FakeClock())

    assert limiter.hit("a") == 0
    assert limiter.hit("a") > 0
    assert limiter.hit("b") == 0


def test_limit_zero_disables_the_limiter() -> None:
    """0 means off: nothing is ever refused."""
    limiter = SlidingWindowLimiter(0, clock=FakeClock())

    assert limiter.enabled is False
    assert all(limiter.hit("a") == 0 for _ in range(1000))


def test_negative_limit_and_bad_window_are_rejected() -> None:
    """Invalid construction fails loudly."""
    with pytest.raises(ValueError):
        SlidingWindowLimiter(-1)
    with pytest.raises(ValueError):
        SlidingWindowLimiter(1, window=0)


def test_idle_keys_are_swept_so_the_table_does_not_grow() -> None:
    """Keys seen once and then idle are dropped by the periodic sweep."""
    clock = FakeClock()
    limiter = SlidingWindowLimiter(5, window=60.0, clock=clock)

    for index in range(255):
        limiter.hit(f"ip-{index}")
    clock.now += 61.0
    limiter.hit("fresh")

    assert list(limiter._hits) == ["fresh"]


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({}, 20),
        ({"RATE_LIMIT_PER_MINUTE": ""}, 20),
        ({"RATE_LIMIT_PER_MINUTE": "  "}, 20),
        ({"RATE_LIMIT_PER_MINUTE": "5"}, 5),
        ({"RATE_LIMIT_PER_MINUTE": " 30 "}, 30),
        ({"RATE_LIMIT_PER_MINUTE": "0"}, 0),
    ],
)
def test_limit_from_env_reads_the_documented_values(environ, expected) -> None:
    """Unset or blank gives 20, a number is used, 0 turns it off."""
    assert limit_from_env(environ) == expected


@pytest.mark.parametrize("raw", ["abc", "1.5", "-1", "10/min"])
def test_limit_from_env_rejects_bad_values(raw: str) -> None:
    """A typo must stop startup instead of silently disabling the limit."""
    with pytest.raises(ValueError):
        limit_from_env({"RATE_LIMIT_PER_MINUTE": raw})


def test_trust_forwarded_only_accepts_the_value_one() -> None:
    """Anything but 1 keeps the safe default."""
    assert trust_forwarded_from_env({"RATE_LIMIT_TRUST_FORWARDED": "1"}) is True
    assert trust_forwarded_from_env({"RATE_LIMIT_TRUST_FORWARDED": "true"}) is False
    assert trust_forwarded_from_env({"RATE_LIMIT_TRUST_FORWARDED": "0"}) is False
    assert trust_forwarded_from_env({}) is False


def test_client_key_ignores_forwarded_header_by_default() -> None:
    """Without trust, a forged header cannot change the key."""
    assert client_key("9.9.9.9", "10.0.0.1", trust_forwarded=False) == "10.0.0.1"


def test_client_key_takes_the_rightmost_forwarded_entry_when_trusted() -> None:
    """The nearest proxy's entry is the last one and cannot be forged."""
    header = "6.6.6.6, 203.0.113.7"

    assert client_key(header, "10.0.0.1", trust_forwarded=True) == "203.0.113.7"


def test_client_key_falls_back_to_the_peer_and_then_unknown() -> None:
    """Missing or empty headers fall back safely."""
    assert client_key(None, "10.0.0.1", trust_forwarded=True) == "10.0.0.1"
    assert client_key(" , ", "10.0.0.1", trust_forwarded=True) == "10.0.0.1"
    assert client_key(None, None, trust_forwarded=True) == "unknown"
