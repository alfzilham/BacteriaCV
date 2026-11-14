"""In-memory sliding-window rate limiter for the prediction endpoint.

The limiter keeps the timestamps of the most recent accepted requests per
client key and refuses a request when the key already used its quota inside
the window. State lives in the process memory: it is not shared between
replicas and it resets on restart. That is enough to stop one client from
saturating a single-instance deployment, which is the goal here.

This module has no third-party imports so its logic can be tested alone. The
HTTP wiring is in app/core/ratelimit_middleware.py.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping

RATE_LIMIT_ENV = "RATE_LIMIT_PER_MINUTE"
TRUST_FORWARDED_ENV = "RATE_LIMIT_TRUST_FORWARDED"
DEFAULT_LIMIT_PER_MINUTE = 20
WINDOW_SECONDS = 60.0
UNKNOWN_CLIENT = "unknown"

# Every SWEEP_EVERY checks, keys whose window is empty are dropped so the
# table cannot grow without bound when many different clients appear once.
SWEEP_EVERY = 256


class SlidingWindowLimiter:
    """Allow at most ``limit`` accepted requests per key inside ``window``.

    A limit of 0 disables the limiter: every request is accepted.
    """

    def __init__(
        self,
        limit: int,
        window: float = WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a limiter.

        Args:
            limit: Accepted requests per key inside the window. 0 disables it.
            window: Window length in seconds.
            clock: Monotonic time source in seconds, replaceable in tests.
        """
        if limit < 0:
            raise ValueError("limit must not be negative")
        if window <= 0:
            raise ValueError("window must be positive")
        self.limit = limit
        self.window = window
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._checks = 0
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        """Whether the limiter refuses anything at all."""
        return self.limit > 0

    def hit(self, key: str) -> int:
        """Register one request for ``key``.

        Args:
            key: The client identity, normally an IP address.

        Returns:
            0 when the request is accepted (and recorded), otherwise the
            whole number of seconds, at least 1, until it would be accepted.
            A refused request is not recorded.
        """
        if not self.enabled:
            return 0
        with self._lock:
            now = self._clock()
            self._checks += 1
            if self._checks % SWEEP_EVERY == 0:
                self._sweep(now)
            hits = self._hits.setdefault(key, deque())
            self._expire(hits, now)
            if len(hits) >= self.limit:
                return max(1, math.ceil(hits[0] + self.window - now))
            hits.append(now)
            return 0

    def _expire(self, hits: deque[float], now: float) -> None:
        """Drop timestamps that left the window."""
        while hits and now - hits[0] >= self.window:
            hits.popleft()

    def _sweep(self, now: float) -> None:
        """Remove keys that have no timestamp left inside the window."""
        for key in list(self._hits):
            hits = self._hits[key]
            self._expire(hits, now)
            if not hits:
                del self._hits[key]


def limit_from_env(environ: Mapping[str, str]) -> int:
    """Read the per-minute limit from the environment.

    Args:
        environ: The environment mapping.

    Returns:
        The configured limit, DEFAULT_LIMIT_PER_MINUTE when the variable is
        unset or blank. 0 means the limiter is off.

    Raises:
        ValueError: When the value is not a whole number of at least 0.
            Failing at startup is deliberate: a typo must not silently turn
            the protection off.
    """
    raw = environ.get(RATE_LIMIT_ENV, "").strip()
    if not raw:
        return DEFAULT_LIMIT_PER_MINUTE
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(
            f"{RATE_LIMIT_ENV} must be a whole number, got {raw!r}"
        ) from error
    if value < 0:
        raise ValueError(f"{RATE_LIMIT_ENV} must not be negative, got {value}")
    return value


def trust_forwarded_from_env(environ: Mapping[str, str]) -> bool:
    """Whether to take the client address from X-Forwarded-For.

    Only the value "1" turns it on, so the default never trusts a header the
    caller could forge.
    """
    return environ.get(TRUST_FORWARDED_ENV, "").strip() == "1"


def client_key(
    forwarded_for: str | None, peer: str | None, trust_forwarded: bool
) -> str:
    """Pick the identity a request is counted under.

    Args:
        forwarded_for: The raw X-Forwarded-For header value, if any.
        peer: The address of the direct TCP peer, if known.
        trust_forwarded: Whether a reverse proxy in front of the API is
            known to append the real client address to X-Forwarded-For.

    Returns:
        With trust_forwarded, the right-most non-empty X-Forwarded-For entry,
        which is the one the nearest proxy added and a client cannot forge
        when exactly one proxy sits in front. Otherwise the direct peer.
        UNKNOWN_CLIENT when neither is available.
    """
    if trust_forwarded and forwarded_for:
        entries = [part.strip() for part in forwarded_for.split(",")]
        entries = [part for part in entries if part]
        if entries:
            return entries[-1]
    return peer or UNKNOWN_CLIENT
