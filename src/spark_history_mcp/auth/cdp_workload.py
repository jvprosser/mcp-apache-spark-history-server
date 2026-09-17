"""CDP workload JWT provider.

Cloudera CDP DataHub clusters expose services (Spark History Server, HMS,
etc.) behind the Knox gateway; agents authenticate with a **workload JWT**
issued by CDP IAM. The canonical way to obtain one from a CML runtime is
the CDP CLI:

    cdp iam generate-workload-auth-token --workload-name <workload>

which prints ``{ "token": "...", "expireAt": "2026-…Z" }``. Tokens are
short-lived (typically ~10 minutes), so any long-running client needs to
refresh them.

This module wraps that flow:

* :class:`CDPWorkloadTokenProvider` caches the token in memory, refreshes
  ahead of expiry (``refresh_skew_seconds``), and is safe to call from
  multiple threads.
* When the ``cdp`` CLI is not on ``$PATH`` (local dev, unit tests) the
  provider falls back to the ``CDP_WORKLOAD_TOKEN`` env var. In that mode
  the token is treated as opaque and never refreshed — use it only for
  short-lived local scripts.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_ENV_FALLBACK = "CDP_WORKLOAD_TOKEN"
_CLI_NAME = "cdp"


class CDPWorkloadTokenError(RuntimeError):
    """Raised when a workload token cannot be produced or parsed."""


@dataclass(frozen=True)
class _CachedToken:
    token: str
    # Monotonic seconds; when ``time.monotonic() >= this``, refresh.
    refresh_at: float


class CDPWorkloadTokenProvider:
    """Thread-safe supplier of CDP workload JWTs.

    Parameters:
        workload_name: The CDP workload service name (e.g. ``"DE"``).
        refresh_skew_seconds: How early to refresh before ``expireAt``.
        cli_path: Override the discovered ``cdp`` binary. Mostly for tests.
        clock: Injected monotonic clock for tests. Returns seconds.
    """

    def __init__(
        self,
        workload_name: str,
        refresh_skew_seconds: int = 120,
        cli_path: Optional[str] = None,
        clock=time.monotonic,
    ) -> None:
        if not workload_name:
            raise ValueError("workload_name is required")
        self._workload_name = workload_name
        self._skew = max(0, int(refresh_skew_seconds))
        self._cli_path = cli_path or shutil.which(_CLI_NAME)
        self._clock = clock
        self._lock = threading.Lock()
        self._cached: Optional[_CachedToken] = None

    @property
    def workload_name(self) -> str:
        return self._workload_name

    def get_token(self) -> str:
        """Return a valid JWT, refreshing if the cached one is near expiry."""
        with self._lock:
            cached = self._cached
            if cached is not None and self._clock() < cached.refresh_at:
                return cached.token
            self._cached = self._fetch_locked()
            return self._cached.token

    def force_refresh(self) -> str:
        """Discard the cached token and fetch a new one immediately.

        Called by the REST client when a request returns 401/403 — the
        cached token was accepted at issue time but has since been rejected
        (e.g. server-side revocation, clock skew).
        """
        with self._lock:
            self._cached = self._fetch_locked()
            return self._cached.token

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _fetch_locked(self) -> _CachedToken:
        # Env var takes precedence when set — it's the "someone already
        # gave us a token" escape hatch used by tests and local dev.
        env_token = os.environ.get(_ENV_FALLBACK)
        if env_token:
            logger.debug(
                "Using CDP workload token from %s (no refresh possible)",
                _ENV_FALLBACK,
            )
            # No expiry info; refresh no sooner than in an hour so we don't
            # thrash if the caller wraps this in a tight retry loop.
            return _CachedToken(
                token=env_token,
                refresh_at=self._clock() + 3600,
            )

        if not self._cli_path:
            raise CDPWorkloadTokenError(
                f"'{_CLI_NAME}' CLI not found on PATH and "
                f"{_ENV_FALLBACK} is not set; cannot obtain a workload JWT "
                f"for workload_name={self._workload_name!r}."
            )

        try:
            completed = subprocess.run(
                [
                    self._cli_path,
                    "iam",
                    "generate-workload-auth-token",
                    "--workload-name",
                    self._workload_name,
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
        except subprocess.CalledProcessError as exc:
            raise CDPWorkloadTokenError(
                f"cdp iam generate-workload-auth-token exited "
                f"{exc.returncode}: {exc.stderr.strip() or exc.stdout.strip()}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise CDPWorkloadTokenError(
                "cdp iam generate-workload-auth-token timed out after 30s"
            ) from exc

        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise CDPWorkloadTokenError(
                f"cdp CLI returned non-JSON output: {completed.stdout[:200]!r}"
            ) from exc

        token = payload.get("token")
        expire_at = payload.get("expireAt")
        if not token:
            raise CDPWorkloadTokenError(
                f"cdp CLI response missing 'token' field: {payload!r}"
            )

        refresh_at = self._compute_refresh_at(expire_at)
        logger.info(
            "Obtained CDP workload JWT for workload_name=%s (expires %s)",
            self._workload_name,
            expire_at or "unknown",
        )
        return _CachedToken(token=token, refresh_at=refresh_at)

    def _compute_refresh_at(self, expire_at: Optional[str]) -> float:
        """Translate the CLI's absolute ISO expiry to our monotonic clock.

        Uses ``time.time`` for the wall-clock delta only; the cached deadline
        is stored in the injected monotonic clock's domain so tests can
        advance it deterministically.
        """
        now_mono = self._clock()
        if not expire_at:
            # Unknown expiry — default to 5 minutes, half of the typical TTL.
            return now_mono + 300

        try:
            # CDP returns e.g. "2026-09-17T12:34:56.789Z".
            iso = expire_at.replace("Z", "+00:00")
            expiry_dt = datetime.fromisoformat(iso)
            if expiry_dt.tzinfo is None:
                expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
        except ValueError:
            logger.warning(
                "Could not parse CDP expireAt=%r; defaulting to 5 min TTL",
                expire_at,
            )
            return now_mono + 300

        ttl_seconds = expiry_dt.timestamp() - time.time()
        return now_mono + max(0.0, ttl_seconds - self._skew)
