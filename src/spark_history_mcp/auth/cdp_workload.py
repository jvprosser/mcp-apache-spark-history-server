"""CDP workload JWT provider.

Cloudera CDP DataHub clusters expose services (Spark History Server, HMS,
etc.) behind the Knox gateway; agents authenticate with a **workload JWT**
issued by CDP IAM. The canonical way to obtain one is:

    cdp iam generate-workload-auth-token --workload-name <workload>

which prints ``{ "token": "...", "expireAt": "2026-…Z" }``. Tokens are
short-lived (typically ~10 minutes), so any long-running client needs to
refresh them.

:class:`CDPWorkloadTokenProvider` caches the token in memory, refreshes
ahead of expiry (``refresh_skew_seconds``), and is safe to call from
multiple threads. It tries these sources in order:

1. **Env var** — ``CDP_WORKLOAD_TOKEN`` / ``CDP_WORKLOAD_AUTH_TOKEN`` /
   ``WORKLOAD_AUTH_TOKEN`` — for tests and container runtimes that inject
   a token at process start.
2. **Mounted file** — ``/tmp/jwt`` /
   ``/etc/machine-user-credentials/workload_token`` — the pattern some
   CML / CAI Workbench runtimes use to hand out the workload identity.
3. **cdpcli Python library** — for containers that ship the CDP SDK on
   ``sys.path`` but no ``cdp`` binary (Cloudera AI Agent Studio's default
   image). Uses ``iam.generate_workload_auth_token`` via the same CDP
   Control Plane API the CLI hits.
4. **cdp CLI subprocess** — the CML notebook / edge-node case where the
   binary is in ``$PATH`` but the SDK isn't importable.
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
# Additional env var names Cloudera runtimes have used for the same thing;
# check them in order and use whichever is populated. Keeps the door open
# to Agent Studio / CML / CAI Workbench without asking the user which one.
_ENV_FALLBACK_ALIASES = (
    "CDP_WORKLOAD_TOKEN",
    "CDP_WORKLOAD_AUTH_TOKEN",
    "WORKLOAD_AUTH_TOKEN",
)
# Filesystem locations Cloudera runtimes have mounted the workload JWT to.
_TOKEN_FILE_CANDIDATES = (
    "/tmp/jwt",
    "/etc/machine-user-credentials/workload_token",
)
_CLI_NAME = "cdp"


class CDPWorkloadTokenError(RuntimeError):
    """Raised when a workload token cannot be produced or parsed."""


class _CDPCLINotAvailable(Exception):
    """Signal the ``cdpcli`` library path can't produce a token; try CLI."""


# Known env vars / file paths where CDP creds live. Used only to include a
# hint in the "no token source" error message.
_CDP_CRED_ENV_VARS = (
    "CDP_ACCESS_KEY_ID",
    "CDP_PRIVATE_KEY",
    "CDP_PROFILE",
    "CDP_REGION",
    "CDP_ENDPOINT_URL",
    "CDP_WORKLOAD_AUTH_TOKEN",  # some CML runtimes inject this
    "WORKLOAD_AUTH_TOKEN",  # older Cloudera Machine Learning name
)
_CDP_CRED_FILE_PATHS = (
    os.path.expanduser("~/.cdp/credentials"),
    os.path.expanduser("~/.cdp/config"),
    "/tmp/jwt",  # CML/CAI sometimes drops the workload token here
    "/etc/machine-user-credentials/workload_token",
)


def _probe_cdp_context() -> str:
    """Return a short 'what we found in the container' string for error msgs."""
    present_env = [v for v in _CDP_CRED_ENV_VARS if os.environ.get(v)]
    present_files = [p for p in _CDP_CRED_FILE_PATHS if os.path.exists(p)]
    try:
        import cdpcli  # type: ignore  # noqa: F401
        cdpcli_installed = True
    except ImportError:
        cdpcli_installed = False
    return (
        f"Container probe: cdpcli_installed={cdpcli_installed}, "
        f"cdp_env_vars_present={present_env or 'none'}, "
        f"cdp_cred_files_present={present_files or 'none'}."
    )


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
        # Escape hatches, in order:
        #   1. Env var (set by tests, local dev, or Cloudera runtime injection).
        #   2. Well-known mounted file path (some CML/CAI runtimes drop a JWT
        #      here and refresh it out-of-band; we just re-read it).
        for env_name in _ENV_FALLBACK_ALIASES:
            env_token = os.environ.get(env_name)
            if env_token:
                logger.info(
                    "Using CDP workload token from env %s (no in-process "
                    "refresh; whatever populated it must refresh it)",
                    env_name,
                )
                return _CachedToken(
                    token=env_token.strip(),
                    # Re-read after 5 minutes in case the runtime rotated it.
                    refresh_at=self._clock() + 300,
                )
        for path in _TOKEN_FILE_CANDIDATES:
            try:
                with open(path, "r") as fh:
                    file_token = fh.read().strip()
                if file_token:
                    logger.info(
                        "Using CDP workload token from file %s "
                        "(re-reading every 5 minutes)",
                        path,
                    )
                    return _CachedToken(
                        token=file_token,
                        refresh_at=self._clock() + 300,
                    )
            except OSError:
                continue

        # In-process ``cdpcli`` fallback for containers that ship the Python
        # SDK but not the ``cdp`` binary (e.g. Cloudera AI Agent Studio's
        # runtime image). Same underlying REST call, no subprocess.
        try:
            return self._fetch_via_cdpcli()
        except _CDPCLINotAvailable:
            pass  # library not installed; fall through to CLI subprocess.

        if not self._cli_path:
            raise CDPWorkloadTokenError(
                f"'{_CLI_NAME}' CLI not found on PATH, cdpcli Python "
                f"library is unavailable, and {_ENV_FALLBACK} is not set; "
                f"cannot obtain a workload JWT for "
                f"workload_name={self._workload_name!r}. "
                f"{_probe_cdp_context()}"
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

        return self._parse_and_cache(completed.stdout)

    def _fetch_via_cdpcli(self) -> _CachedToken:
        """Call ``iam.generate_workload_auth_token`` via the cdpcli Python API.

        Same REST endpoint the CLI hits; avoids the subprocess dependency.
        Raises :class:`_CDPCLINotAvailable` when the package isn't installed
        or credentials can't be resolved, so the caller can fall back.
        """
        try:
            from cdpcli.client import ClientCreator, Context  # type: ignore
            from cdpcli.endpoint import (  # type: ignore
                EndpointCreator,
                EndpointResolver,
            )
            from cdpcli.loader import Loader  # type: ignore
            from cdpcli.parser import ResponseParserFactory  # type: ignore
        except ImportError as exc:
            raise _CDPCLINotAvailable(f"cdpcli not importable: {exc}") from exc

        try:
            loader = Loader()
            context = Context()
            creator = ClientCreator(
                loader,
                context,
                EndpointCreator(EndpointResolver()),
                "spark-history-mcp/cdp_workload",
                ResponseParserFactory(),
                retryhandler=None,
            )
            # ``get_credentials`` with parsed_globals=None consults env vars
            # (CDP_ACCESS_KEY_ID/CDP_PRIVATE_KEY) then ~/.cdp/credentials.
            credentials = context.get_credentials(parsed_globals=None)
            iam = creator.create_client(
                "iam",
                explicit_endpoint_url=None,
                region=None,
                tls_verification=True,
                credentials=credentials,
            )
            payload = iam.generate_workload_auth_token(
                workloadName=self._workload_name
            )
        except Exception as exc:  # noqa: BLE001
            # Any error here (missing creds, network, service error) — treat
            # as "cdpcli path unusable" so we can fall through to the CLI.
            raise _CDPCLINotAvailable(
                f"cdpcli generate_workload_auth_token failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        token = payload.get("token") if isinstance(payload, dict) else None
        expire_at = (
            payload.get("expireAt") if isinstance(payload, dict) else None
        )
        endpoint_url = (
            payload.get("endpointUrl") if isinstance(payload, dict) else None
        )
        if not token:
            raise _CDPCLINotAvailable(
                f"cdpcli returned no token: {payload!r}"
            )
        logger.info(
            "Obtained CDP workload JWT via cdpcli library "
            "(workload_name=%s, endpointUrl=%s, expires %s)",
            self._workload_name,
            endpoint_url or "unknown",
            expire_at or "unknown",
        )
        return _CachedToken(
            token=token,
            refresh_at=self._compute_refresh_at(expire_at),
        )

    def _parse_and_cache(self, stdout: str) -> _CachedToken:
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise CDPWorkloadTokenError(
                f"cdp CLI returned non-JSON output: {stdout[:200]!r}"
            ) from exc
        token = payload.get("token")
        expire_at = payload.get("expireAt")
        if not token:
            raise CDPWorkloadTokenError(
                f"cdp CLI response missing 'token' field: {payload!r}"
            )
        logger.info(
            "Obtained CDP workload JWT via cdp CLI "
            "(workload_name=%s, expires %s)",
            self._workload_name,
            expire_at or "unknown",
        )
        return _CachedToken(
            token=token,
            refresh_at=self._compute_refresh_at(expire_at),
        )

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
