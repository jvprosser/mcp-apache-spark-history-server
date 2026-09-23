import asyncio
import logging
import os
import ssl
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from typing import Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from spark_history_mcp.api.emr_persistent_ui_client import EMRPersistentUIClient
from spark_history_mcp.api.spark_client import SparkRestClient
from spark_history_mcp.auth.cdp_workload import CDPWorkloadTokenError
from spark_history_mcp.config.config import Config, load_config

from ..utils.utils import ApplicationDiscovery

# How often the CDP workload token refresh loop wakes up. The provider's own
# skew (default 120s) decides whether an actual refresh happens; this only
# bounds latency between "token about to expire" and "we noticed."
_CDP_REFRESH_INTERVAL_SECONDS = 300

# Shared by the startup TLS check and the probe's error hint: both diagnose
# the same empty-cipher condition, one before any request and one after a
# request has already failed.
_NO_CIPHERS_REMEDY = (
    "uv-managed CPython bundles its own OpenSSL but still reads the host "
    "/etc/ssl/openssl.cnf, and a RHEL-family crypto policy (CDSW/CML included) "
    "can leave it with an empty cipher list. Retry with OPENSSL_CONF=/dev/null, "
    "or use the system interpreter (uvx --python /usr/bin/python3 ...). "
    "verify_ssl and ssl_ca_cert will not help, since this fails before any "
    "certificate is checked."
)

logger = logging.getLogger(__name__)


def _emr_cookie_reauth(emr_client: EMRPersistentUIClient) -> str:
    """Re-establish the EMR session and return a fresh Cookie header value."""
    emr_client.setup_http_session()
    return emr_client.cookie_header()


@dataclass
class AppContext:
    clients: dict[str, SparkRestClient]
    default_client: Optional[SparkRestClient] = None
    app_discovery: Optional[ApplicationDiscovery] = None


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    try:
        async with _app_lifespan_impl(server) as ctx:
            yield ctx
    except BaseException as exc:  # noqa: BLE001
        # FastMCP runs lifespan inside a TaskGroup; unhandled exceptions
        # surface as "unhandled errors in a TaskGroup (1 sub-exception)"
        # with no traceback. Log the real cause here.
        logger.error("Spark HS MCP startup failed: %s", exc, exc_info=True)
        raise


@asynccontextmanager
async def _app_lifespan_impl(server: FastMCP) -> AsyncIterator[AppContext]:
    config = load_config()
    logger.info(
        "Loaded config: servers=%s default_transport=%s",
        list(config.servers.keys()),
        config.mcp.transport,
    )
    _warn_if_tls_unusable(config)

    clients: dict[str, SparkRestClient] = {}
    default_client = None

    for name, server_config in config.servers.items():
        logger.info(
            "Initialising Spark client name=%s url=%s auth_type=%s",
            name,
            server_config.url,
            getattr(server_config.auth, "type", None) if server_config.auth else None,
        )
        # Check if this is an EMR server configuration
        if server_config.emr_cluster_arn:
            # Create EMR client
            emr_client = EMRPersistentUIClient(server_config)

            # Initialize EMR client (create persistent UI, get presigned URL, setup session)
            base_url, _session = emr_client.initialize()

            # Create a modified server config with the base URL
            emr_server_config = server_config.model_copy()
            emr_server_config.url = base_url

            # Route EMR through the generated client using the session cookies as
            # a Cookie header, with a re-auth callback to refresh on 401/403.
            spark_client = SparkRestClient(emr_server_config)
            spark_client.configure_cookies(
                emr_client.cookie_header(),
                reauth=partial(_emr_cookie_reauth, emr_client),
            )

            clients[name] = spark_client
        else:
            # Regular Spark REST client
            clients[name] = SparkRestClient(server_config)

        if server_config.default:
            default_client = clients[name]

        if server_config.probe_on_startup:
            _log_startup_probe(name, clients[name])

    app_discovery = ApplicationDiscovery(clients)

    # Start a background refresh loop for any clients using CDP workload
    # tokens. This is a preemptive belt-and-braces alongside the on-401
    # refresh in ``_resilient_call`` — keeps hot paths from paying a retry
    # round-trip when the token is close to expiry.
    refresh_task = _start_cdp_refresh_task(clients)

    # A stdio server emits nothing more until a client speaks JSON-RPC on
    # stdin, which is indistinguishable from a hang when run by hand. Say so
    # explicitly rather than leaving the last line looking like a stall.
    transport = os.getenv("SHS_MCP_TRANSPORT") or config.mcp.transport or "unknown"
    if transport == "stdio":
        logger.info(
            "MCP server ready: transport=stdio, waiting for JSON-RPC on stdin. "
            "No further log output is expected until a client connects -- this "
            "is not a hang."
        )
    else:
        logger.info(
            "MCP server ready: transport=%s listening on %s:%s",
            transport,
            config.mcp.address,
            config.mcp.port,
        )

    try:
        yield AppContext(
            clients=clients,
            default_client=default_client,
            app_discovery=app_discovery,
        )
    finally:
        if refresh_task is not None:
            refresh_task.cancel()
            try:
                await refresh_task
            except (asyncio.CancelledError, Exception) as exc:  # noqa: BLE001
                # Nothing above cares whether the loop shut down cleanly, but
                # swallowing it in silence would hide real teardown bugs.
                logger.debug("CDP refresh task teardown: %s", exc)


def _log_startup_probe(name: str, client: SparkRestClient) -> None:
    """Log the result of a one-shot connectivity/auth probe for one server.

    Log-and-continue by design: a transient SHS outage must not stop the MCP
    server from starting, since the tools recover on their own once the
    server is reachable again.
    """
    result = client.probe()
    if result["error"]:
        hint = _probe_error_hint(result["error"])
        logger.error(
            "Startup probe %s: GET %s failed after %dms -- %s (auth=%s)%s",
            name,
            result["url"],
            result["elapsed_ms"],
            result["error"],
            client.auth_summary(),
            f". {hint}" if hint else "",
        )
        return

    status = result["status"]
    challenge = result["www_authenticate"]
    if 200 <= status < 300:
        logger.info(
            "Startup probe %s: GET %s -> %d in %dms (auth=%s)",
            name,
            result["url"],
            status,
            result["elapsed_ms"],
            client.auth_summary(),
        )
        return

    # Non-2xx: the status and the challenge scheme are the whole diagnosis.
    # 404 -> base URL path is wrong. 401 + Basic -> credentials wrong/missing
    # but the path is right. 401 + Negotiate -> Kerberos SPNEGO, which this
    # client does not implement.
    logger.error(
        "Startup probe %s: GET %s -> %d in %dms, WWW-Authenticate=%s (auth=%s). %s",
        name,
        result["url"],
        status,
        result["elapsed_ms"],
        challenge or "<none>",
        client.auth_summary(),
        _probe_hint(status, challenge),
    )


def _probe_error_hint(error: str) -> str:
    """Turn a probe transport failure into an actionable sentence.

    Only covers failures whose cause is *not* recoverable from the exception
    text. A refused connection or a DNS miss already says what is wrong;
    padding those adds noise, so they return "" and the raw error stands.
    """
    if "LIBRARY_HAS_NO_CIPHERS" in error:
        return (
            "The TLS handshake offered no ciphers at all, which points at the "
            f"interpreter's OpenSSL rather than this server: {_NO_CIPHERS_REMEDY} "
            "If curl reaches this same URL, that confirms it."
        )
    return ""


def _warn_if_tls_unusable(config: Config) -> None:
    """Warn when HTTPS is configured but OpenSSL offers no ciphers at all.

    An empty cipher list fails every HTTPS request with
    LIBRARY_HAS_NO_CIPHERS. ``_probe_error_hint`` explains that once a
    request has failed, but the probe is opt-in (``probe_on_startup``
    defaults to False), so check it here too -- this way the diagnosis needs
    no configuration to appear. Skipped when no server uses TLS, to keep
    plain-HTTP local and e2e runs quiet.
    """
    if not any((s.url or "").startswith("https://") for s in config.servers.values()):
        return
    try:
        ciphers = ssl.create_default_context().get_ciphers()
    except Exception as exc:  # noqa: BLE001
        # A default context that will not build is itself worth reporting,
        # but it is not the empty-cipher case and gets no remedy text.
        logger.warning("Could not inspect the OpenSSL cipher list: %s", exc)
        return
    if not ciphers:
        logger.warning(
            "OpenSSL reports no available ciphers, so every HTTPS request "
            "will fail. %s",
            _NO_CIPHERS_REMEDY,
        )


def _probe_hint(status: int, challenge: Optional[str]) -> str:
    """Turn a probe status/challenge pair into an actionable sentence."""
    scheme = (challenge or "").split(" ", 1)[0].lower()
    if status == 404:
        return (
            "404 means the base URL path is wrong, not that auth failed -- "
            "check the server.url value (for Cloudera Knox, the Spark 3 "
            "topology is '<datahub>/cdp-proxy-api/spark3history')."
        )
    if status in (401, 403):
        if scheme == "negotiate":
            return (
                "The server wants Kerberos SPNEGO, which this client does not "
                "implement -- use a gateway that accepts Basic auth instead."
            )
        if scheme == "basic":
            return (
                "The server accepts Basic auth, so the path is correct and the "
                "credentials were rejected or missing -- check auth.username "
                "and auth.password."
            )
        return "Credentials were rejected; the base URL path itself is valid."
    return "Unexpected status; the base URL resolved but did not serve the API."


def _start_cdp_refresh_task(
    clients: dict[str, SparkRestClient],
) -> Optional[asyncio.Task]:
    """Kick off a background refresher for every CDP-workload-authed client.

    Returns ``None`` when nothing needs refreshing so we don't spin an
    idle task on non-Cloudera deployments.
    """
    watched = [c for c in clients.values() if c.token_provider is not None]
    if not watched:
        return None
    return asyncio.create_task(
        _cdp_refresh_loop(watched), name="cdp-workload-token-refresh"
    )


async def _cdp_refresh_loop(clients: list[SparkRestClient]) -> None:
    while True:
        try:
            await asyncio.sleep(_CDP_REFRESH_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            return
        for client in clients:
            provider = client.token_provider
            if provider is None:
                continue
            try:
                # ``get_token`` is a no-op when still within the validity
                # window; only reissues near expiry.
                token = await asyncio.to_thread(provider.get_token)
                client._apply_bearer(token)  # noqa: SLF001 (owner writes)
            except CDPWorkloadTokenError as exc:
                logger.warning(
                    "CDP workload token refresh failed for workload=%s: %s",
                    provider.workload_name,
                    exc,
                )


def run(config: Config):
    # Auto-detect AWS credentials and register troubleshooting tools if available
    try:
        import boto3

        session = boto3.Session()
        creds = session.get_credentials()
        if creds is not None and session.region_name:
            from spark_history_mcp.tools.aws_troubleshooting import (
                register_troubleshooting_tools,
            )

            register_troubleshooting_tools(session.region_name)
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).debug(
            "AWS troubleshooting tools not registered: %s", e
        )

    mcp.settings.host = config.mcp.address
    mcp.settings.port = int(config.mcp.port)
    mcp.settings.debug = bool(config.mcp.debug)

    # Configure transport security settings for DNS rebinding protection
    # See: https://github.com/modelcontextprotocol/python-sdk/issues/1798
    if config.mcp.transport_security:
        ts_config = config.mcp.transport_security
        mcp.settings.transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=ts_config.enable_dns_rebinding_protection,
            allowed_hosts=ts_config.allowed_hosts,
            allowed_origins=ts_config.allowed_origins,
        )

    mcp_cfg = config.mcp
    transport = os.getenv("SHS_MCP_TRANSPORT") or mcp_cfg.transport
    if (
        not transport
        and "transports" in mcp_cfg.model_fields_set
        and mcp_cfg.transports
    ):
        transport = mcp_cfg.transports[0]
    transport = transport or "streamable-http"
    mcp.run(transport=transport)


mcp = FastMCP("Spark Events", lifespan=app_lifespan)

# Import tools and prompts to register them with MCP
from spark_history_mcp.prompts import prompts  # noqa: E402,F401
from spark_history_mcp.tools import tools  # noqa: E402,F401
