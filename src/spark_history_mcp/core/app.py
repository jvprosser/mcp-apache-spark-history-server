import asyncio
import logging
import os
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

    app_discovery = ApplicationDiscovery(clients)

    # Start a background refresh loop for any clients using CDP workload
    # tokens. This is a preemptive belt-and-braces alongside the on-401
    # refresh in ``_resilient_call`` — keeps hot paths from paying a retry
    # round-trip when the token is close to expiry.
    refresh_task = _start_cdp_refresh_task(clients)

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
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                # Nothing above cares whether the loop shut down cleanly.
                pass


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
