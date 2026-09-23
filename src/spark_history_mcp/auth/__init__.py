"""Authentication providers for the Spark History Server client.

This package holds pluggable providers used by
:class:`spark_history_mcp.api.spark_client.SparkRestClient`. The stock repo
supports basic auth and static bearer tokens inline; providers here handle
flows that need lifecycle management (refresh, cluster-specific issuance).

Currently just :mod:`cdp_workload`, for Cloudera CDP workload JWTs.
"""

from spark_history_mcp.auth.cdp_workload import (  # noqa: F401
    CDPWorkloadTokenError,
    CDPWorkloadTokenProvider,
)
