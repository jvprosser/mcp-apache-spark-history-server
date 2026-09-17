"""Integration tests for SparkRestClient's CDP workload auth path."""

import unittest
from unittest.mock import MagicMock, patch

from spark_history_mcp.api.spark_client import SparkRestClient
from spark_history_mcp.api_client.exceptions import UnauthorizedException
from spark_history_mcp.config.config import AuthConfig, ServerConfig


def _server_config() -> ServerConfig:
    return ServerConfig(
        url="https://gw.example.com/env/cdp-proxy/sparkhistoryserver",
        auth=AuthConfig(type="cdp_workload", workload_name="DE"),
    )


class TestSparkRestClientCDPWorkload(unittest.TestCase):
    def test_initial_bearer_header_from_provider(self) -> None:
        with patch(
            "spark_history_mcp.api.spark_client.CDPWorkloadTokenProvider"
        ) as ProviderCls:
            provider = MagicMock()
            provider.get_token.return_value = "jwt-1"
            ProviderCls.return_value = provider

            client = SparkRestClient(_server_config())

        header = client._api.api_client.default_headers.get("Authorization")
        self.assertEqual(header, "Bearer jwt-1")
        self.assertIs(client.token_provider, provider)
        ProviderCls.assert_called_once_with(
            workload_name="DE",
            refresh_skew_seconds=120,
        )

    def test_401_triggers_force_refresh_and_retry(self) -> None:
        with patch(
            "spark_history_mcp.api.spark_client.CDPWorkloadTokenProvider"
        ) as ProviderCls:
            provider = MagicMock()
            provider.get_token.return_value = "jwt-1"
            provider.force_refresh.return_value = "jwt-2"
            ProviderCls.return_value = provider

            client = SparkRestClient(_server_config())

            # First call fails with 401, second succeeds. The generated
            # DefaultApi is mocked so we exercise the _resilient_call path
            # on ``list_jobs``.
            api = MagicMock()
            api.list_jobs.side_effect = [
                UnauthorizedException(status=401, reason="expired"),
                [MagicMock(status="SUCCEEDED")],
            ]
            client._api = api

            result = client.list_jobs("app-1")

        self.assertEqual(len(result), 1)
        provider.force_refresh.assert_called_once()
        # After refresh the Authorization header must have been rewritten
        # via set_default_header on the (mocked) api_client.
        client._api.api_client.set_default_header.assert_called_with(
            "Authorization", "Bearer jwt-2"
        )

    def test_workload_name_required(self) -> None:
        cfg = ServerConfig(
            url="https://gw.example.com/env/cdp-proxy/sparkhistoryserver",
            auth=AuthConfig(type="cdp_workload"),
        )
        with self.assertRaises(ValueError):
            SparkRestClient(cfg)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
