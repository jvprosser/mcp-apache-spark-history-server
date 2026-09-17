"""Tests for :mod:`spark_history_mcp.auth.cdp_workload`."""

import json
import os
import subprocess
import time
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from spark_history_mcp.auth.cdp_workload import (
    CDPWorkloadTokenError,
    CDPWorkloadTokenProvider,
)


def _cli_stdout(token: str = "jwt-abc", ttl_seconds: int = 600) -> str:
    """Fake CDP CLI JSON output with an ISO expiry ``ttl_seconds`` from now."""
    expire = datetime.now(tz=timezone.utc) + timedelta(seconds=ttl_seconds)
    return json.dumps(
        {
            "token": token,
            "expireAt": expire.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        }
    )


class _FakeClock:
    """A monotonic clock we can advance in tests."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestCDPWorkloadTokenProvider(unittest.TestCase):
    def setUp(self) -> None:
        # Clear the env fallback so it doesn't pollute the CLI-driven tests.
        self._prev_env = os.environ.pop("CDP_WORKLOAD_TOKEN", None)

    def tearDown(self) -> None:
        if self._prev_env is not None:
            os.environ["CDP_WORKLOAD_TOKEN"] = self._prev_env
        else:
            os.environ.pop("CDP_WORKLOAD_TOKEN", None)

    def _provider(self, clock: _FakeClock) -> CDPWorkloadTokenProvider:
        # cli_path="/usr/bin/cdp" bypasses shutil.which so we're not
        # dependent on the CDP CLI actually being installed.
        return CDPWorkloadTokenProvider(
            workload_name="DE",
            refresh_skew_seconds=60,
            cli_path="/usr/bin/cdp",
            clock=clock,
        )

    def test_get_token_shells_out_once_within_validity(self) -> None:
        clock = _FakeClock()
        provider = self._provider(clock)
        completed = MagicMock(returncode=0, stdout=_cli_stdout(ttl_seconds=600))
        with patch(
            "spark_history_mcp.auth.cdp_workload.subprocess.run",
            return_value=completed,
        ) as run:
            self.assertEqual(provider.get_token(), "jwt-abc")
            # Inside the validity window, no second shell-out.
            clock.advance(300)
            self.assertEqual(provider.get_token(), "jwt-abc")
        run.assert_called_once()

    def test_get_token_refreshes_past_expiry_minus_skew(self) -> None:
        clock = _FakeClock()
        provider = self._provider(clock)
        first = MagicMock(returncode=0, stdout=_cli_stdout("t1", ttl_seconds=600))
        second = MagicMock(returncode=0, stdout=_cli_stdout("t2", ttl_seconds=600))
        with patch(
            "spark_history_mcp.auth.cdp_workload.subprocess.run",
            side_effect=[first, second],
        ) as run:
            self.assertEqual(provider.get_token(), "t1")
            # ttl 600 minus skew 60 → refresh after 540s.
            clock.advance(541)
            self.assertEqual(provider.get_token(), "t2")
        self.assertEqual(run.call_count, 2)

    def test_force_refresh_bypasses_cache(self) -> None:
        clock = _FakeClock()
        provider = self._provider(clock)
        first = MagicMock(returncode=0, stdout=_cli_stdout("t1", ttl_seconds=600))
        second = MagicMock(returncode=0, stdout=_cli_stdout("t2", ttl_seconds=600))
        with patch(
            "spark_history_mcp.auth.cdp_workload.subprocess.run",
            side_effect=[first, second],
        ):
            self.assertEqual(provider.get_token(), "t1")
            self.assertEqual(provider.force_refresh(), "t2")

    def test_env_fallback_when_cli_missing(self) -> None:
        os.environ["CDP_WORKLOAD_TOKEN"] = "static-jwt"
        # No cli_path and shutil.which returns None → env fallback wins.
        with patch(
            "spark_history_mcp.auth.cdp_workload.shutil.which", return_value=None
        ):
            provider = CDPWorkloadTokenProvider(
                workload_name="DE",
                refresh_skew_seconds=60,
                clock=_FakeClock(),
            )
            self.assertEqual(provider.get_token(), "static-jwt")

    def test_missing_cli_and_no_env_raises(self) -> None:
        with patch(
            "spark_history_mcp.auth.cdp_workload.shutil.which", return_value=None
        ):
            provider = CDPWorkloadTokenProvider(
                workload_name="DE",
                refresh_skew_seconds=60,
                clock=_FakeClock(),
            )
            with self.assertRaises(CDPWorkloadTokenError):
                provider.get_token()

    def test_cli_nonzero_exit_raises(self) -> None:
        clock = _FakeClock()
        provider = self._provider(clock)
        err = subprocess.CalledProcessError(
            returncode=2,
            cmd=["cdp"],
            output="",
            stderr="workload not found",
        )
        with patch(
            "spark_history_mcp.auth.cdp_workload.subprocess.run", side_effect=err
        ):
            with self.assertRaises(CDPWorkloadTokenError) as ctx:
                provider.get_token()
        self.assertIn("workload not found", str(ctx.exception))

    def test_cli_non_json_raises(self) -> None:
        clock = _FakeClock()
        provider = self._provider(clock)
        completed = MagicMock(returncode=0, stdout="not-json")
        with patch(
            "spark_history_mcp.auth.cdp_workload.subprocess.run",
            return_value=completed,
        ):
            with self.assertRaises(CDPWorkloadTokenError):
                provider.get_token()

    def test_workload_name_required(self) -> None:
        with self.assertRaises(ValueError):
            CDPWorkloadTokenProvider(workload_name="")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
