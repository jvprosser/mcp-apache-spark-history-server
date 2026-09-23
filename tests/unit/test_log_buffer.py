"""Tests for :mod:`spark_history_mcp.core.log_buffer`."""

import logging
import unittest

from spark_history_mcp.core import log_buffer


class TestLogBuffer(unittest.TestCase):
    def setUp(self) -> None:
        log_buffer.reset()

    def tearDown(self) -> None:
        log_buffer.reset()

    def test_records_empty_before_install(self) -> None:
        """Reading before install must be harmless, not an error."""
        self.assertEqual(log_buffer.records(), [])

    def test_captures_package_log_records(self) -> None:
        log_buffer.install()
        logging.getLogger("spark_history_mcp.core.app").warning("probe failed: boom")

        records = log_buffer.records()
        self.assertEqual(len(records), 1)
        self.assertIn("probe failed: boom", records[0])
        # The level and logger name are the useful part when the only reader is
        # an agent quoting this back to a human.
        self.assertIn("WARNING", records[0])
        self.assertIn("spark_history_mcp.core.app", records[0])

    def test_ignores_records_from_other_packages(self) -> None:
        """urllib3 retry chatter must not evict our own records."""
        log_buffer.install()
        logging.getLogger("urllib3.connectionpool").warning("retrying")

        self.assertEqual(log_buffer.records(), [])

    def test_evicts_oldest_past_capacity(self) -> None:
        log_buffer.install(capacity=3)
        logger = logging.getLogger("spark_history_mcp.test")
        for i in range(5):
            logger.warning("record-%d", i)

        records = log_buffer.records()
        self.assertEqual(len(records), 3)
        self.assertIn("record-2", records[0])
        self.assertIn("record-4", records[-1])

    def test_install_is_idempotent(self) -> None:
        """A second install must not double-record."""
        first = log_buffer.install()
        second = log_buffer.install()
        self.assertIs(first, second)

        logging.getLogger("spark_history_mcp.test").warning("once")
        self.assertEqual(len(log_buffer.records()), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
