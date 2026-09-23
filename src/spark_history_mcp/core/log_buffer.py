"""In-memory ring buffer of recent log records, readable through a tool.

Everything this server logs goes to stderr. Some MCP hosts -- Cloudera AI
Agent Studio among them -- do not surface an MCP subprocess's stderr anywhere
the operator can reach, which makes the startup probe, the per-server
connection warnings and the TLS cipher check unobservable. A connectivity
failure then looks exactly like an agent that declined to call a tool, and
there is no way to tell the two apart.

A tool result is the only channel the MCP protocol guarantees will reach the
caller: log notifications are fire-and-forget and a host may drop them
silently. So keep a bounded copy of what was logged and let a tool return it.

Bounded because nothing ever drains this -- the process lives as long as the
host keeps the MCP session open.
"""

import logging
from collections import deque
from typing import Deque, List, Optional

# Enough to cover startup plus a handful of tool calls, while staying small
# enough to return whole inside a tool result.
DEFAULT_CAPACITY = 200

# Attach to the package logger rather than the root: this captures everything
# this project logs and nothing from urllib3/botocore, whose retry chatter
# would evict our own records from a buffer this size.
PACKAGE_LOGGER = "spark_history_mcp"

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class RingBufferHandler(logging.Handler):
    """Keep the most recent ``capacity`` formatted records in memory."""

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        super().__init__()
        self._records: Deque[str] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        # A handler must never raise into the code that logged; a diagnostic
        # aid that can break the thing it is diagnosing is worse than none.
        try:
            self._records.append(self.format(record))
        except Exception:  # noqa: BLE001
            self.handleError(record)

    def records(self) -> List[str]:
        return list(self._records)


_handler: Optional[RingBufferHandler] = None


def install(capacity: int = DEFAULT_CAPACITY) -> RingBufferHandler:
    """Attach the buffer to the package logger.

    Idempotent: repeated calls return the existing handler rather than
    stacking duplicates, so importing or re-entering the entry point cannot
    double-record.
    """
    global _handler
    if _handler is None:
        _handler = RingBufferHandler(capacity)
        _handler.setFormatter(logging.Formatter(_FORMAT))
        logging.getLogger(PACKAGE_LOGGER).addHandler(_handler)
    return _handler


def records() -> List[str]:
    """Buffered records, oldest first; empty when :func:`install` never ran."""
    return _handler.records() if _handler is not None else []


def reset() -> None:
    """Detach and discard the buffer. For tests."""
    global _handler
    if _handler is not None:
        logging.getLogger(PACKAGE_LOGGER).removeHandler(_handler)
        _handler = None
