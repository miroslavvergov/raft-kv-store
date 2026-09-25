"""Public API of tracing: etcd-format log lines and trace events about a node.

`traced` wraps node methods from outside, so the logic holds no tracing
code; `NodeTracer` derives what to report from the state before and after
each call and emits it through `logging`. Nothing is emitted unless those
loggers are enabled. Import from `raftkv.tracing`, never from its submodules.
"""

from raftkv.tracing.node_snapshot import NodeSnapshot
from raftkv.tracing.node_tracer import LOG_LINES_LOGGER, TRACE_EVENTS_LOGGER, NodeTracer
from raftkv.tracing.trace_event import TraceEvent, TraceMessage
from raftkv.tracing.traced_calls import traced

__all__ = [
    "LOG_LINES_LOGGER",
    "TRACE_EVENTS_LOGGER",
    "NodeSnapshot",
    "NodeTracer",
    "TraceEvent",
    "TraceMessage",
    "traced",
]
