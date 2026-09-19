"""Public API for tracing: what a node reports about its own decisions.

Modeled on etcd's raft, which reports itself in two ways: human-readable
log lines and structured trace events. The node's own logic contains no
tracing code: `traced` wraps a method from outside, and `NodeTracer` works
out from the node's state before and after the call what to report, through
the standard `logging` module. Both are off unless those loggers are
enabled.

Other layers should import from `raftkv.tracing` only, never from its
submodules directly.
"""

from raftkv.tracing.node_snapshot import NodeSnapshot
from raftkv.tracing.node_tracer import LOG_LINES, TRACE_EVENTS, NodeTracer
from raftkv.tracing.trace_event import TraceEvent, TraceMessage
from raftkv.tracing.traced_calls import traced

__all__ = [
    "LOG_LINES",
    "TRACE_EVENTS",
    "NodeSnapshot",
    "NodeTracer",
    "TraceEvent",
    "TraceMessage",
    "traced",
]
