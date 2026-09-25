"""Exceptions raised by the runtime that drives a node."""


class PeerUnreachableError(Exception):
    """Raised by a transport when an RPC gets no answer: lost, refused, or timed out.

    The sender gives up on that RPC; the next heartbeat interval sends again (FAIL-2,
    FAIL-3, DD-30).
    """


class NodeStoppedError(Exception):
    """Raised when a stopped node is asked to start, tick, append a command, or handle an RPC."""
