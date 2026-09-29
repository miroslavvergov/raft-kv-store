"""Exceptions raised by the runtime that drives a node."""


class PeerUnreachableError(Exception):
    """Raised by a transport when an RPC gets no answer: lost, refused, or timed out.

    The sender gives up on that RPC; the next heartbeat interval sends again (FAIL-2,
    FAIL-3, DD-30).
    """


class NodeStoppedError(Exception):
    """Raised when a stopped node is asked to start, tick, append a command, or handle an RPC.

    `RaftNode.propose` raises it too when the node stops while its command waits (DD-33).
    """


class LeadershipLostError(Exception):
    """Raised by `RaftNode.propose` when its node stops being Leader before its command is applied.

    The command's entry is in the node's log, and a later Leader may still commit or overwrite
    it, so the caller cannot tell whether the command takes effect; retrying it with the same
    session and sequence number is safe either way (DD-15, DD-33). When the message says the
    entry was overwritten, the command was not applied.
    """
