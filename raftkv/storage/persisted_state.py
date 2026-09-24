"""The durable portion of a node's state, as reloaded from stable storage."""

from dataclasses import dataclass

from raftkv.consensus import Log


@dataclass(frozen=True)
class PersistedState:
    """A node's persisted term, vote, and log, as read back on start-up.

    Exactly the state that must survive a crash (PERSIST-1, PERSIST-2,
    PERSIST-3) and be reloaded before any RPC (PERSIST-4, PERSIST-5,
    PERSIST-6). Role is not included: every node restarts as a Follower
    (STATE-2).

    Attributes:
        current_term: The last persisted term; 0 for a node that never ran.
        voted_for: The node voted for in `current_term`, or None.
        log: The node's log.
    """

    current_term: int
    voted_for: int | None
    log: Log
