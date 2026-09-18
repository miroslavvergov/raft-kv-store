"""The durable portion of a node's state, as reloaded from stable storage."""

from dataclasses import dataclass
from typing import Optional

from raftkv.consensus import Log


@dataclass(frozen=True)
class PersistedState:
    """Everything a node persisted before stopping, read back on start-up.

    These are exactly the three pieces of state PERSIST-1 through
    PERSIST-3 require to survive a crash, and exactly what PERSIST-4
    through PERSIST-6 require to be reloaded before the node accepts or
    issues any RPC. Role is not part of this state: STATE-2 has every
    node start as a Follower regardless of the role it held before it
    stopped, so role is never persisted.

    Attributes:
        current_term: The last term the node persisted (0 for a node
            that has never run).
        voted_for: The node ID it voted for in `current_term`, or None
            if it had not voted in that term.
        log: The node's log, in index order starting at index 1.
    """

    current_term: int
    voted_for: Optional[int]
    log: Log
