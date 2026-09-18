"""The vocabulary of roles a Raft node can hold."""

from enum import Enum


class Role(Enum):
    """The three mutually exclusive roles a node can be in at any moment.

    STATE-1's three roles — nothing else is a legal value of
    `NodeState.role`, so every later requirement can say "as a Follower"
    or "as a Leader" without also having to rule out some fourth
    possibility.
    """

    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"
