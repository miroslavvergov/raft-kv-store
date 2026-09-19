"""The vocabulary of roles a Raft node can hold."""

from enum import Enum


class Role(Enum):
    """A node's role: exactly one of these at any moment, and nothing else (STATE-1)."""

    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"
