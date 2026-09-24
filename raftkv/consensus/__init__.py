"""Public API of the pure Raft consensus core: no I/O, asyncio, or persistence.

Import from `raftkv.consensus`, never from its submodules, so their layout
can change without breaking other layers.
"""

from raftkv.consensus.append_entries import AppendEntriesRequest, AppendEntriesResponse
from raftkv.consensus.candidacy import Candidacy
from raftkv.consensus.cluster import Cluster
from raftkv.consensus.errors import (
    CommittedEntryConflictError,
    IllegalTransitionError,
    NotLeaderError,
)
from raftkv.consensus.follower_progress import FollowerProgress
from raftkv.consensus.leadership import Leadership
from raftkv.consensus.log import Log, LogEntry
from raftkv.consensus.log_position import LogPosition
from raftkv.consensus.request_vote import RequestVoteRequest, RequestVoteResponse
from raftkv.consensus.role import Role
from raftkv.consensus.state import NodeState

__all__ = [
    "AppendEntriesRequest",
    "AppendEntriesResponse",
    "Candidacy",
    "Cluster",
    "CommittedEntryConflictError",
    "FollowerProgress",
    "IllegalTransitionError",
    "Leadership",
    "Log",
    "LogEntry",
    "LogPosition",
    "NodeState",
    "NotLeaderError",
    "RequestVoteRequest",
    "RequestVoteResponse",
    "Role",
]
