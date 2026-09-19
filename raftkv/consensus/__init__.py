"""Public API for the pure Raft consensus core.

The Log, LogPosition, NodeState, Cluster, Candidacy, and Leadership
classes, the RequestVote messages, plus the Role vocabulary and
per-Follower FollowerProgress — no I/O, no asyncio, no persistence.

This is the layer's public surface. Other layers (persistence, RPC, the KV
store) should import from `raftkv.consensus` only — never reach into
`raftkv.consensus.log`, `.log_position`, `.state`, or any other submodule
directly — so this package's internal file layout stays free to change
without breaking anything outside it.
"""

from raftkv.consensus.candidacy import Candidacy
from raftkv.consensus.cluster import Cluster
from raftkv.consensus.errors import IllegalTransition
from raftkv.consensus.follower_progress import FollowerProgress
from raftkv.consensus.leadership import Leadership
from raftkv.consensus.log import Log, LogEntry
from raftkv.consensus.log_position import LogPosition
from raftkv.consensus.request_vote import RequestVoteRequest, RequestVoteResponse
from raftkv.consensus.role import Role
from raftkv.consensus.state import NodeState

__all__ = [
    "LogEntry",
    "Log",
    "LogPosition",
    "Cluster",
    "RequestVoteRequest",
    "RequestVoteResponse",
    "Candidacy",
    "FollowerProgress",
    "Leadership",
    "IllegalTransition",
    "NodeState",
    "Role",
]
