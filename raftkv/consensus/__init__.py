"""Public API for the pure Raft consensus core.

The Log, LogPosition, and NodeState classes, plus the Role vocabulary and
FollowerProgress tracking — no I/O, no asyncio, no persistence.

This is the layer's public surface. Other layers (persistence, RPC, the KV
store) should import from `raftkv.consensus` only — never reach into
`raftkv.consensus.log`, `.log_position`, `.state`, or any other submodule
directly — so this package's internal file layout stays free to change
without breaking anything outside it.
"""

from raftkv.consensus.errors import IllegalTransition
from raftkv.consensus.follower_progress import FollowerProgress
from raftkv.consensus.log import Log, LogEntry
from raftkv.consensus.log_position import LogPosition
from raftkv.consensus.role import Role
from raftkv.consensus.state import NodeState

__all__ = [
    "LogEntry",
    "Log",
    "LogPosition",
    "FollowerProgress",
    "IllegalTransition",
    "NodeState",
    "Role",
]
