"""Public API for the pure Raft consensus core.

Log matching, vote comparison, and the Follower/Candidate/Leader state
machine — no I/O, no asyncio, no persistence.

This is the layer's public surface. Other layers (persistence, RPC, the KV
store) should import from `raftkv.consensus` only — never reach into
`raftkv.consensus.log`, `.election`, or `.state` directly — so this
package's internal file layout stays free to change without breaking
anything outside it.
"""

from raftkv.consensus.election import is_log_up_to_date
from raftkv.consensus.errors import IllegalTransition
from raftkv.consensus.log import (
    LogEntry,
    last_log_index,
    last_log_term,
    log_after_append_entries,
    log_matches,
    next_index_after_rejection,
)
from raftkv.consensus.role import Role
from raftkv.consensus.state import NodeState

__all__ = [
    "LogEntry",
    "log_after_append_entries",
    "last_log_index",
    "last_log_term",
    "log_matches",
    "next_index_after_rejection",
    "is_log_up_to_date",
    "IllegalTransition",
    "NodeState",
    "Role",
]
