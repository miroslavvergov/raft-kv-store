"""Public API for node persistence: stable storage for term, vote, and log.

Other layers should import from `raftkv.persistence` only, never from its
submodules directly.
"""

from raftkv.persistence.persisted_state import PersistedState
from raftkv.persistence.sqlite_store import SqliteStore

__all__ = [
    "PersistedState",
    "SqliteStore",
]
