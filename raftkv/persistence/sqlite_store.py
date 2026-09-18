"""A node's stable storage: one SQLite file holding its term, vote, and log."""

import contextlib
from types import TracebackType
from typing import AsyncIterator, Optional

import aiosqlite

from raftkv.consensus import Log, LogEntry
from raftkv.persistence.persisted_state import PersistedState

_SCHEMA = """
CREATE TABLE IF NOT EXISTS node_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    current_term INTEGER NOT NULL CHECK (current_term >= 0),
    voted_for INTEGER CHECK (voted_for IS NULL OR voted_for > 0)
) STRICT;

CREATE TABLE IF NOT EXISTS log (
    idx INTEGER PRIMARY KEY CHECK (idx > 0),
    term INTEGER NOT NULL,
    command TEXT NOT NULL
) STRICT;

INSERT OR IGNORE INTO node_state (id, current_term, voted_for) VALUES (1, 0, NULL);
"""


class SqliteStore:
    """One node's SQLite database file (DD-6), accessed through aiosqlite (DD-19).

    Holds the three pieces of state PERSIST-1 through PERSIST-3 require
    to survive a crash, in two tables (DD-23):

    - `node_state` is a single row, enforced by `CHECK (id = 1)`,
      holding `current_term` and `voted_for`. It is updated in place and
      keeps no history: the only reader is start-up reload (PERSIST-4,
      PERSIST-5), which needs the latest values, and STATE-6 discards a
      vote the moment its term is left behind, so no earlier value is
      ever meaningful again.
    - `log` holds one row per entry, keyed by its 1-based Raft index, so
      appending is an insert and REPL-8's "delete this entry and
      everything after it" is a single range delete.

    Both tables are STRICT, so a column declared INTEGER accepts only
    integers: without STRICT, SQLite would store a text value such as
    `'n1'` in `voted_for`, and `'n1' > 0` evaluates true because SQLite
    orders all text above all integers, so the CHECK constraint alone
    would not stop it. Together with `CHECK (voted_for > 0)`, this makes
    DD-20's rule — a node ID is a positive integer — hold for every value
    that reaches disk. `command` is stored exactly as given, never
    decoded or re-encoded (DD-21, APPLY-7).

    Every write happens inside one transaction that is either committed
    whole or rolled back whole (DD-7), and the connection runs with
    `PRAGMA synchronous = FULL`, so a committed transaction has been
    flushed to the disk rather than left in the operating system's
    cache — which is what "persist to stable storage" (PERSIST-1 through
    PERSIST-3) requires.

    Used as an async context manager: entering opens the file, creating
    the schema if it doesn't exist yet; exiting closes it.
    """

    def __init__(self, path: str) -> None:
        """Prepare a store backed by the SQLite file at `path`.

        Args:
            path: Filesystem path of this node's database file. It is
                created on first open if it doesn't already exist.
        """
        self._path = path
        self._connection: Optional[aiosqlite.Connection] = None

    async def __aenter__(self) -> "SqliteStore":
        """Open the database file, creating the schema on first use.

        On a file that has never been used, this also inserts the single
        `node_state` row with `current_term = 0` and no vote — the state
        STATE-2 has a brand-new node start from. On an existing file,
        that insert is ignored and the persisted row is left untouched.

        Returns:
            This store, ready for reads and writes.
        """
        self._connection = await aiosqlite.connect(self._path)
        await self._connection.execute("PRAGMA synchronous = FULL")
        await self._connection.executescript(_SCHEMA)
        await self._connection.commit()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Close the database file."""
        await self._connection.close()

    async def save_term_and_vote(
        self, current_term: int, voted_for: Optional[int]
    ) -> None:
        """Persist the node's current term and vote, replacing the previous ones.

        Implements PERSIST-1 and PERSIST-2 as one atomic update of the
        single `node_state` row. Both values change together wherever
        they change at all — ELECT-3 and ELECT-4 on becoming Candidate,
        STATE-5 and STATE-6 on observing a higher term — so they are
        written together. This method returns only once the transaction
        has committed, so a caller that awaits it before responding to an
        RPC satisfies "persist before responding".

        Args:
            current_term: The term to persist.
            voted_for: The node ID voted for in `current_term`, or None
                for no vote.

        Raises:
            sqlite3.IntegrityError: If `current_term` is not a
                non-negative integer, or `voted_for` is neither None nor
                a positive integer. Nothing is written in that case.
        """
        async with self._transaction() as connection:
            await connection.execute(
                "UPDATE node_state SET current_term = ?, voted_for = ? WHERE id = 1",
                (current_term, voted_for),
            )

    async def save_log_from(self, index: int, entries: list[LogEntry]) -> None:
        """Replace the persisted log from `index` onward with `entries`.

        Implements PERSIST-3 for REPL-8's overwrite rule: every persisted
        entry at `index` or later is deleted, then `entries` are written
        at `index`, `index + 1`, and so on. Entries before `index` are
        not touched. Passing no entries simply truncates the log from
        `index` onward.

        The delete and the inserts are one transaction (DD-7): a failure
        partway through rolls the whole thing back, so the persisted log
        is never left with the old tail deleted and only part of the new
        one written.

        Args:
            index: The 1-based index of the first entry to replace. The
                entries before it must already be persisted.
            entries: The entries to persist, in order, starting at
                `index`.

        Raises:
            sqlite3.IntegrityError: If an entry's term cannot be stored
                as an integer. Nothing is written in that case.
        """
        async with self._transaction() as connection:
            await connection.execute("DELETE FROM log WHERE idx >= ?", (index,))
            await connection.executemany(
                "INSERT INTO log (idx, term, command) VALUES (?, ?, ?)",
                [
                    (index + offset, entry.term, entry.command)
                    for offset, entry in enumerate(entries)
                ],
            )

    async def load(self) -> PersistedState:
        """Read back everything the node persisted.

        Implements the reads behind PERSIST-4, PERSIST-5, and PERSIST-6:
        the node's current term, its vote, and its whole log in index
        order. A file that has never been written to yields
        `current_term = 0`, no vote, and an empty log.

        Returns:
            The node's persisted term, vote, and log.
        """
        async with self._connection.execute(
            "SELECT current_term, voted_for FROM node_state WHERE id = 1"
        ) as cursor:
            current_term, voted_for = await cursor.fetchone()
        async with self._connection.execute(
            "SELECT term, command FROM log ORDER BY idx"
        ) as cursor:
            rows = await cursor.fetchall()
        return PersistedState(
            current_term=current_term,
            voted_for=voted_for,
            log=Log([LogEntry(term=term, command=command) for term, command in rows]),
        )

    @contextlib.asynccontextmanager
    async def _transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        """Run the enclosed writes as one transaction: all committed, or all rolled back."""
        try:
            yield self._connection
            await self._connection.commit()
        except BaseException:
            await self._connection.rollback()
            raise
