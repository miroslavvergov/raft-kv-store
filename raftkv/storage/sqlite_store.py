"""A node's stable storage: one SQLite file holding its term, vote, and log."""

import contextlib
from collections.abc import AsyncIterator
from types import TracebackType

import aiosqlite

from raftkv.consensus import Log, LogEntry
from raftkv.storage.persisted_state import PersistedState

_SCHEMA = """
CREATE TABLE IF NOT EXISTS node_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    current_term INTEGER NOT NULL CHECK (current_term >= 0),
    voted_for INTEGER CHECK (voted_for IS NULL OR voted_for > 0)
) STRICT;

CREATE TABLE IF NOT EXISTS log (
    idx INTEGER PRIMARY KEY CHECK (idx > 0),
    term INTEGER NOT NULL CHECK (term > 0),
    cluster_time INTEGER NOT NULL CHECK (cluster_time >= 0),
    command TEXT NOT NULL
) STRICT;

INSERT OR IGNORE INTO node_state (id, current_term, voted_for) VALUES (1, 0, NULL);
"""


class SqliteStore:
    """One node's SQLite database file (DD-6), accessed through aiosqlite (DD-19).

    Two STRICT tables (DD-23). `node_state` is one row (`CHECK (id = 1)`) of
    `current_term` and `voted_for`, updated in place: only the latest values
    are reloaded (PERSIST-4, PERSIST-5), and an old term's vote is void
    (STATE-6). `log` has one row per entry, keyed by its 1-based index so
    REPL-8's truncation is one range delete, and holding its term, cluster time
    (DD-32), and command. STRICT stores only integers in
    INTEGER columns and rejects values it cannot convert losslessly; without
    it `'n1'` would pass `CHECK (voted_for > 0)`, since SQLite orders text
    above integers. So every node ID on disk is a positive integer (DD-20).
    Commands are stored verbatim (DD-21, APPLY-7).

    Each write is one transaction, committed or rolled back whole (DD-7), and
    `PRAGMA synchronous = FULL` makes a commit reach the disk, not only the OS
    cache (DD-24), as stable storage requires (PERSIST-1, PERSIST-2,
    PERSIST-3).

    Use as an async context manager: entering opens the file and creates the
    schema if missing; exiting closes it.
    """

    def __init__(self, path: str) -> None:
        """Prepare a store for the SQLite file at `path`; nothing is opened yet.

        Args:
            path: The node's database file, created on first open if missing.
        """
        self._path = path
        self._connection: aiosqlite.Connection | None = None

    async def __aenter__(self) -> "SqliteStore":
        """Open the file with `synchronous = FULL` and create the schema if missing.

        A new file gets its `node_state` row at term 0 with no vote, where a
        brand-new node starts (PERSIST-4, PERSIST-5); an existing row is left
        untouched.

        Returns:
            This store, ready for reads and writes.
        """
        self._connection = await aiosqlite.connect(self._path)
        # NOTE: synchronous is a per-connection setting, not stored in the file, so every open
        # sets FULL rather than relying on the build's default (DD-24).
        await self._connection.execute("PRAGMA synchronous = FULL")
        # NOTE: the schema's INSERT OR IGNORE seeds term 0 only into a new file, so reopening
        # never resets a persisted term or vote (PERSIST-4, PERSIST-5).
        await self._connection.executescript(_SCHEMA)
        await self._connection.commit()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the database file."""
        await self._connection.close()

    async def save_term_and_vote(self, current_term: int, voted_for: int | None) -> None:
        """Persist `current_term` and `voted_for`, replacing the previous values.

        One atomic update of the `node_state` row (PERSIST-1, PERSIST-2), returning
        only after the commit, so awaiting it before answering an RPC persists
        before responding. Used when only the term or vote changes: becoming
        Candidate (ELECT-3, ELECT-4), catching up to a higher term (STATE-5,
        STATE-6), or granting a vote.

        Args:
            current_term: The term to persist.
            voted_for: The node voted for in `current_term`, or None.

        Raises:
            sqlite3.IntegrityError: If `current_term` is negative, `voted_for` is
                not positive, or either cannot be stored losslessly as an integer.
                Nothing is written.
        """
        async with self._transaction() as connection:
            await self._write_term_and_vote(connection, current_term, voted_for)

    async def replace_log_from(self, index: int, entries: list[LogEntry]) -> None:
        """Replace the persisted log from `index` onward with `entries` (PERSIST-3, REPL-8).

        Deletes every entry at `index` or later, then writes `entries` at `index`,
        `index + 1`, and so on, in one transaction (DD-7), so a failure never
        leaves the old tail deleted and the new one partly written. With no
        entries, it only truncates.

        Args:
            index: The 1-based index of the first entry to replace; at most one past
                the last persisted entry.
            entries: The entries to persist, in order, from `index`.

        Raises:
            ValueError: If `index` would leave a gap after the last persisted entry,
                which would reload as a renumbered log. Nothing is written.
            sqlite3.IntegrityError: If an entry's term or cluster time cannot be
                stored losslessly as an integer, or `index` is below 1 with entries
                given. Nothing is written.
        """
        async with self._transaction() as connection:
            await self._write_log_from(connection, index, entries)

    async def save_term_vote_and_log_from(
        self, current_term: int, voted_for: int | None, index: int, entries: list[LogEntry]
    ) -> None:
        """Persist a term, a vote, and a log change together, in one transaction (DD-7).

        What `save_term_and_vote` and `replace_log_from` each do, committed or
        rolled back as one, for an AppendEntries that both raises the term and
        changes the log. Two separate writes would leave a crash between them with
        entries stored under a term the node never recorded, or a term recorded
        without the entries that came with it.

        Args:
            current_term: The term to persist.
            voted_for: The node voted for in `current_term`, or None.
            index: The 1-based index of the first entry to replace; at most one past
                the last persisted entry.
            entries: The entries to persist, in order, from `index`.

        Raises:
            ValueError: If `index` would leave a gap after the last persisted entry.
                Nothing is written, the term and vote included.
            sqlite3.IntegrityError: If any value cannot be stored losslessly as an
                integer. Nothing is written.
        """
        async with self._transaction() as connection:
            await self._write_term_and_vote(connection, current_term, voted_for)
            await self._write_log_from(connection, index, entries)

    async def _write_term_and_vote(
        self, connection: aiosqlite.Connection, current_term: int, voted_for: int | None
    ) -> None:
        """Update the `node_state` row inside a caller's open transaction."""
        await connection.execute(
            "UPDATE node_state SET current_term = ?, voted_for = ? WHERE id = 1",
            (current_term, voted_for),
        )

    async def _write_log_from(
        self, connection: aiosqlite.Connection, index: int, entries: list[LogEntry]
    ) -> None:
        """Truncate the log from `index` and write `entries` there, in a caller's transaction."""
        async with connection.execute("SELECT COALESCE(MAX(idx), 0) FROM log") as cursor:
            (last_index,) = await cursor.fetchone()
        # NOTE: `load` rebuilds the log by row order, dropping idx, so a gap would renumber
        # every later entry on reload.
        if index > last_index + 1:
            raise ValueError(f"index {index} would leave a gap after entry {last_index}")
        # NOTE: deleting from `index` to the end, rather than upserting, also removes old
        # entries past the new last one, as a conflict truncation requires (REPL-8).
        await connection.execute("DELETE FROM log WHERE idx >= ?", (index,))
        await connection.executemany(
            "INSERT INTO log (idx, term, cluster_time, command) VALUES (?, ?, ?, ?)",
            [
                (index + offset, entry.term, entry.cluster_time, entry.command)
                for offset, entry in enumerate(entries)
            ],
        )

    async def load(self) -> PersistedState:
        """Return the persisted term, vote, and whole log (PERSIST-4, PERSIST-5, PERSIST-6).

        A file never written to yields term 0, no vote, and an empty log.
        """
        async with self._connection.execute(
            "SELECT current_term, voted_for FROM node_state WHERE id = 1"
        ) as cursor:
            current_term, voted_for = await cursor.fetchone()
        async with self._connection.execute(
            "SELECT term, cluster_time, command FROM log ORDER BY idx"
        ) as cursor:
            rows = await cursor.fetchall()
        return PersistedState(
            current_term=current_term,
            voted_for=voted_for,
            log=Log(
                [
                    LogEntry(term=term, command=command, cluster_time=cluster_time)
                    for term, cluster_time, command in rows
                ]
            ),
        )

    @contextlib.asynccontextmanager
    async def _transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        """Run the enclosed writes as one transaction: all committed, or all rolled back."""
        try:
            yield self._connection
            await self._connection.commit()
        # NOTE: BaseException, so a cancelled write rolls back too; an open transaction would
        # otherwise be committed by the next write.
        except BaseException:
            await self._connection.rollback()
            raise
