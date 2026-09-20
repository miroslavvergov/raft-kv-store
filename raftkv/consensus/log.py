"""The replicated log: consistency check and repair (REPL-5, REPL-8).

Every index is 1-based, as Raft numbers entries; 0 means "before the first
entry".
"""

from collections.abc import Iterator
from dataclasses import dataclass

from raftkv.consensus.log_position import LogPosition


@dataclass(frozen=True)
class LogEntry:
    """One entry of a node's replicated log.

    Attributes:
        term: The term of the Leader that first appended this entry.
        command: The client command, serialized once by the KV Store layer
            (DD-21). Raft never decodes or re-encodes it (APPLY-7), so every
            replica applies the same value, as APPLY-6's determinism requires.

    Raises:
        TypeError: If `term` is not an int or `command` not a str. Any other type
            would be converted on its way to disk (SQLite stores 5 in a TEXT
            column as '5') and reload unequal to the entry appended.
        ValueError: If `term` is below 1: every entry is appended by the Leader
            of some term, and terms start at 1.
    """

    term: int
    command: str

    def __post_init__(self) -> None:
        # NOTE: bool is an int subclass, so True would pass as term 1 without this check.
        if isinstance(self.term, bool) or not isinstance(self.term, int):
            raise TypeError(f"LogEntry.term must be an int, got {type(self.term).__name__}")
        if self.term < 1:
            raise ValueError(f"LogEntry.term must be at least 1, got {self.term}")
        if not isinstance(self.command, str):
            raise TypeError(f"LogEntry.command must be a str, got {type(self.command).__name__}")


class Log:
    """A node's replicated log; immutable, so every change produces a new `Log`."""

    def __init__(self, entries: list[LogEntry] | None = None) -> None:
        """Build a log from a copy of `entries`.

        Args:
            entries: The entries in index order, index 1 first; empty if omitted.
        """
        self._entries: list[LogEntry] = list(entries) if entries else []

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[LogEntry]:
        return iter(self._entries)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Log):
            return NotImplemented
        return self._entries == other._entries

    def __repr__(self) -> str:
        return f"Log({self._entries!r})"

    @property
    def last_index(self) -> int:
        """The 1-based index of the last entry; 0 if empty (ELECT-7)."""
        return len(self._entries)

    @property
    def last_term(self) -> int:
        """The term of the last entry; 0 if empty (ELECT-7)."""
        return self._entries[-1].term if self._entries else 0

    @property
    def last_position(self) -> LogPosition:
        """The last entry's (term, index), as ELECT-10 compares it."""
        return LogPosition(term=self.last_term, index=self.last_index)

    def entry_at(self, index: int) -> LogEntry:
        """Return the entry at `index`.

        Args:
            index: A 1-based index, from 1 to `last_index`.

        Raises:
            IndexError: If `index` is outside that range.
        """
        # NOTE: index 0 would read list position -1, the last entry, so the range check
        # must reject it before the lookup.
        if not 1 <= index <= len(self._entries):
            raise IndexError(f"index {index} is outside the log's 1..{len(self._entries)}")
        return self._entries[index - 1]

    def term_at(self, index: int) -> int:
        """Return the term of the entry at `index`; 0 for index 0.

        Args:
            index: A 1-based index, from 0 to `last_index`.

        Raises:
            IndexError: If `index` is outside that range.
        """
        if index == 0:
            return 0
        return self.entry_at(index).term

    def entries_from(self, index: int) -> list[LogEntry]:
        """Return a new list of the entries from `index` to the end.

        Args:
            index: A 1-based index, from 1 to `last_index + 1`; `last_index + 1`
                yields an empty list.

        Raises:
            IndexError: If `index` is outside that range.
        """
        if not 1 <= index <= len(self._entries) + 1:
            raise IndexError(f"index {index} is outside the log's 1..{len(self._entries) + 1}")
        return self._entries[index - 1 :]

    def matches(self, prev_log_index: int, prev_log_term: int) -> bool:
        """Check whether this log agrees with the Leader's through `prev_log_index` (REPL-5).

        Two logs holding an entry with the same index and term are identical up to
        it, so checking that one entry covers the whole prefix. A missing entry or a
        different term means no agreement. `prev_log_index` 0 always agrees, since
        there is no earlier entry; this lets an empty log accept its first entries.

        Args:
            prev_log_index: The 1-based index of the entry preceding the RPC's
                entries; 0 if there is none.
            prev_log_term: The term that entry must have.

        Returns:
            True if the logs agree; False if the caller must reject the RPC.
        """
        if prev_log_index == 0:
            return True
        # NOTE: a log shorter than prev_log_index rejects the RPC; term_at would raise (REPL-5).
        if prev_log_index > len(self._entries):
            return False
        return self.term_at(prev_log_index) == prev_log_term

    def after_append_entries(self, prev_log_index: int, entries: list[LogEntry]) -> "Log":
        """Return the log that results from accepting an AppendEntries RPC (REPL-8).

        At the first position where an existing entry's term differs from the
        incoming one, that entry and everything after it are deleted; incoming
        entries not already present are then appended. Nothing else is deleted:
        matching entries, and entries past the last incoming one with no conflict
        before them, stay. So a heartbeat leaves a stale, uncommitted tail in place.

        The caller must first confirm `matches(prev_log_index, prev_log_term)`
        (REPL-5); this method does not, and without it returns a wrong log without
        raising.

        Args:
            prev_log_index: The 1-based index of the entry preceding `entries`.
            entries: The Leader's entries, in order, from `prev_log_index + 1`.

        Returns:
            A new `Log`; this one is unchanged.
        """
        new_entries = list(self._entries)
        for offset, entry in enumerate(entries):
            position = prev_log_index + offset
            if position < len(new_entries):
                # NOTE: a matching entry stays, so a delayed or duplicated RPC cannot delete
                # entries a later one appended.
                if new_entries[position].term != entry.term:
                    new_entries = new_entries[:position] + [entry]
            else:
                new_entries.append(entry)
        return Log(new_entries)

    def first_differing_index(self, other: "Log") -> int | None:
        """Return the 1-based index of the first entry that differs from `other`'s.

        If one log is a prefix of the other, that is the index just past the
        shorter one. Every entry before it is identical in both logs, so a durable
        copy is brought in line by rewriting only from that index on.

        Args:
            other: The log to compare against.

        Returns:
            The first differing index, or None if the logs are identical.
        """
        # NOTE: comparing against a shorter or longer log is normal, so the pairs stop at the
        # shorter one and the length check below reports where it ends.
        for position, (mine, theirs) in enumerate(zip(self._entries, other._entries, strict=False)):
            if mine != theirs:
                return position + 1
        if len(self._entries) != len(other._entries):
            return min(len(self._entries), len(other._entries)) + 1
        return None
