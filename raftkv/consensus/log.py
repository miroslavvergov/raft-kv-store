"""The Log class: Raft's replicated log and its matching/repair mechanics.

The log itself, and the mechanics that keep a Follower's copy of it
consistent with the Leader's (REPL-5 through REPL-8). A `Log` owns its
entries and answers its own consistency and repair questions about
itself. No I/O, no asyncio, no persistence — these are pure decisions
("does this log already agree with the Leader at this position?", "what
should the log look like after this AppendEntries RPC?") that need to be
correct on their own before any networking or disk-access code can be
trusted to call them correctly.

Raft's own 1-based indexing is used throughout (the first entry in the log
is index 1, and index 0 conventionally means "before the first entry, no
entry required") rather than Python's native 0-based list indexing, so
this class reads the same way the requirements below and the paper itself
describe it. The conversion to a 0-based Python list position happens
once, at the point of use, inside each method.
"""

from dataclasses import dataclass
from typing import Iterator, Optional

from raftkv.consensus.log_position import LogPosition


@dataclass(frozen=True)
class LogEntry:
    """A single entry in a node's replicated log.

    Tagged with the term the Leader was in when it first appended the
    command to its own log (§5.3). `term` is what the Log Matching
    Property, and the whole consistency check below, is built on — two
    entries at the same index are guaranteed identical (and everything
    before them too) precisely because a Leader only ever writes one
    entry per (index, term) pair and never rewrites its own log
    afterward.

    `command` is an opaque string (DD-21): the KV Store layer serializes
    a client command exactly once, when the Leader first proposes it, and
    from then on every node — the Leader included — stores, replicates,
    persists, and applies that exact string. Nothing in the Raft layer
    ever decodes or re-encodes it, which is what makes APPLY-7 hold: a
    value that went through a decode/re-encode round trip (a tuple coming
    back as a list, an integer dict key coming back as a string) would no
    longer be the command the Leader appended, and replicas applying
    different values would break the premise APPLY-6's determinism
    depends on.

    Frozen because a log entry, once created, must never be mutated in
    place — the only way an entry ever goes away is by being replaced
    wholesale, as part of computing a new log in `Log.after_append_entries`
    when a genuine conflict forces an overwrite.

    A non-string command is rejected at construction. Anything else would
    be silently converted somewhere downstream — SQLite, for instance,
    stores the integer 5 in a TEXT column as the string '5' — and the
    entry read back after a restart would no longer equal the one that
    was appended.

    Attributes:
        term: The term the leader was in when this entry was appended.
        command: The client command, already serialized by the KV Store
            layer, carried verbatim.

    Raises:
        TypeError: If `command` is not a str.
    """

    term: int
    command: str

    def __post_init__(self) -> None:
        if not isinstance(self.command, str):
            raise TypeError(
                f"LogEntry.command must be a str, got {type(self.command).__name__}"
            )


class Log:
    """A node's replicated log, exposing REPL-5/REPL-8's own mechanics.

    Wraps an ordered sequence of LogEntry and answers exactly the two
    questions a Follower needs answered about its own log when handling
    AppendEntries: "do I already agree with the Leader here?"
    (`matches`) and "what should my log become after this RPC?"
    (`after_append_entries`). Immutable by construction — `matches` and
    the read-only properties never change anything, and
    `after_append_entries` returns a new `Log` rather than mutating this
    one, consistent with `LogEntry` itself being frozen.
    """

    def __init__(self, entries: Optional[list[LogEntry]] = None) -> None:
        """Wrap a sequence of log entries.

        Args:
            entries: The entries this log starts with, in order (index 1
                first). Copied on construction, so later mutating the
                list passed in has no effect on this `Log`. Defaults to
                an empty log.
        """
        self._entries: list[LogEntry] = list(entries) if entries else []

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[LogEntry]:
        return iter(self._entries)

    def __getitem__(self, key):
        return self._entries[key]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Log):
            return NotImplemented
        return self._entries == other._entries

    @property
    def last_index(self) -> int:
        """The 1-based index of this log's last entry, or 0 if empty.

        0 doubles as "no entries yet" throughout this class, matching how
        `matches` treats a `prev_log_index` of 0 as automatically
        satisfied. This is the fact ELECT-7 requires a RequestVote RPC to
        carry about the candidate's own log, and one half of
        `last_position`.
        """
        return len(self._entries)

    @property
    def last_term(self) -> int:
        """The term of this log's last entry, or 0 if empty.

        Paired with `last_index`, this is exactly the (term, index) fact
        ELECT-7 requires a RequestVote RPC to carry, and the other half
        of `last_position`.
        """
        return self._entries[-1].term if self._entries else 0

    @property
    def last_position(self) -> LogPosition:
        """This log's last entry, as the LogPosition ELECT-10 compares.

        A convenience combining `last_term` and `last_index` into the
        single value `LogPosition.is_at_least_as_up_to_date_as` actually
        takes, so a caller doing an ELECT-10 comparison doesn't have to
        assemble the pair by hand.
        """
        return LogPosition(term=self.last_term, index=self.last_index)

    def matches(self, prev_log_index: int, prev_log_term: int) -> bool:
        """Check whether this log already agrees with the leader at a position.

        Implements REPL-5's consistency check: "a Follower shall reject
        an AppendEntries RPC whenever its own log does not contain an
        entry at the RPC's previous-entry index whose term matches the
        RPC's previous-entry term." This method IS that check, phrased
        as the acceptance condition rather than the rejection one — a
        caller rejects the RPC precisely when this returns False.

        `prev_log_index == 0` always returns True, because index 0 means
        "the Leader is proposing to replace everything from the very
        start of the log" — there is no preceding entry for the two logs
        to agree on, so nothing can disagree either. This is what lets a
        brand-new, empty log accept its very first AppendEntries.

        For any other `prev_log_index`, the check is exactly the
        single-point comparison the Log Matching Property (§5.3) says is
        sufficient: if this log doesn't even have an entry that far in,
        or the entry it has there was written in a different term, the
        two logs cannot be assumed to agree before that point either, so
        the RPC is rejected outright rather than trusting a shorter or
        differently-originated prefix.

        Args:
            prev_log_index: The 1-based index of the entry immediately
                preceding the entries under consideration, as carried by
                an AppendEntries RPC. 0 means there is no preceding
                entry.
            prev_log_term: The term the entry at `prev_log_index` is
                expected to have.

        Returns:
            True if this log already agrees with the leader at
            `prev_log_index` (or `prev_log_index` is 0), False
            otherwise.
        """
        if prev_log_index == 0:
            return True
        if prev_log_index > len(self._entries):
            return False
        return self._entries[prev_log_index - 1].term == prev_log_term

    def after_append_entries(
        self, prev_log_index: int, entries: list[LogEntry]
    ) -> "Log":
        """Compute the log that results from applying an AppendEntries RPC.

        Implements REPL-8 — "once an AppendEntries RPC is accepted, the
        Leader's replication logic shall overwrite any conflicting
        entries already present in that Follower's log with the
        Leader's own entries" — by applying the two receiver rules
        Figure 2 of the paper actually specifies this as: the moment an
        existing entry conflicts with an incoming one (same position,
        different term), delete that entry and everything after it;
        then append whatever of the incoming entries didn't already fit.

        Deliberately does NOT delete anything outside that rule: an
        entry beyond the range `entries` covers, or one that already
        matches its incoming counterpart term-for-term, is left exactly
        as it was. This is why a stale, uncommitted tail entry from an
        old, abandoned Leader (Figure 7, scenarios (c) and (d) of the
        paper) survives an ordinary heartbeat untouched — a heartbeat
        carries no new entries to conflict with it — and is only ever
        overwritten once the current Leader actually produces a
        genuinely conflicting entry at that same position.

        Callers must have already confirmed
        `self.matches(prev_log_index, prev_log_term)` — REPL-5 — before
        calling this; it does not repeat that check itself, and calling
        it against a `prev_log_index` the two logs don't actually agree
        on yet would silently corrupt the result rather than raise.

        Returns a new `Log` rather than mutating this one — consistent
        with `LogEntry` being frozen, nothing in this class ever mutates
        a log, it only ever computes what the log should become next.

        Args:
            prev_log_index: The 1-based index of the entry immediately
                preceding `entries`, as already confirmed by a prior
                `matches` call.
            entries: The leader's entries to reconcile into this log, in
                order, starting immediately after `prev_log_index`.

        Returns:
            A new `Log` representing this log after applying the
            AppendEntries RPC.
        """
        new_entries = list(self._entries)
        for offset, entry in enumerate(entries):
            position = prev_log_index + offset
            if position < len(new_entries):
                if new_entries[position].term != entry.term:
                    new_entries = new_entries[:position] + [entry]
            else:
                new_entries.append(entry)
        return Log(new_entries)

    def first_differing_index(self, other: "Log") -> Optional[int]:
        """Find the first index at which this log and another log differ.

        Compares entries position by position and reports the 1-based
        index of the first position where they are not equal. If one log
        is a strict prefix of the other, the first differing index is the
        one just past the end of the shorter log — the first position
        that exists in only one of them.

        This is what lets a durable copy of the log be brought in line
        with a newly computed one by rewriting only what changed: every
        entry before the returned index is identical in both logs, and
        everything from it onward must be replaced. In particular, when
        `after_append_entries` leaves a stale, unconflicted
        tail in place (Figure 7, scenarios (c) and (d)), the logs don't
        differ there, so that tail is never touched on disk either.

        Args:
            other: The log to compare against.

        Returns:
            The 1-based index of the first differing position, or None if
            the two logs are identical.
        """
        for position, (mine, theirs) in enumerate(zip(self._entries, other._entries)):
            if mine != theirs:
                return position + 1
        if len(self._entries) != len(other._entries):
            return min(len(self._entries), len(other._entries)) + 1
        return None
