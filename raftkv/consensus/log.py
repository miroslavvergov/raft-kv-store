"""Pure functions for Raft's log-matching and log-repair mechanics.

The Raft log itself, and the mechanics that keep a Follower's copy of it
consistent with the Leader's (REPL-5 through REPL-8). Everything here is a
plain function over an ordinary Python list of LogEntry — no I/O, no
asyncio, no persistence — because these are pure decisions ("does this log
already agree with the Leader at this position?", "what should the log
look like after this AppendEntries RPC?") that need to be correct on their
own before any networking or disk-access code can be trusted to call them
correctly.

Raft's own 1-based indexing is used throughout (the first entry in the log
is index 1, and index 0 conventionally means "before the first entry, no
entry required") rather than Python's native 0-based list indexing, so
these functions read the same way the requirements below and the paper
itself describe them. The conversion to a 0-based Python list position
happens once, at the point of use, inside each function.
"""

from dataclasses import dataclass
from typing import Any


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

    Frozen because a log entry, once created, must never be mutated in
    place — the only way an entry ever goes away is by being replaced
    wholesale, as part of computing a new log in
    `log_after_append_entries` when a genuine conflict forces an
    overwrite.

    Attributes:
        term: The term the leader was in when this entry was appended.
        command: The client command this entry carries.
    """

    term: int
    command: Any


def last_log_index(log: list[LogEntry]) -> int:
    """Return the 1-based index of the last entry in the log.

    0 doubles as "no entries yet" throughout this module, matching how
    `log_matches` treats a `prev_log_index` of 0 as automatically
    satisfied. This is the fact ELECT-7 requires a RequestVote RPC to
    carry about the candidate's own log, and one half of what ELECT-10's
    up-to-date comparison is defined over.

    Args:
        log: The log to measure.

    Returns:
        The 1-based index of the last entry, or 0 if `log` is empty.
    """
    return len(log)


def last_log_term(log: list[LogEntry]) -> int:
    """Return the term of the last entry in the log.

    Paired with `last_log_index`, this is exactly the (term, index) fact
    ELECT-7 requires a RequestVote RPC to carry and ELECT-10's up-to-date
    comparison is defined over.

    Args:
        log: The log to inspect.

    Returns:
        The term of the last entry, or 0 if `log` is empty.
    """
    return log[-1].term if log else 0


def log_matches(log: list[LogEntry], prev_log_index: int, prev_log_term: int) -> bool:
    """Check whether the log already agrees with the leader at a position.

    Implements REPL-5's consistency check: "a Follower shall reject an
    AppendEntries RPC whenever its own log does not contain an entry at
    the RPC's previous-entry index whose term matches the RPC's
    previous-entry term." This function IS that check, phrased as the
    acceptance condition rather than the rejection one — a caller rejects
    the RPC precisely when this returns False.

    `prev_log_index == 0` always returns True, because index 0 means "the
    Leader is proposing to replace everything from the very start of the
    log" — there is no preceding entry for the two logs to agree on, so
    nothing can disagree either. This is what lets a brand-new,
    empty-log Follower accept its very first AppendEntries.

    For any other `prev_log_index`, the check is exactly the single-point
    comparison the Log Matching Property (§5.3) says is sufficient: if
    `log` doesn't even have an entry that far in, or the entry it has
    there was written in a different term, the two logs cannot be assumed
    to agree before that point either, so the RPC is rejected outright
    rather than trusting a shorter or differently-originated prefix.

    Args:
        log: The log being checked (typically the follower's own log).
        prev_log_index: The 1-based index of the entry immediately
            preceding the entries under consideration, as carried by an
            AppendEntries RPC. 0 means there is no preceding entry.
        prev_log_term: The term the entry at `prev_log_index` is expected
            to have.

    Returns:
        True if `log` already agrees with the leader at `prev_log_index`
        (or `prev_log_index` is 0), False otherwise.
    """
    if prev_log_index == 0:
        return True
    if prev_log_index > len(log):
        return False
    return log[prev_log_index - 1].term == prev_log_term


def log_after_append_entries(
    log: list[LogEntry], prev_log_index: int, entries: list[LogEntry]
) -> list[LogEntry]:
    """Compute the log that results from applying an AppendEntries RPC.

    Implements REPL-8 — "once an AppendEntries RPC is accepted, the
    Leader's replication logic shall overwrite any conflicting entries
    already present in that Follower's log with the Leader's own
    entries" — by applying the two receiver rules Figure 2 of the paper
    actually specifies this as: the moment an existing entry conflicts
    with an incoming one (same position, different term), delete that
    entry and everything after it; then append whatever of the incoming
    entries didn't already fit.

    Deliberately does NOT delete anything outside that rule: an entry
    beyond the range `entries` covers, or one that already matches its
    incoming counterpart term-for-term, is left exactly as it was. This is
    why a stale, uncommitted tail entry from an old, abandoned Leader
    (Figure 7, scenarios (c) and (d) of the paper) survives an ordinary
    heartbeat untouched — a heartbeat carries no new entries to conflict
    with it — and is only ever overwritten once the current Leader
    actually produces a genuinely conflicting entry at that same position.

    Callers must have already confirmed
    `log_matches(log, prev_log_index, prev_log_term)` — REPL-5 — before
    calling this; it does not repeat that check itself, and calling it
    against a `prev_log_index` the two logs don't actually agree on yet
    would silently corrupt the result rather than raise.

    Does not mutate `log` — consistent with `LogEntry` being frozen,
    nothing in this module ever mutates a log, it only ever computes what
    the log should become next.

    Args:
        log: The log to start from (typically the follower's own log).
            Not mutated.
        prev_log_index: The 1-based index of the entry immediately
            preceding `entries`, as already confirmed by a prior
            `log_matches` call.
        entries: The leader's entries to reconcile into `log`, in order,
            starting immediately after `prev_log_index`.

    Returns:
        A new list of LogEntry representing the log after applying the
        AppendEntries RPC.
    """
    new_log = list(log)
    for offset, entry in enumerate(entries):
        position = prev_log_index + offset
        if position < len(new_log):
            if new_log[position].term != entry.term:
                new_log = new_log[:position] + [entry]
        else:
            new_log.append(entry)
    return new_log


def next_index_after_rejection(next_index: int) -> int:
    """Compute the nextIndex to retry with after an AppendEntries rejection.

    Implements REPL-6: "whenever a Leader's AppendEntries RPC is
    rejected under REPL-5, the Leader shall decrement its stored
    nextIndex for that Follower." The floor at 1 exists because
    `log_matches` already treats `prev_log_index == 0` as automatically
    satisfied — nextIndex can never usefully fall below 1, since index 0
    needs no agreement check at all, and decrementing past it would just
    repeat an already-guaranteed-to-succeed probe forever.

    This is only half of the repair loop REPL-6 and REPL-7 describe
    together: the retrying itself — calling this, then `log_matches`
    again with the new, lower index, and repeating until it succeeds — is
    REPL-7's job, and is therefore the caller's responsibility, not this
    function's.

    Args:
        next_index: The Leader's current nextIndex for the follower
            whose AppendEntries RPC was just rejected.

    Returns:
        The decremented nextIndex, floored at 1.
    """
    return max(1, next_index - 1)
