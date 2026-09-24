"""A Leader's log, six follower logs that each differ from it, and the Leader's repair loop.

Each list gives the term of each entry, by 1-based index. The Leader leads term 8 and holds
10 entries. The followers are missing entries, hold extra ones, or both, across several terms.
"""

from dataclasses import dataclass

from raftkv.consensus import FollowerProgress, Log, LogEntry

LEADER_TERMS = [1, 1, 1, 4, 4, 5, 5, 6, 6, 6]

FOLLOWER_TERMS = {
    # Up to date except for the Leader's last entry.
    "missing_last_entry": [1, 1, 1, 4, 4, 5, 5, 6, 6],
    # Stopped receiving entries after index 4.
    "missing_last_six_entries": [1, 1, 1, 4],
    # Plus entry 11 from term 6: the term-6 Leader sent it only here, then crashed.
    "one_extra_stale_entry": [1, 1, 1, 4, 4, 5, 5, 6, 6, 6, 6],
    # Plus entries 11-12 from term 7: this node led term 7 and crashed before replicating them.
    "two_extra_stale_entries": [1, 1, 1, 4, 4, 5, 5, 6, 6, 6, 7, 7],
    # Agrees through index 5; its never-committed term-4 entries 6-7 conflict with the
    # Leader's, and it is missing 8-10.
    "conflicts_from_index_6": [1, 1, 1, 4, 4, 4, 4],
    # Agrees through index 3; it led terms 2 and 3 and crashed each time before committing,
    # so all seven of its later entries conflict with the Leader's.
    "conflicts_from_index_4": [1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
}


def make_log(terms: list[int]) -> Log:
    """Build a log with one entry per term, commanded "cmd1", "cmd2", ... by position.

    Commands depend only on position, so two logs built this way hold equal entries wherever
    their terms agree, as two real logs do at an index where the terms agree.
    """
    return Log([LogEntry(term=t, command=f"cmd{i + 1}") for i, t in enumerate(terms)])


def append_entries_for(leader_log: Log, next_index: int) -> tuple[int, int, list[LogEntry]]:
    """Return the (prev_log_index, prev_log_term, entries) a Leader sends at `next_index`."""
    prev_log_index = next_index - 1
    return prev_log_index, leader_log.term_at(prev_log_index), leader_log.entries_from(next_index)


@dataclass(frozen=True)
class Repair:
    """How the Leader's repair loop ended for one follower.

    Attributes:
        progress: The Leader's record of the follower after the accepted AppendEntries.
        rejections: How many AppendEntries the follower rejected first.
        agreed_at: The prev_log_index of the accepted AppendEntries.
        repaired_log: The follower's log after it accepted.
    """

    progress: FollowerProgress
    rejections: int
    agreed_at: int
    repaired_log: Log


def repair(leader_log: Log, follower_log: Log) -> Repair:
    """Probe, back off one index per rejection, and append, until `follower_log` accepts.

    A probe with prev_log_index 0 always succeeds, so more rejections than the Leader has
    entries fails the test instead of looping forever.
    """
    progress = FollowerProgress(next_index=leader_log.last_index + 1)
    rejections = 0
    while True:
        prev_log_index, prev_log_term, entries = append_entries_for(leader_log, progress.next_index)
        if follower_log.matches(prev_log_index, prev_log_term):
            progress.record_success(prev_log_index, len(entries))
            repaired_log = follower_log.after_append_entries(prev_log_index, entries)
            return Repair(progress, rejections, prev_log_index, repaired_log)
        progress.record_rejection()
        rejections += 1
        assert rejections <= leader_log.last_index, "never reached an index where the logs agree"
