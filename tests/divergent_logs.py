"""A Leader's log, and six follower logs that each differ from it in a different way.

Every list gives the term of each entry, by 1-based index. The Leader is
the Leader for term 8 and holds 10 entries. Together, the followers cover
the ways a follower's log can differ from a newly elected Leader's: it can
be missing entries, hold extra entries the Leader doesn't have, or both —
and the missing or extra entries can span several terms.
"""

from raftkv.consensus import Log, LogEntry

LEADER_TERMS = [1, 1, 1, 4, 4, 5, 5, 6, 6, 6]

FOLLOWER_TERMS = {
    # Up to date except for the Leader's last entry.
    "missing_last_entry": [1, 1, 1, 4, 4, 5, 5, 6, 6],
    # Stopped receiving entries after index 4, so it is missing 5 to 10.
    "missing_last_six_entries": [1, 1, 1, 4],
    # Matches the Leader, plus entry 11 from term 6: the term-6 Leader
    # appended it and sent it only here before crashing.
    "one_extra_stale_entry": [1, 1, 1, 4, 4, 5, 5, 6, 6, 6, 6],
    # Matches the Leader, plus entries 11-12 from term 7: this node led
    # term 7, appended two entries, and crashed before replicating them.
    "two_extra_stale_entries": [1, 1, 1, 4, 4, 5, 5, 6, 6, 6, 7, 7],
    # Agrees through index 5, then holds two never-committed term-4 entries
    # (6-7) that conflict with the Leader's term-5 entries, and is missing
    # 8 to 10.
    "conflicts_from_index_6": [1, 1, 1, 4, 4, 4, 4],
    # Agrees through index 3. This node led term 2 and appended entries 4-6,
    # crashed, then led term 3 and appended 7-10, and crashed again before
    # committing any of them — so all seven conflict with the Leader's.
    "conflicts_from_index_4": [1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
}


def make_log(terms: list[int]) -> Log:
    """Build a log with one entry per term, commanded "cmd1", "cmd2", ... by position.

    Commands depend only on position, so two logs built this way hold
    equal entries wherever their terms agree — just as two real logs do
    when an entry has the same index and term in both.
    """
    return Log([LogEntry(term=t, command=f"cmd{i + 1}") for i, t in enumerate(terms)])
