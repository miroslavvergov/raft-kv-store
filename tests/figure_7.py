"""Figure 7 of the Raft paper, as test fixtures.

A leader for term 8, and the six follower logs (a)-(f) it might find on
coming to power — transcribed from the paper's own diagram (page 7 of the
extended version). Terms are listed by 1-based index, matching the paper's
labeling.
"""

from raftkv.consensus import Log, LogEntry

LEADER_TERMS = [1, 1, 1, 4, 4, 5, 5, 6, 6, 6]

FOLLOWER_TERMS = {
    "a": [1, 1, 1, 4, 4, 5, 5, 6, 6],  # missing entry 10
    "b": [1, 1, 1, 4],  # missing entries 5-10
    "c": [1, 1, 1, 4, 4, 5, 5, 6, 6, 6, 6],  # extra uncommitted entry 11 (term 6)
    "d": [1, 1, 1, 4, 4, 5, 5, 6, 6, 6, 7, 7],  # extra entries 11-12 (term 7, 7)
    "e": [1, 1, 1, 4, 4, 4, 4],  # diverges at index 6 (term 4 vs leader's term 5)
    "f": [1, 1, 1, 2, 2, 2, 3, 3, 3, 3],  # diverges at index 4 (term 2 vs leader's term 4)
}


def make_log(terms: list[int]) -> Log:
    """Build a log with one entry per term, commanded "cmd1", "cmd2", ... by position.

    Commands depend only on position, so two logs built this way hold
    equal entries wherever their terms agree — consistent with the Log
    Matching Property.
    """
    return Log([LogEntry(term=t, command=f"cmd{i + 1}") for i, t in enumerate(terms)])
