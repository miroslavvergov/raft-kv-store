"""The up-to-date log comparison used to decide RequestVote grants.

The single comparison ELECT-9 hands its vote-granting decision off to:
whether a candidate's log is at least as up to date as the voter's own. No
I/O, no asyncio — this takes the bare (term, index) facts about each
side's last log entry (as `last_log_term`/`last_log_index` would produce
them) rather than a full log, because the comparison never needs to look
at more than that single last entry.
"""


def is_log_up_to_date(
    candidate_last_term: int,
    candidate_last_index: int,
    voter_last_term: int,
    voter_last_index: int,
) -> bool:
    """Check whether a candidate's log is at least as up to date as a voter's.

    Implements ELECT-10's rule exactly as stated: "a log is more up to
    date than another if its last entry has a later term; if the last
    entries share a term, the longer log is more up to date." Term is
    checked first and decides the outcome outright whenever the two terms
    differ — length is only ever consulted as a tiebreaker once the terms
    already match.

    This ordering is what actually makes the safety argument work
    (§5.4.1): combined with quorum overlap (ELECT-12), it guarantees a
    winning candidate's log already contains every entry any earlier
    Leader could have committed, because a committed entry can only exist
    in a term already reflected by every sufficiently-up-to-date voter's
    own last term. A simpler "whichever log is longer wins" rule breaks
    this outright: a candidate can be safely electable with a log that is
    both shorter and, by a pure length comparison, "behind" a voter's,
    provided its last entry's term is later — see the note "Why Longest
    Log Wins Is the Wrong Rule for Raft Elections" in the accompanying
    notes for the worked counterexample this module's own test suite
    reproduces directly.

    Args:
        candidate_last_term: The term of the candidate's last log entry.
        candidate_last_index: The 1-based index of the candidate's last
            log entry (0 if its log is empty).
        voter_last_term: The term of the voter's own last log entry.
        voter_last_index: The 1-based index of the voter's own last log
            entry (0 if its log is empty).

    Returns:
        True if the candidate's log is at least as up to date as the
        voter's, including a tie (equal term, equal length) — ELECT-9
        grants a vote to a candidate "at least as" up to date as the
        voter, so a tie must count.
    """
    if candidate_last_term != voter_last_term:
        return candidate_last_term > voter_last_term
    return candidate_last_index >= voter_last_index
