"""A log's last entry, reduced to just what ELECT-10's comparison needs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LogPosition:
    """The (term, index) of a log's last entry, used to compare recency.

    ELECT-9/ELECT-10 never need to inspect more than a log's last entry
    to decide whether one log is at least as up to date as another, so
    this holds only the two facts that comparison operates on, not a full
    log. `Log.last_position` produces one of
    these for a real log, and both sides of a RequestVote RPC's own
    last-log-index/term fields (ELECT-7) reduce to exactly this pair too.

    Attributes:
        term: The term of the log's last entry (0 for an empty log).
        index: The 1-based index of the log's last entry (0 for an empty
            log).
    """

    term: int
    index: int

    def is_at_least_as_up_to_date_as(self, other: "LogPosition") -> bool:
        """Check whether this position is at least as up to date as another.

        Implements ELECT-10's rule exactly as stated: "a log is more up
        to date than another if its last entry has a later term; if the
        last entries share a term, the longer log is more up to date."
        Term is checked first and decides the outcome outright whenever
        the two terms differ — index is only ever consulted as a
        tiebreaker once the terms already match.

        This ordering is what keeps committed entries safe across
        elections. A committed entry is stored on a majority of nodes,
        and a candidate needs votes from a majority, so at least one of
        its voters always holds every committed entry (ELECT-12).
        Comparing last terms first makes that voter refuse any candidate
        whose log could be missing it.

        A simpler "whichever log is longer wins" rule breaks this. Take
        five nodes, A to E, all holding one entry from term 1:

        - A becomes Leader for term 2, appends three entries to its own
          log only, and crashes. A's log is now [1, 2, 2, 2] (each
          entry's term, by index).
        - B becomes Leader for term 3 and commits one new entry on B, C,
          and D, whose logs are now [1, 3]. E still has [1].
        - B crashes, A comes back, and A runs for Leader in term 4.

        By length, A's log beats everyone's, so C, D, and E would all
        vote for it, and as Leader it would overwrite the committed
        term-3 entry at index 2 with its own term-2 entry. Comparing
        last terms first, C and D see A's term 2 against their own term
        3 and refuse; A gets only its own vote and E's, which is short
        of a majority, and the committed entry survives.

        Args:
            other: The position being compared against — typically the
                voter's own last log position, when `self` is the
                candidate's.

        Returns:
            True if this position is at least as up to date as `other`,
            including a tie (equal term, equal index) — ELECT-9 grants a
            vote to a candidate "at least as" up to date as the voter,
            so a tie must count.
        """
        if self.term != other.term:
            return self.term > other.term
        return self.index >= other.index
