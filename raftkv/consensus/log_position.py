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

        This ordering is what actually makes the safety argument work
        (§5.4.1): combined with quorum overlap (ELECT-12), it guarantees
        a winning candidate's log already contains every entry any
        earlier Leader could have committed, because a committed entry
        can only exist in a term already reflected by every
        sufficiently-up-to-date voter's own last term. A simpler
        "whichever log is longer wins" rule breaks this outright: a
        candidate can be safely electable with a log that is both
        shorter and, by a pure length comparison, "behind" a voter's,
        provided its last entry's term is later — see the note "Why
        Longest Log Wins Is the Wrong Rule for Raft Elections" in the
        accompanying notes for the worked counterexample this module's
        own test suite reproduces directly.

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
