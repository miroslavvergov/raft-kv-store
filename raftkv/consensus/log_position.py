"""A log's last entry, reduced to what ELECT-10's comparison needs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LogPosition:
    """The (term, index) of a log's last entry: all that ELECT-9/ELECT-10 compare.

    Comes from `Log.last_position` for a node's own log and from
    `RequestVoteRequest.last_log_position` for a Candidate's (ELECT-7).

    Attributes:
        term: The last entry's term; 0 for an empty log.
        index: The last entry's 1-based index; 0 for an empty log.
    """

    term: int
    index: int

    def is_at_least_as_up_to_date_as(self, other: "LogPosition") -> bool:
        """Check whether this position is at least as up to date as `other` (ELECT-10).

        A later last term wins outright; the index decides only between equal
        terms. Term comes first because every majority includes a node holding each
        committed entry (ELECT-12), and that node must refuse any Candidate that
        could be missing one, which a longer log from an older term can be.

        Args:
            other: The position to compare against, typically the voter's own.

        Returns:
            True if `self` has a later term, or the same term and an index at least
            as high. A tie counts: ELECT-9 grants a vote to a Candidate at least as
            up to date as the voter.
        """
        if self.term != other.term:
            return self.term > other.term
        return self.index >= other.index
