"""One log entry's position: the term it carries and its index."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LogPosition:
    """The (term, index) of one log entry, which identifies it across every log.

    Two logs holding an entry with the same term at the same index hold the same
    entry (REPL-5). A log's last position is all ELECT-9 and ELECT-10 compare: it
    comes from `Log.last_position` for a node's own log and from
    `RequestVoteRequest.last_log_position` for a Candidate's (ELECT-7). A newly
    appended entry's position lets its caller tell later whether that entry, and
    not another in its place, was committed.

    Attributes:
        term: The entry's term; 0 for the position before an empty log's first entry.
        index: The entry's 1-based index; 0 for that same position.
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
