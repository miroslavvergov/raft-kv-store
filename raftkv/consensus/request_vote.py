"""The RequestVote RPC: what a Candidate sends, and what a voter answers."""

from dataclasses import dataclass

from raftkv.consensus.log_position import LogPosition


@dataclass(frozen=True)
class RequestVoteRequest:
    """A Candidate's request for one node's vote in one term.

    Sent to every other member (ELECT-6) once the Candidate's new term and
    self-vote are persisted (ELECT-5). It carries only the index and term of the
    Candidate's last entry (ELECT-7), which is all ELECT-9/ELECT-10 compare.

    Attributes:
        term: The Candidate's `current_term`, just incremented (ELECT-3).
        candidate_id: The Candidate's node ID; a voter that grants records it
            as `voted_for`.
        last_log_index: The index of the Candidate's last entry; 0 if empty.
        last_log_term: The term of the Candidate's last entry; 0 if empty.

    Raises:
        ValueError: If `term` is below 1; the last index or term is negative;
            exactly one of them is 0; or `last_log_term` exceeds `term`, which no
            Candidate's log can hold.
    """

    term: int
    candidate_id: int
    last_log_index: int
    last_log_term: int

    def __post_init__(self) -> None:
        if self.term < 1:
            raise ValueError(f"term must be at least 1, got {self.term}")
        if self.last_log_index < 0 or self.last_log_term < 0:
            raise ValueError(
                f"last_log_index and last_log_term cannot be negative, got "
                f"{self.last_log_index} and {self.last_log_term}"
            )
        if (self.last_log_index == 0) != (self.last_log_term == 0):
            raise ValueError(
                f"last_log_term is 0 exactly when last_log_index is, got "
                f"{self.last_log_index} and {self.last_log_term}"
            )
        if self.last_log_term > self.term:
            raise ValueError(
                f"last_log_term {self.last_log_term} cannot exceed the term {self.term}"
            )

    @property
    def last_log_position(self) -> LogPosition:
        """The Candidate's last log entry, as the LogPosition ELECT-10 compares."""
        return LogPosition(term=self.last_log_term, index=self.last_log_index)


@dataclass(frozen=True)
class RequestVoteResponse:
    """A voter's answer to one RequestVoteRequest.

    Attributes:
        term: The voter's `current_term` after handling the request; equal to
            the request's term whenever the vote is granted. A Candidate seeing
            a higher term catches up and stops campaigning (STATE-4, STATE-5).
        vote_granted: Whether the voter gave the Candidate its vote for the
            request's term.
    """

    term: int
    vote_granted: bool
