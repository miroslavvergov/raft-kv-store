"""The RequestVote RPC: what a Candidate sends, and what a voter answers."""

from dataclasses import dataclass

from raftkv.consensus.log_position import LogPosition


@dataclass(frozen=True)
class RequestVoteRequest:
    """A Candidate's request for one node's vote in one term.

    A Candidate sends the same request to every other member (ELECT-6)
    right after persisting its new term and its vote for itself (ELECT-5).

    It carries the facts about the Candidate's log that the voter needs
    for ELECT-9: the index and term of the Candidate's last log entry
    (ELECT-7). Comparing just those two numbers is enough to tell whether
    the Candidate's log is at least as up to date as the voter's
    (ELECT-10), so the log itself is never sent.

    Attributes:
        term: The term the Candidate is running in — its own
            `current_term`, just incremented (ELECT-3).
        candidate_id: The Candidate's node ID. A voter that grants the
            vote records this as its `voted_for`.
        last_log_index: The index of the Candidate's last log entry (0
            for an empty log).
        last_log_term: The term of the Candidate's last log entry (0 for
            an empty log).
    """

    term: int
    candidate_id: int
    last_log_index: int
    last_log_term: int

    @property
    def last_log_position(self) -> LogPosition:
        """The Candidate's last log entry, as the LogPosition ELECT-10 compares."""
        return LogPosition(term=self.last_log_term, index=self.last_log_index)


@dataclass(frozen=True)
class RequestVoteResponse:
    """A voter's answer to one RequestVoteRequest.

    Attributes:
        term: The voter's `current_term` after handling the request.
            A Candidate that finds this higher than its own term has
            fallen behind: it catches up and stops campaigning (STATE-4,
            STATE-5). When the vote is granted, this equals the request's
            term.
        vote_granted: True if the voter gave its vote for the request's
            term to the Candidate.
    """

    term: int
    vote_granted: bool
