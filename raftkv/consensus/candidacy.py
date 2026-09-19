"""One election attempt: the Candidate's term and the votes it has received."""

from raftkv.consensus.cluster import Cluster


class Candidacy:
    """The votes a Candidate has collected in one term.

    Starts with the Candidate's own vote, already persisted (ELECT-4, ELECT-5).
    An answer is recorded only if its RequestVote was sent in this term, since
    the voter may have given this term's vote to another node (ELECT-8). Only a
    voter's first answer is kept: a retried or duplicated reply (FAIL-2)
    repeats it, and a voter cannot change it within a term. The caller handles
    a higher term in an answer with `NodeState.handle_observed_term`.

    Attributes:
        term: The term this Candidate is running in.
        votes_granted: The members that granted their vote this term, the
            Candidate included.
        votes_refused: The members whose recorded answer this term is a refusal.
    """

    def __init__(self, term: int, candidate_id: int, cluster: Cluster) -> None:
        """Start counting, with the Candidate's own vote already in.

        Args:
            term: The term the Candidate is running in.
            candidate_id: The Candidate's node ID; its self-vote is persisted.
            cluster: The cluster whose majority must be reached.

        Raises:
            ValueError: If `candidate_id` is not a member of `cluster`.
        """
        if candidate_id not in cluster.members:
            raise ValueError(f"node {candidate_id} is not a member of {sorted(cluster.members)}")
        self._term = term
        self._cluster = cluster
        self._answers: dict[int, bool] = {candidate_id: True}

    @property
    def term(self) -> int:
        return self._term

    @property
    def votes_granted(self) -> frozenset[int]:
        return frozenset(voter for voter, granted in self._answers.items() if granted)

    @property
    def votes_refused(self) -> frozenset[int]:
        return frozenset(voter for voter, granted in self._answers.items() if not granted)

    @property
    def has_majority(self) -> bool:
        """Whether the granted votes form a strict majority (ELECT-11, ELECT-12)."""
        return self._cluster.is_majority(self.votes_granted)

    def record_vote(self, voter: int, sent_in_term: int, granted: bool) -> bool:
        """Record a voter's answer if it answers this term's request and is its first.

        A refusal goes to `votes_refused` and still uses up the voter's one answer.

        Args:
            voter: The ID of the node that answered.
            sent_in_term: The term the answered RequestVote was sent in.
            granted: Whether the voter granted its vote.

        Returns:
            True if recorded; False if ignored (another term, or already answered).

        Raises:
            KeyError: If `sent_in_term` is this term but `voter` is not a member.
        """
        if sent_in_term != self._term:
            return False
        if voter not in self._cluster.members:
            raise KeyError(voter)
        if voter in self._answers:
            return False
        self._answers[voter] = granted
        return True
