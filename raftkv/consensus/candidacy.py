"""One election attempt: the Candidate's term and the votes it has received."""

from raftkv.consensus.cluster import Cluster


class Candidacy:
    """The votes a Candidate has collected in one term.

    A node creates a Candidacy each time it becomes Candidate and discards
    it when it stops being one: on winning, on stepping down, or on
    starting a new election in the next term, which creates a new
    Candidacy. It starts with the Candidate's vote for itself (ELECT-4),
    which is already on disk by then (ELECT-5).

    `has_majority` becomes True once the voters that granted their vote
    make up a strict majority of the cluster, the Candidate included
    (ELECT-11, ELECT-12). Three rules decide which answers count:

    - **Only this term's answers.** A voter's answer is recorded only if
      it answers a RequestVote sent in this Candidacy's term. A grant
      from an earlier election says nothing about this one: that voter
      has one vote per term (ELECT-8), and in this term it may already
      have given it to someone else. The term compared is the one the
      request was sent in, and every other term is ignored, earlier or
      later.
    - **One answer per voter.** The first answer recorded from a voter is
      kept, and any later one from the same voter in this term is
      ignored. A request can be retried after a timeout (FAIL-2), and
      the network can deliver a reply more than once, so the same grant
      may arrive several times; it still counts as one vote. A voter
      cannot legitimately change its answer within a term, so ignoring
      later answers loses nothing.
    - **Only members count.** RequestVote goes only to members, so an
      answer from any other node is a caller error and raises
      `KeyError`; it can never become part of a majority.

    Separately, an answer carrying a term higher than this Candidacy's
    means another election has already moved on, and the node must step
    down. That concerns the node's role and term, and is handled by
    `NodeState.handle_observed_term`, not here.

    Attributes:
        term: The term this Candidate is running in.
        votes_granted: The IDs of every node that granted its vote in
            this term, the Candidate itself included.
        votes_refused: The IDs of every node whose recorded answer in
            this term was a refusal.
    """

    def __init__(self, term: int, candidate_id: int, cluster: Cluster) -> None:
        """Start counting votes, with the Candidate's own vote already in.

        Args:
            term: The term the Candidate is running in.
            candidate_id: The Candidate's own node ID. It has already
                voted for itself and persisted that vote.
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
        """Whether the votes granted so far make up a strict majority of the cluster."""
        return self._cluster.is_majority(self.votes_granted)

    def record_vote(self, voter: int, sent_in_term: int, granted: bool) -> bool:
        """Record a voter's answer — if it belongs to this Candidacy and is the voter's first.

        If `sent_in_term` is not this Candidacy's term, or this voter's
        answer for this term has already been recorded, nothing changes.
        Otherwise the answer is kept: a grant adds the voter to
        `votes_granted`, and a refusal adds nothing but still counts as
        the voter's one answer for this term.

        Args:
            voter: The node ID of the node that answered.
            sent_in_term: The term the RequestVote being answered was
                sent in.
            granted: Whether the voter granted its vote.

        Returns:
            True if the answer was recorded, False if it was ignored.

        Raises:
            KeyError: If the request was sent in this term but `voter`
                is not a member of the cluster.
        """
        if sent_in_term != self._term:
            return False
        if voter not in self._cluster.members:
            raise KeyError(voter)
        if voter in self._answers:
            return False
        self._answers[voter] = granted
        return True
