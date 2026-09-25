"""A node's role state machine with its term and vote (STATE-1 through STATE-7).

Each method applies one event with all its effects in one step, e.g.
becoming Candidate also increments the term and votes for self (ELECT-3,
ELECT-4). Nothing is persisted here: `DurableNodeState` persists term and
vote (PERSIST-1, PERSIST-2) and serializes changes under its lock (DD-8).
"""

from raftkv.consensus.errors import IllegalTransitionError
from raftkv.consensus.log_position import LogPosition
from raftkv.consensus.request_vote import RequestVoteRequest, RequestVoteResponse
from raftkv.consensus.role import Role


class NodeState:
    """One node's Raft role, current term, and vote.

    All three change only through the methods below, each one event with all its
    effects, so no caller can reach a role by a transition STATE-3 forbids.

    Attributes:
        node_id: The node's permanent positive-integer identity, distinct from
            its network address (NODE-8, DD-20).
        role: The node's current Role (STATE-1).
        current_term: The node's current term.
        voted_for: The node voted for in `current_term`, or None.
    """

    def __init__(self, node_id: int) -> None:
        """Create a never-run node: a Follower (STATE-2) at term 0 with no vote.

        Use `reloaded` for a node restarting from persisted state.

        Args:
            node_id: The node's permanent positive-integer identity.
        """
        self._node_id = node_id
        self._role = Role.FOLLOWER
        self._current_term = 0
        self._voted_for: int | None = None

    @property
    def node_id(self) -> int:
        return self._node_id

    @property
    def role(self) -> Role:
        return self._role

    @property
    def current_term(self) -> int:
        return self._current_term

    @property
    def voted_for(self) -> int | None:
        return self._voted_for

    @classmethod
    def reloaded(cls, node_id: int, current_term: int, voted_for: int | None) -> "NodeState":
        """Build a restarting node from its persisted term and vote, as a Follower.

        The node is a Follower whatever role it held before stopping (STATE-2):
        role is not persisted, and leadership is regained only by a new election.
        Term and vote are restored exactly (PERSIST-4, PERSIST-5), so a vote cast
        before a crash still blocks a different vote in that term (PERSIST-1,
        PERSIST-2).

        Args:
            node_id: The node's permanent positive-integer identity.
            current_term: The persisted term; 0 for a node that never ran.
            voted_for: The persisted vote in `current_term`, or None.

        Returns:
            A Follower carrying the reloaded term and vote.
        """
        state = cls(node_id)
        state._current_term = current_term
        state._voted_for = voted_for
        return state

    def become_candidate(self) -> None:
        """Transition to Candidate, incrementing the term and voting for self.

        STATE-3's Follower-to-Candidate or Candidate-to-Candidate edge, with
        ELECT-3's increment of `current_term` and ELECT-4's self-vote in one step.

        Raises:
            IllegalTransitionError: If the node is Leader, which must step down to
                Follower first (STATE-4). Nothing changes.
        """
        if self._role not in (Role.FOLLOWER, Role.CANDIDATE):
            raise IllegalTransitionError(
                f"{self._role.value} -> candidate is not a legal STATE-3 edge"
            )
        self._role = Role.CANDIDATE
        self._current_term += 1
        self._voted_for = self._node_id

    def become_leader(self) -> None:
        """Transition to Leader: STATE-3's Candidate-to-Leader edge.

        Enacts a win a `Candidacy` has already counted (ELECT-11); counts no votes.

        Raises:
            IllegalTransitionError: If the node is not a Candidate. Nothing changes.
        """
        if self._role is not Role.CANDIDATE:
            raise IllegalTransitionError(
                f"{self._role.value} -> leader is not a legal STATE-3 edge"
            )
        self._role = Role.LEADER

    def handle_observed_term(self, term: int) -> bool:
        """Catch up to `term` if it is higher than `current_term`.

        A strictly higher term, seen in any RPC or RPC response, becomes
        `current_term` (STATE-5), clears `voted_for`, which belonged to the old term
        (STATE-6), and turns a Candidate or Leader into a Follower (STATE-4). An
        equal or lower term changes nothing, so a term already caught up to never
        re-triggers a step-down or clears a current vote.

        Args:
            term: The term seen in an incoming RPC or RPC response.

        Returns:
            True if `term` was higher and the catch-up was applied, False otherwise.
        """
        if term <= self._current_term:
            return False
        self._current_term = term
        self._voted_for = None
        if self._role in (Role.CANDIDATE, Role.LEADER):
            self._role = Role.FOLLOWER
        return True

    def handle_vote_request(
        self, request: RequestVoteRequest, own_last_log_position: LogPosition
    ) -> RequestVoteResponse:
        """Decide a Candidate's RequestVote, recording the vote if granted.

        In order:

        1. Catch up to a higher request term (`handle_observed_term`: STATE-4,
           STATE-5, STATE-6).
        2. Refuse a request from an earlier term: only the vote for `current_term`
           is remembered, so granting could give a second vote in that term
           (ELECT-8).
        3. Refuse if already voted for another node in this term (ELECT-8); a
           Candidate or Leader has voted for itself. A repeated request from the
           node already voted for passes this step: it is a retry (FAIL-2) or a
           lost answer, and granting it twice is not a second vote (FAIL-1).
        4. Refuse unless the Candidate's last entry is at least as up to date as
           `own_last_log_position` (ELECT-9, ELECT-10).
        5. Grant: `voted_for` becomes the Candidate.

        Nothing is persisted here: the caller persists any change to
        `current_term` or `voted_for` before sending the answer (PERSIST-1,
        PERSIST-2) and resets its election timeout on a grant (ELECT-13).

        Args:
            request: The Candidate's RequestVote.
            own_last_log_position: This node's `Log.last_position`.

        Returns:
            The answer, carrying `current_term` after step 1 so that a Candidate
            from an older term learns the newer one.
        """
        self.handle_observed_term(request.term)
        granted = (
            request.term == self._current_term
            and self._voted_for in (None, request.candidate_id)
            and request.last_log_position.is_at_least_as_up_to_date_as(own_last_log_position)
        )
        # NOTE: a refusal records no vote, so this term's vote stays free for a Candidate with
        # a complete log (ELECT-9).
        if granted:
            self._voted_for = request.candidate_id
        return RequestVoteResponse(term=self._current_term, vote_granted=granted)

    def recognize_leader(self, term: int) -> bool:
        """Recognize an AppendEntries sender as Leader of `term`, if the term allows it.

        Decided on the RPC's term alone, before any entries:

        - Higher than `current_term`: catch up first (`handle_observed_term`:
          STATE-4, STATE-5, STATE-6), then recognize.
        - Lower: the sender is an outdated Leader. It is not recognized, nothing
          changes, and the reply's `current_term` tells it so.
        - Equal: the sender is this term's only possible Leader, since each node
          votes once per term (ELECT-8) and winning takes a majority (ELECT-11). A
          Candidate reverts to Follower, keeping its term and self-vote (STATE-7);
          clearing the vote would allow a second vote in the term. A Leader, which
          cannot legitimately receive one, does not recognize it and stays Leader,
          so no node can overwrite its log under its own term.

        Args:
            term: The term carried by the AppendEntries RPC.

        Returns:
            True if recognized: this node is now a Follower in `term`, and the
            caller goes on to check and append the entries. False if the caller
            must reject the RPC.
        """
        self.handle_observed_term(term)
        # NOTE: the role check decides only at an equal term, a lower one being refused
        # already: no rival Leader can hold this Leader's term.
        if term < self._current_term or self._role is Role.LEADER:
            return False
        self._role = Role.FOLLOWER
        return True
