"""The Follower/Candidate/Leader role state machine and its term bookkeeping.

Implements STATE-1 through STATE-7, with the term/vote side effects of
ELECT-3, ELECT-4, and ELECT-11 folded into the same methods that cause
them — because, per the requirements themselves, they are one atomic
event each, not several independently-triggerable ones — and the voter's
side of an election, ELECT-8 through ELECT-10. No I/O, no asyncio, no
persistence: `current_term` and `voted_for` are held here as ordinary
in-memory attributes; `DurableNodeState` persists them (PERSIST-1,
PERSIST-2) and guards this same state against concurrent mutation across
an `await` (DD-8).
"""

from typing import Optional

from raftkv.consensus.errors import IllegalTransition
from raftkv.consensus.log_position import LogPosition
from raftkv.consensus.request_vote import RequestVoteRequest, RequestVoteResponse
from raftkv.consensus.role import Role


class NodeState:
    """One node's Raft role, current term, and vote.

    Attributes:
        node_id: This node's permanent identity (NODE-8, DD-20): a
            positive integer, distinct from its network address, that
            never changes across restarts. Recorded as `voted_for` when
            the node votes for itself on becoming a Candidate.
        role: The node's current Role (STATE-1).
        current_term: The node's current term.
        voted_for: The node ID this node voted for in `current_term`, or
            None if it hasn't voted yet this term.
    """

    def __init__(self, node_id: int) -> None:
        """Initialize a brand-new node in the Follower role with no term history.

        STATE-2: a node starts in the Follower role. A node that has
        never run before has no opinion about any term yet, so it starts
        at `current_term = 0` with no vote. A node restarting from
        persisted state is built with `reloaded` instead.

        Args:
            node_id: This node's permanent positive-integer identity.
        """
        self.node_id = node_id
        self.role = Role.FOLLOWER
        self.current_term = 0
        self.voted_for: Optional[int] = None

    @classmethod
    def reloaded(
        cls, node_id: int, current_term: int, voted_for: Optional[int]
    ) -> "NodeState":
        """Build a node from the term and vote it had persisted before restarting.

        STATE-2: a node starts in the Follower role immediately after
        reloading its persisted state (PERSIST-4, PERSIST-5) — and that
        holds regardless of the role it had before it stopped. A node
        that crashed while a Candidate, or even while Leader, comes back
        as a Follower; leadership is never persisted and is only ever
        regained by winning a new election.

        The reloaded `current_term` and `voted_for` are restored exactly
        as they were. This is what prevents the double vote illustrated
        under PERSIST-1/PERSIST-2: a node that already voted in some term
        before crashing still remembers that vote afterwards, and so
        refuses a second, different vote in that same term.

        For a node that has never run before, the store reports
        `current_term = 0` and no vote, which makes the result identical
        to constructing a new `NodeState` directly.

        Args:
            node_id: This node's permanent positive-integer identity.
            current_term: The term reloaded from stable storage.
            voted_for: The vote reloaded from stable storage, or None if
                the node had not voted in `current_term`.

        Returns:
            A NodeState in the Follower role carrying the reloaded term
            and vote.
        """
        state = cls(node_id)
        state.current_term = current_term
        state.voted_for = voted_for
        return state

    def become_candidate(self) -> None:
        """Transition to Candidate, incrementing the term and voting for self.

        Implements the Follower-to-Candidate and Candidate-to-Candidate
        edges of STATE-3, together with ELECT-3 and ELECT-4 — the two
        things the requirements say must happen "upon becoming
        Candidate": incrementing `current_term` and voting for self.

        Raises:
            IllegalTransition: If the node's current role is not
                Follower or Candidate — a Leader cannot become a
                Candidate without first reverting to Follower (see
                STATE-4). Raised before anything is mutated, so a
                rejected call leaves `current_term`, `voted_for`, and
                `role` exactly as they were.
        """
        if self.role not in (Role.FOLLOWER, Role.CANDIDATE):
            raise IllegalTransition(
                f"{self.role.value} -> candidate is not a legal STATE-3 edge"
            )
        self.role = Role.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id

    def become_leader(self) -> None:
        """Transition to Leader.

        Implements STATE-3's Candidate-to-Leader edge — the transition
        ELECT-11 triggers once a Candidate has collected votes from a
        majority of the cluster, including itself, in the same term.
        Those votes are counted by a `Candidacy`; this method only
        enacts the role change once that decision has already been made.

        Raises:
            IllegalTransition: If the node's current role is not
                Candidate, since Candidate-to-Leader is the only edge in
                STATE-3 that ends in Leader. State is left untouched.
        """
        if self.role is not Role.CANDIDATE:
            raise IllegalTransition(
                f"{self.role.value} -> leader is not a legal STATE-3 edge"
            )
        self.role = Role.LEADER

    def handle_observed_term(self, term: int) -> bool:
        """Catch up current_term, voted_for, and role if `term` is higher.

        The one trigger STATE-4, STATE-5, and STATE-6 all fire from:
        observing, in any RPC or RPC response, a term higher than this
        node's own `current_term`. All three consequences of that
        trigger take effect in this single call.

        Does nothing if `term` is not strictly greater than
        `current_term` — STATE-4, STATE-5, and STATE-6 all say "higher
        than", never "higher than or equal to", so a term the node has
        already caught up to (including its own current term) must
        never re-trigger a step-down or re-clear an already-current
        vote.

        When it does fire: `current_term` is raised to `term` (STATE-5,
        true regardless of role — even a plain Follower must update
        it), `voted_for` is reset to unset in the same step (STATE-6 —
        a vote recorded against the old term means nothing once the
        term has moved on, and leaving it in place would risk
        withholding a vote in the new term the node is actually free to
        give), and if the node was a Candidate or a Leader it reverts
        to Follower (STATE-4 — the role consequence, scoped to those
        two roles specifically, because a plain Follower has no role
        left to give up).

        Args:
            term: The term observed in an incoming RPC or RPC response.

        Returns:
            True if `term` was higher than `current_term` and the
            catch-up was applied, False if `term` was not higher and
            nothing changed.
        """
        if term <= self.current_term:
            return False
        self.current_term = term
        self.voted_for = None
        if self.role in (Role.CANDIDATE, Role.LEADER):
            self.role = Role.FOLLOWER
        return True

    def handle_vote_request(
        self, request: RequestVoteRequest, own_last_log: LogPosition
    ) -> RequestVoteResponse:
        """Decide whether to grant a Candidate's vote request, recording the vote if granted.

        Everything a node does on receiving a RequestVote, in order:

        1. Catch up to the request's term if it is higher
           (`handle_observed_term`: STATE-4, STATE-5, STATE-6). A
           Candidate or Leader steps down, and the vote cast in the old
           term is cleared, leaving this node free to vote in the new
           one.
        2. Refuse if the request's term is lower than `current_term`.
           The node remembers only its vote for `current_term`; whatever
           it did in an earlier term is gone, so it can no longer tell
           whether it already voted in that term, and granting could
           give a second vote in one term — what ELECT-8 forbids.
        3. Refuse if it has already voted for a different node in this
           term (ELECT-8). A repeated request from the node it already
           voted for is granted again: the request was retried (FAIL-2)
           or the first answer was lost, and answering it the same way
           twice gives no second vote (FAIL-1).
        4. Refuse unless the Candidate's last log entry is at least as up
           to date as this node's own (ELECT-9), by ELECT-10's rule
           (`LogPosition.is_at_least_as_up_to_date_as`). A committed
           entry is on a majority of nodes, and every majority includes
           at least one of them, so this is what stops a Candidate that
           is missing a committed entry from winning.
        5. Otherwise grant: `voted_for` becomes the Candidate.

        A Candidate or Leader still in the request's term refuses at step
        3, since it voted for itself in that term.

        The answer carries `current_term` as it stands after step 1, so a
        Candidate running in an old term learns the newer one and stops
        campaigning.

        Nothing is persisted here. Whenever this call changes
        `current_term` or `voted_for`, the caller must persist them
        before sending the answer (PERSIST-1, PERSIST-2): a vote that was
        sent but never saved would be forgotten on restart, and the node
        could then vote for a different Candidate in the same term. A
        caller that grants a vote must also reset its election timeout
        (ELECT-13).

        Args:
            request: The Candidate's RequestVote.
            own_last_log: This node's own last log entry, as
                `Log.last_position` reports it.

        Returns:
            The answer to send back to the Candidate.
        """
        self.handle_observed_term(request.term)
        granted = (
            request.term == self.current_term
            and self.voted_for in (None, request.candidate_id)
            and request.last_log_position.is_at_least_as_up_to_date_as(own_last_log)
        )
        if granted:
            self.voted_for = request.candidate_id
        return RequestVoteResponse(term=self.current_term, vote_granted=granted)

    def recognize_leader(self, term: int) -> bool:
        """Recognize the sender of an AppendEntries as its term's Leader, if the term allows it.

        What the term of an incoming AppendEntries RPC requires on its
        own, before any of its entries are looked at:

        - A term higher than `current_term` is caught up to first
          (`handle_observed_term`: STATE-4, STATE-5, STATE-6), and its
          sender is then recognized.
        - A term lower than `current_term` comes from a Leader that has
          since been replaced. Its sender is not recognized and nothing
          changes; the caller rejects the RPC, and the reply's
          `current_term` tells that old Leader it is out of date.
        - A term equal to `current_term` comes from the one Leader this
          term can have: each node votes at most once per term (ELECT-8)
          and winning takes a majority (ELECT-11), so two nodes can never
          both win it. A Candidate in this term has therefore lost the
          election and reverts to Follower (STATE-7), keeping
          `current_term` and its vote for itself — clearing that vote
          would leave it free to vote a second time in the same term. A
          Follower stays a Follower.

        A Leader never receives an AppendEntries in its own term, since
        that would take a second Leader in the same term. If one arrives
        anyway, its sender is not recognized and this node stays Leader,
        so no other node can overwrite a Leader's log under the Leader's
        own term.

        Args:
            term: The term carried by the AppendEntries RPC.

        Returns:
            True if the sender is recognized as the Leader of `term` —
            this node is now a Follower in that term, and the caller goes
            on to check and append the entries — or False if the RPC
            must be rejected.
        """
        self.handle_observed_term(term)
        if term < self.current_term or self.role is Role.LEADER:
            return False
        self.role = Role.FOLLOWER
        return True
