"""The Follower/Candidate/Leader role state machine and its term bookkeeping.

Implements STATE-1 through STATE-6, with the term/vote side effects of
ELECT-3, ELECT-4, and ELECT-11 folded into the same methods that cause
them — because, per the requirements themselves, they are one atomic
event each, not several independently-triggerable ones. No I/O, no
asyncio, no persistence: `current_term` and `voted_for` are held here as
ordinary in-memory attributes; a later, stateful layer is responsible for
persisting them (PERSIST-1/PERSIST-2) and for guarding this same state
against concurrent mutation across an `await` (DD-8) once this class is
driven by real network I/O instead of direct method calls.
"""

from typing import Optional

from raftkv.consensus.errors import IllegalTransition
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
        That vote-counting itself happens outside this class, wherever
        RequestVote responses are actually being tallied; this method
        only enacts the role change once that decision has already been
        made.

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
