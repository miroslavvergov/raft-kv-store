"""A node's state at one moment, as the tracer compares it before and after a call."""

from dataclasses import dataclass, field, replace
from typing import Any, Optional

from raftkv.consensus import Candidacy, LogPosition, Role


@dataclass(frozen=True)
class NodeSnapshot:
    """Everything the tracer reports about a node, read through its public properties.

    Attributes:
        node_id: The node's ID.
        role: Its role.
        term: Its current term.
        vote: Its vote in that term, or None.
        last_log: Its last log entry.
        peers: The IDs of every other member of the cluster.
        candidacy: The node's Candidacy object itself, while Candidate —
            kept as the live object, so that after a call the tracer can
            still read the tally it ended with, even if the node has
            discarded it on winning.
        votes_granted: The Candidacy's granted votes when the snapshot
            was taken.
        votes_refused: The Candidacy's refusals when the snapshot was
            taken.
        next_index: Each Follower's next index, while Leader.
        match_index: Each Follower's match index, while Leader.
    """

    node_id: int
    role: Role
    term: int
    vote: Optional[int]
    last_log: LogPosition
    peers: frozenset[int]
    candidacy: Optional[Candidacy] = None
    votes_granted: frozenset[int] = frozenset()
    votes_refused: frozenset[int] = frozenset()
    next_index: dict[int, int] = field(default_factory=dict)
    match_index: dict[int, int] = field(default_factory=dict)

    @classmethod
    def of(cls, node: Any) -> "NodeSnapshot":
        """Read a `DurableNodeState`'s current state through its public properties."""
        candidacy, leadership = node.candidacy, node.leadership
        followers = sorted(leadership.followers) if leadership is not None else []
        return cls(
            node_id=node.node_id,
            role=node.role,
            term=node.current_term,
            vote=node.voted_for,
            last_log=node.log.last_position,
            peers=node.peers,
            candidacy=candidacy,
            votes_granted=candidacy.votes_granted if candidacy else frozenset(),
            votes_refused=candidacy.votes_refused if candidacy else frozenset(),
            next_index={f: leadership.next_index(f) for f in followers},
            match_index={f: leadership.match_index(f) for f in followers},
        )

    def as_role(self, role: Role) -> "NodeSnapshot":
        """The same snapshot with a different role."""
        return replace(self, role=role)
