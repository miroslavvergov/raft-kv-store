"""A node's state at one moment, as the tracer compares it before and after a call."""

from dataclasses import dataclass, field, replace
from typing import Any

from raftkv.consensus import Candidacy, LogPosition, Role


@dataclass(frozen=True)
class NodeSnapshot:
    """Everything the tracer reports about a node, read through its public properties.

    Attributes:
        node_id: The node's ID.
        role: Its role.
        current_term: Its current term.
        voted_for: Its vote in that term, or None.
        last_log_position: Its last log position.
        peers: The IDs of every other cluster member.
        live_candidacy: The Candidacy object itself while Candidate, not a copy,
            so the tracer can read its final tally after the call even if the
            node discarded it on winning.
        votes_granted: The Candidacy's grants when the snapshot was taken.
        votes_refused: The Candidacy's refusals when the snapshot was taken.
        next_index: Each Follower's next index, while Leader.
        match_index: Each Follower's match index, while Leader.
    """

    node_id: int
    role: Role
    current_term: int
    voted_for: int | None
    last_log_position: LogPosition
    peers: frozenset[int]
    live_candidacy: Candidacy | None = None
    votes_granted: frozenset[int] = frozenset()
    votes_refused: frozenset[int] = frozenset()
    next_index: dict[int, int] = field(default_factory=dict)
    match_index: dict[int, int] = field(default_factory=dict)

    @classmethod
    def of(cls, node: Any) -> "NodeSnapshot":
        """Return a snapshot of `node`'s current state, read through its public properties.

        Args:
            node: The `DurableNodeState` to read.
        """
        candidacy, leadership = node.candidacy, node.leadership
        followers = sorted(leadership.followers) if leadership is not None else []
        return cls(
            node_id=node.node_id,
            role=node.role,
            current_term=node.current_term,
            voted_for=node.voted_for,
            last_log_position=node.log.last_position,
            peers=node.peers,
            live_candidacy=candidacy,
            votes_granted=candidacy.votes_granted if candidacy else frozenset(),
            votes_refused=candidacy.votes_refused if candidacy else frozenset(),
            next_index={f: leadership.next_index(f) for f in followers},
            match_index={f: leadership.match_index(f) for f in followers},
        )

    def with_role(self, role: Role) -> "NodeSnapshot":
        """Return a copy of this snapshot with `role` replaced."""
        return replace(self, role=role)
