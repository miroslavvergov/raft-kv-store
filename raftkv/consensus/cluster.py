"""The cluster's voting members, and what counts as a majority of them."""

from typing import Iterable


class Cluster:
    """The fixed set of nodes that vote, and the majority every decision needs.

    Every decision Raft reaches by agreement — electing a Leader
    (ELECT-11), committing an entry (APPLY-1) — needs a strict majority
    of the configured voting members (ELECT-12): more than half of them,
    `len(members) // 2 + 1`. That is 2 of 3, 3 of 5, and also 3 of 4:
    half is never enough, because two halves of a 4-node cluster need not
    share a single node.

    Any two majorities of the same cluster always share at least one node,
    since together they hold more nodes than the cluster has. That shared
    node is what makes Raft safe. It can vote only once per term
    (ELECT-8), so two Candidates can never both win the same term; and it
    holds every committed entry, so any Candidate missing one is refused
    by it (ELECT-9).

    `is_majority` counts each member at most once and ignores any ID
    that is not a member, so repeating a vote, or a reply from a node
    outside the cluster, can never make up a majority.

    Members are identified by positive integers (NODE-8, DD-20).

    Attributes:
        members: The IDs of every voting member, this node included.
        majority: How many members make a strict majority.
    """

    def __init__(self, members: Iterable[int]) -> None:
        """Define a cluster by its members.

        Args:
            members: The ID of every voting member. Each must be a
                positive integer and appear only once.

        Raises:
            ValueError: If `members` is empty, contains an ID that is
                not a positive integer, or contains an ID more than once.
        """
        member_list = list(members)
        if not member_list:
            raise ValueError("a cluster needs at least one member")
        for member in member_list:
            if isinstance(member, bool) or not isinstance(member, int) or member <= 0:
                raise ValueError(f"node IDs are positive integers, got {member!r}")
        if len(set(member_list)) != len(member_list):
            raise ValueError(f"node IDs must be unique, got {member_list}")
        self._members = frozenset(member_list)

    @property
    def members(self) -> frozenset[int]:
        return self._members

    @property
    def majority(self) -> int:
        return len(self._members) // 2 + 1

    def peers_of(self, node_id: int) -> frozenset[int]:
        """Return every member except `node_id`.

        These are the nodes a Candidate sends RequestVote to (ELECT-6)
        and a Leader replicates to (REPL-2).

        Args:
            node_id: A member of this cluster.

        Returns:
            The IDs of all other members.

        Raises:
            ValueError: If `node_id` is not a member.
        """
        if node_id not in self._members:
            raise ValueError(f"node {node_id} is not a member of {sorted(self._members)}")
        return self._members - {node_id}

    def is_majority(self, node_ids: Iterable[int]) -> bool:
        """Check whether `node_ids` include a strict majority of the members.

        Each member counts once however many times it appears, and IDs
        that are not members count for nothing (ELECT-12).

        Args:
            node_ids: The nodes that agreed — for example, the voters that
                granted a Candidate their vote.

        Returns:
            True if at least `majority` distinct members are among them.
        """
        return len(self._members.intersection(node_ids)) >= self.majority
