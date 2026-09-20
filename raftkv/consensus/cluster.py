"""The cluster's voting members, and what counts as a majority of them."""

from collections.abc import Iterable


class Cluster:
    """The fixed set of voting members and the strict majority decisions need.

    A majority is `len(members) // 2 + 1`, more than half (ELECT-12): 3 of 4,
    since two halves need not share a node. Any two majorities share a node:
    it votes once per term (ELECT-8), so no term has two winners, and it holds
    every committed entry, so it refuses a Candidate missing one (ELECT-9).
    Used for elections (ELECT-11) and commitment (APPLY-1).

    Attributes:
        members: The IDs of every voting member, this node included (NODE-8,
            DD-20).
        majority: How many members make a strict majority.
    """

    def __init__(self, members: Iterable[int]) -> None:
        """Define a cluster by its members.

        Args:
            members: Every voting member's ID; unique positive integers.

        Raises:
            ValueError: If `members` is empty or has a duplicate or an ID that is
                not a positive integer (a bool included).
        """
        member_list = list(members)
        if not member_list:
            raise ValueError("a cluster needs at least one member")
        for member in member_list:
            # NOTE: bool is an int subclass, so True would pass as node 1 without this check.
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

        These receive a Candidate's RequestVote (ELECT-6) and a Leader's
        replication (REPL-2).

        Args:
            node_id: A member of this cluster.

        Raises:
            ValueError: If `node_id` is not a member.
        """
        if node_id not in self._members:
            raise ValueError(f"node {node_id} is not a member of {sorted(self._members)}")
        return self._members - {node_id}

    def is_majority(self, node_ids: Iterable[int]) -> bool:
        """Check whether `node_ids` include a strict majority of the members.

        Each member counts once however often it appears, and non-members count
        for nothing (ELECT-12).

        Args:
            node_ids: The nodes that agreed, such as the voters that granted a vote.

        Returns:
            True if at least `majority` distinct members are among them.
        """
        return len(self._members.intersection(node_ids)) >= self.majority
