"""What a node needs from the network: one call per RPC, answered or refused."""

from typing import Protocol

from raftkv.consensus import (
    AppendEntriesRequest,
    AppendEntriesResponse,
    RequestVoteRequest,
    RequestVoteResponse,
)


class Transport(Protocol):
    """Sends one node's RPCs to its peers and returns their answers.

    Every call either returns the peer's answer or raises `PeerUnreachableError`,
    and does so within a bounded time: a call that could wait forever would hold
    that peer's one RPC in flight forever. A transport never retries; resending is
    the node's decision.
    """

    async def request_vote(self, peer: int, request: RequestVoteRequest) -> RequestVoteResponse:
        """Send a RequestVote to `peer` and return its answer.

        Raises:
            PeerUnreachableError: If no answer arrives.
        """
        ...

    async def append_entries(
        self, peer: int, request: AppendEntriesRequest
    ) -> AppendEntriesResponse:
        """Send an AppendEntries to `peer` and return its answer.

        Raises:
            PeerUnreachableError: If no answer arrives.
        """
        ...
