"""An in-memory Transport: RPCs are calls on the peer's RaftNode, over links a test can cut.

A delivery runs as the network's own task, so cancelling or stopping the sender never cuts a
receiver's handling short, just as a real receiver keeps going after its sender gives up. A link
is checked both ways, so a cut can lose the request or only its answer.
"""

import asyncio

from raftkv.runtime import NodeStoppedError, PeerUnreachableError
from tests.traces.recorder import trace_step


class InMemoryNetwork:
    """Routes RPCs between attached RaftNodes, except across cut links or from or to detached ones.

    Attributes:
        errors: Every exception a receiver raised while handling an RPC, other than being
            stopped; its sender saw only `PeerUnreachableError`.
        busy: Whether any delivery is still being handled.
    """

    def __init__(self):
        self.errors = []
        self._nodes = {}
        self._cut = set()
        self._deliveries = set()

    @property
    def busy(self):
        return bool(self._deliveries)

    def attach(self, raft_node):
        """Connect a running node, replacing any earlier one with its ID."""
        self._nodes[raft_node.node_id] = raft_node

    def detach(self, node_id):
        """Disconnect a node: RPCs from it and to it are lost from now on."""
        self._nodes.pop(node_id, None)

    def transport_for(self, node_id):
        """Return the Transport `node_id` sends through."""
        return _Link(self, node_id)

    def isolate(self, node_id, members):
        """Cut every link between `node_id` and the other `members`, both ways."""
        trace_step("net", "node %d is cut off from every other node", node_id)
        for other in members:
            if other != node_id:
                self._cut |= {(node_id, other), (other, node_id)}

    def cut(self, a, b):
        """Cut the link between `a` and `b`, both ways."""
        trace_step("net", "the link between node %d and node %d is cut", a, b)
        self._cut |= {(a, b), (b, a)}

    def heal(self):
        """Restore every cut link."""
        trace_step("net", "every link is restored")
        self._cut.clear()

    def reachable(self, sender, receiver):
        """Whether a message from `sender` reaches `receiver` now: both attached, link not cut."""
        return (
            sender in self._nodes
            and receiver in self._nodes
            and (sender, receiver) not in self._cut
        )

    async def idle(self):
        """Wait until every delivery, including any started meanwhile, has been handled."""
        while self._deliveries:
            await asyncio.wait(set(self._deliveries))

    async def send(self, sender, receiver, handler_name, request):
        """Deliver `request` to `receiver` and return its answer, if both ways are open.

        Raises:
            PeerUnreachableError: If the request or the answer is lost, the receiver is stopped,
                or it raised while handling the request.
        """
        if not self.reachable(sender, receiver):
            raise PeerUnreachableError(f"{sender} -> {receiver}")
        handler = getattr(self._nodes[receiver], handler_name)
        delivery = asyncio.get_running_loop().create_task(handler(request))
        self._deliveries.add(delivery)
        delivery.add_done_callback(self._delivered)
        try:
            response = await asyncio.shield(delivery)
        except Exception as error:
            raise PeerUnreachableError(f"{receiver} gave no answer: {error!r}") from error
        if not self.reachable(receiver, sender):
            raise PeerUnreachableError(f"{receiver} -> {sender}")
        return response

    def _delivered(self, delivery):
        """Forget a finished delivery, keeping any error its receiver raised other than stopping."""
        self._deliveries.discard(delivery)
        # NOTE: read here, not in `send`, so an error is kept even when the sender was cancelled
        # and no longer waits for the answer.
        if delivery.cancelled() or delivery.exception() is None:
            return
        if not isinstance(delivery.exception(), NodeStoppedError):
            self.errors.append(delivery.exception())


class _Link:
    """One node's Transport over an InMemoryNetwork."""

    def __init__(self, network, node_id):
        self._network = network
        self._node_id = node_id

    async def request_vote(self, peer, request):
        return await self._network.send(self._node_id, peer, "handle_request_vote", request)

    async def append_entries(self, peer, request):
        return await self._network.send(self._node_id, peer, "handle_append_entries", request)
