"""RuntimeCluster: real nodes run by RaftNode, over an in-memory network, on a shared tick.

The test advances every node's clock one tick at a time, and every message a tick causes is
delivered before the next tick. The seed fixes each node's election timeouts and the test's own
choices; the order in which concurrent deliveries finish can still vary with aiosqlite's threads,
so two runs of one seed may differ in detail. After every tick the cluster is checked: no node
failed, no receiver raised, no term had two Leaders at the moment of the check, and no committed
entry was changed or lost. A Leader that rose and fell within one tick escapes the per-tick check;
the trace check at teardown sees every one.
"""

import random
from collections import defaultdict

from raftkv.consensus import Role
from raftkv.kvstore import DEFAULT_SESSION_TIMEOUT, OpenSession
from raftkv.runtime import RaftNode, Timing
from tests.cluster.in_process_cluster import InProcessCluster
from tests.support.in_memory_network import InMemoryNetwork
from tests.support.waiting import within_bound
from tests.traces.recorder import trace_step

# NOTE: the shortest election timeout that still leaves several heartbeats before it.
TEST_TIMING = Timing(heartbeat_ticks=1, election_ticks=10)


class RuntimeCluster:
    """Nodes that run themselves; the test only ticks the clock, cuts links, and crashes nodes.

    Attributes:
        files: The InProcessCluster holding each node's file, store, and state machine, and the
            record of every committed entry.
        network: The network every node sends through.
        nodes: Each running node's RaftNode.
        ticks: How many ticks have passed.
        member_ids: Every member's ID, in ascending order.
    """

    def __init__(
        self,
        directory,
        member_ids,
        *,
        timing=TEST_TIMING,
        seed=0,
        session_timeout=DEFAULT_SESSION_TIMEOUT,
    ):
        self.files = InProcessCluster(
            directory, member_ids, check_votes_on_disk=False, session_timeout=session_timeout
        )
        self.network = InMemoryNetwork()
        self.nodes = {}
        self.ticks = 0
        self._stopped_failures = []
        self._timing = timing
        self._seed = seed
        self._leaders_by_term = defaultdict(set)

    @property
    def member_ids(self):
        return self.files.member_ids

    async def start(self, node_id):
        """Start a node from its file, with an empty state machine."""
        await self.files.start(node_id)
        node = RaftNode(
            self.files.nodes[node_id],
            self.network.transport_for(node_id),
            self._timing,
            random.Random(f"{self._seed}-{node_id}"),
        )
        self.nodes[node_id] = node
        self.network.attach(node)

    async def start_all(self):
        for node_id in self.member_ids:
            await self.start(node_id)

    async def stop(self, node_id):
        """Stop a node, losing everything it holds in memory; the next `check` reports a failure."""
        node = self.nodes.pop(node_id)
        self.network.detach(node_id)
        # NOTE: stop waits for the RPCs the node is still handling, so its file closes only
        # once nothing writes to it.
        await within_bound(node.stop())
        await self.files.stop(node_id)
        if node.failure is not None:
            self._stopped_failures.append((node_id, node.failure))

    async def stop_all(self):
        for node_id in list(self.nodes):
            await self.stop(node_id)

    async def restart(self, node_id):
        """Crash a node and bring it back from nothing but its file."""
        trace_step("crash", "node %d crashes and restarts from its file", node_id)
        await self.stop(node_id)
        await self.start(node_id)

    async def tick(self, count=1):
        """Advance every running node's clock `count` times, settling and checking after each."""
        for _ in range(count):
            self.ticks += 1
            for node_id in sorted(self.nodes):
                await self.nodes[node_id].tick()
            await self.settle()
            self.check()

    async def settle(self):
        """Wait until no node has a task running and no message is being delivered.

        Raises:
            AssertionError: If that takes longer than a passing test ever needs.
        """
        await within_bound(self._settle())

    async def run_until(self, condition, *, within):
        """Tick until `condition()` holds; fail if it does not within `within` ticks.

        Returns:
            How many ticks it took.
        """
        for used in range(within + 1):
            if condition():
                return used
            await self.tick()
        raise AssertionError(f"not reached within {within} ticks: {self.describe()}")

    async def append_command(self, node_id, command):
        """Have `node_id` append a client command; return its position once messages settle."""
        position = await self.nodes[node_id].append_command(command)
        await self.settle()
        self.check()
        return position

    async def open_session(self, leader):
        """Open a client session through `leader`; return its client ID once `leader` commits it."""
        position = await self.append_command(leader, OpenSession().encode())
        durable = self.nodes[leader].durable
        assert durable.commit_index >= position.index, "the session did not commit"
        assert durable.log.term_at(position.index) == position.term
        return position.index

    def leader(self):
        """Return the running Leader of the highest term, or None if no node leads."""
        leading = [n for n, node in self.nodes.items() if node.durable.role is Role.LEADER]
        return max(leading, key=lambda n: self.nodes[n].durable.current_term, default=None)

    def has_one_leader_known_to_all(self):
        """Whether one node leads and every running node names it as the Leader."""
        leader = self.leader()
        return leader is not None and all(n.leader_id == leader for n in self.nodes.values())

    def all_caught_up(self):
        """Whether every running node has committed and applied the Leader's whole log."""
        leader = self.leader()
        if leader is None:
            return False
        last = self.nodes[leader].durable.log.last_index
        return all(
            node.durable.commit_index == last and node.durable.last_applied == last
            for node in self.nodes.values()
        )

    def maps(self):
        """Return each running node's key-value map, by node ID."""
        return self.files.maps()

    def check(self):
        """Assert that no node failed, no receiver raised, and no safety rule broke by now."""
        for node_id, node in self.nodes.items():
            assert node.failure is None, f"node {node_id} stopped on {node.failure!r}"
        assert self._stopped_failures == [], f"a stopped node had failed: {self._stopped_failures}"
        assert self.network.errors == [], f"a receiver raised: {self.network.errors!r}"
        for node_id, node in self.nodes.items():
            if node.durable.role is Role.LEADER:
                self._leaders_by_term[node.durable.current_term].add(node_id)
        for term, leaders in self._leaders_by_term.items():
            assert len(leaders) == 1, f"term {term} had leaders {sorted(leaders)}"
        self.files.assert_log_safety()

    def describe(self):
        """Return every node's role, term, log length, commit index, and applied index."""
        return " | ".join(
            f"{n} {node.durable.role.value} t{node.durable.current_term} "
            f"log{node.durable.log.last_index} c{node.durable.commit_index} "
            f"a{node.durable.last_applied}"
            for n, node in sorted(self.nodes.items())
        )

    async def _settle(self):
        while self.network.busy or any(node.busy for node in self.nodes.values()):
            await self.network.idle()
            for node in list(self.nodes.values()):
                await node.idle()
