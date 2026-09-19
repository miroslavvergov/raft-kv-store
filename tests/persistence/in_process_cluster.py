"""InProcessCluster: real nodes, each with its own SQLite file, whose RPCs are direct calls.

The test decides whose election timeout fires and which messages arrive, in what order, and which
are lost or repeated. Each voter's vote is read back from its file before its answer is handed on
(PERSIST-1, PERSIST-2), and every step is reported to the election trace (tests/conftest.py).
"""

import logging
from collections import defaultdict
from dataclasses import dataclass

from raftkv.consensus import Cluster, RequestVoteRequest, RequestVoteResponse, Role
from raftkv.persistence import DurableNodeState, SqliteStore
from tests.divergent_logs import make_log
from tests.election_traces.recorder import HARNESS
from tests.persistence.store_doubles import reload

_harness = logging.getLogger(HARNESS)


def trace_step(source, message, *args, event=None):
    """Report one harness step to the election trace, if one is being recorded.

    Args:
        source: "net", "clock", "crash", "disk", or "state".
        message: A %-style format string, filled from `args`.
        event: The step as a structured event, for the `.jsonl` trace.
    """
    if _harness.isEnabledFor(logging.INFO):
        _harness.info(message, *args, extra={"trace_source": source, "trace_event": event})


@dataclass(frozen=True)
class InFlight:
    """A RequestVote on its way to `voter`, or, once `response` is set, the answer coming back."""

    request: RequestVoteRequest
    voter: int
    response: RequestVoteResponse | None = None

    def describe(self):
        """Return the message as one line of the `.log` trace."""
        if self.response is None:
            return (
                f"RequestVote(term {self.request.term}) "
                f"from {self.request.candidate_id} to {self.voter}"
            )
        answer = "granted" if self.response.vote_granted else "refused"
        return (
            f"RequestVoteResponse(term {self.response.term}, {answer}) "
            f"from {self.voter} to {self.request.candidate_id}"
        )

    def trace_fields(self):
        """Return the message as trace-event fields, in etcd's `msg` shape."""
        if self.response is None:
            return {
                "type": "RequestVote",
                "term": self.request.term,
                "from": self.request.candidate_id,
                "to": self.voter,
            }
        return {
            "type": "RequestVoteResponse",
            "term": self.response.term,
            "from": self.voter,
            "to": self.request.candidate_id,
            "reject": not self.response.vote_granted,
        }


class InProcessCluster:
    """Real nodes whose RPCs are direct calls; nothing happens unless the test does it.

    Messages are delivered at once (`ask_for_vote`, `deliver_vote_response`, `run_election`) or
    put in flight (`send_requests`) to be delivered, duplicated, or dropped later, in any order.
    Every Leader and every granted vote is recorded per term, for `assert_election_safety`.

    Attributes:
        paths: Each node's SQLite file.
        nodes: Each running node's DurableNodeState.
        in_flight: The messages in flight, as InFlight.
        leaders_by_term: Every node ever seen as Leader, per term.
        votes_by_voter_and_term: Every Candidate each node granted its vote, per term,
            its own vote as a Candidate included.
    """

    def __init__(self, directory, member_ids, check_votes_on_disk=True):
        self.cluster = Cluster(member_ids)
        self.paths = {n: str(directory / f"node-{n}.db") for n in member_ids}
        self.nodes = {}
        self.in_flight = []
        self.leaders_by_term = defaultdict(set)
        self.votes_by_voter_and_term = defaultdict(set)
        self._stores = {}
        self._check_votes_on_disk = check_votes_on_disk

    @property
    def member_ids(self):
        """Return every member's ID, in ascending order."""
        return sorted(self.cluster.members)

    async def preload(self, node_id, log_terms=(), current_term=0, voted_for=None):
        """Write a starting log, term, and vote straight to a stopped node's file."""
        async with SqliteStore(self.paths[node_id]) as store:
            await store.replace_log_from(1, list(make_log(list(log_terms))))
            await store.save_term_and_vote(current_term, voted_for)

    async def start(self, node_id):
        """Start a node from its file."""
        store = SqliteStore(self.paths[node_id])
        await store.__aenter__()
        self._stores[node_id] = store
        self.nodes[node_id] = await DurableNodeState.load(node_id, store, self.cluster)

    async def stop(self, node_id):
        """Stop a node, losing everything it holds in memory."""
        del self.nodes[node_id]
        await self._stores.pop(node_id).__aexit__(None, None, None)

    async def restart(self, node_id):
        """Crash a node and bring it back from nothing but its file."""
        trace_step(
            "crash",
            "node %d crashes: its memory is lost; it restarts from its file",
            node_id,
            event={"name": "Restart", "nid": node_id},
        )
        await self.stop(node_id)
        await self.start(node_id)
        self._trace_cluster_state()

    async def start_all(self):
        """Start every member."""
        for node_id in self.member_ids:
            await self.start(node_id)

    async def stop_all(self):
        """Stop every running node."""
        for node_id in list(self.nodes):
            await self.stop(node_id)

    async def fire_election_timeout(self, candidate):
        """Fire `candidate`'s election timeout: it starts an election; return its RequestVote."""
        trace_step(
            "clock",
            "election timeout fires on node %d",
            candidate,
            event={"name": "Timeout", "nid": candidate},
        )
        request = await self.nodes[candidate].start_election()
        self.votes_by_voter_and_term[(candidate, request.term)].add(candidate)
        self._note_leaders()
        self._trace_cluster_state()
        return request

    async def ask_for_vote(self, voter, request):
        """Deliver a RequestVote to `voter` and return its answer, read back from disk first.

        With `check_votes_on_disk`, the test fails unless the voter's file holds the term and
        vote it answered with.
        """
        message = InFlight(request, voter)
        trace_step(
            "net",
            "deliver %s",
            message.describe(),
            event={"name": "Deliver", "msg": message.trace_fields()},
        )
        response = await self.nodes[voter].handle_vote_request(request)
        if response.vote_granted:
            self.votes_by_voter_and_term[(voter, request.term)].add(request.candidate_id)
        if self._check_votes_on_disk or _harness.isEnabledFor(logging.INFO):
            await self._check_vote_on_disk(voter)
        self._note_leaders()
        self._trace_cluster_state()
        return response

    async def deliver_vote_response(self, voter, request, response):
        """Deliver `voter`'s answer to `request` to the Candidate; return whether it won."""
        message = InFlight(request, voter, response)
        trace_step(
            "net",
            "deliver %s",
            message.describe(),
            event={"name": "Deliver", "msg": message.trace_fields()},
        )
        became_leader = await self.nodes[request.candidate_id].handle_vote_response(
            voter, request.term, response
        )
        self._note_leaders()
        self._trace_cluster_state()
        return became_leader

    def send_requests(self, request):
        """Put a Candidate's RequestVote in flight to every one of its peers."""
        for voter in sorted(self.nodes[request.candidate_id].peers):
            self.in_flight.append(InFlight(request, voter))

    async def deliver(self, message, keep_copy=False):
        """Deliver an in-flight message; a delivered request's answer goes in flight.

        Args:
            message: One of `in_flight`.
            keep_copy: Leave a copy in flight, to arrive again later, as a network that
                duplicates messages would.

        Returns:
            The voter's answer if `message` was a request; None if it was an answer.
        """
        if keep_copy:
            trace_step(
                "net",
                "duplicate %s: a copy stays in flight",
                message.describe(),
                event={"name": "Duplicate", "msg": message.trace_fields()},
            )
        else:
            self.in_flight.remove(message)
        if message.response is None:
            response = await self.ask_for_vote(message.voter, message.request)
            self.in_flight.append(InFlight(message.request, message.voter, response))
            return response
        await self.deliver_vote_response(message.voter, message.request, message.response)
        return None

    def drop(self, message):
        """Lose an in-flight message."""
        self.in_flight.remove(message)
        trace_step(
            "net",
            "drop %s",
            message.describe(),
            event={"name": "Drop", "msg": message.trace_fields()},
        )

    async def run_election(self, candidate, reachable=None):
        """Time out `candidate`, then deliver its requests and their answers at once.

        Requests go to every running peer, or only to those in `reachable`.

        Returns:
            The election's RequestVote.
        """
        request = await self.fire_election_timeout(candidate)
        for voter in sorted(self.nodes[candidate].peers):
            if voter not in self.nodes or (reachable is not None and voter not in reachable):
                continue
            response = await self.ask_for_vote(voter, request)
            await self.deliver_vote_response(voter, request, response)
        return request

    def leaders(self):
        """Return the running nodes that are Leader now, of any term."""
        return {n for n, node in self.nodes.items() if node.role is Role.LEADER}

    def assert_election_safety(self):
        """Assert that no term had two Leaders and no node voted for two Candidates in one term.

        Also asserts that every running node holds exactly its role's record for its term: a
        Candidacy while Candidate, a Leadership while Leader, and neither while Follower.
        """
        for term, leaders in self.leaders_by_term.items():
            assert len(leaders) == 1, f"term {term} had leaders {sorted(leaders)}"
        for (voter, term), candidates in self.votes_by_voter_and_term.items():
            assert len(candidates) == 1, (
                f"node {voter} voted for {sorted(candidates)} in term {term}"
            )
        for node in self.nodes.values():
            _assert_role_record_matches(node)

    async def _check_vote_on_disk(self, voter):
        node = self.nodes[voter]
        persisted = await reload(self.paths[voter])
        on_disk = (persisted.current_term, persisted.voted_for)
        in_memory = (node.current_term, node.voted_for)
        trace_step(
            "disk",
            "node %d's file holds term %d, vote %d — %s",
            voter,
            on_disk[0],
            on_disk[1] or 0,
            "same as memory"
            if on_disk == in_memory
            else f"MEMORY HAS term {in_memory[0]}, vote {in_memory[1] or 0}",
            event={
                "name": "DiskCheck",
                "nid": voter,
                "ok": on_disk == in_memory,
                "disk": {"term": on_disk[0], "vote": on_disk[1]},
                "memory": {"term": in_memory[0], "vote": in_memory[1]},
            },
        )
        if self._check_votes_on_disk:
            assert on_disk == in_memory

    def _note_leaders(self):
        for node_id, node in self.nodes.items():
            if node.role is Role.LEADER:
                self.leaders_by_term[node.current_term].add(node_id)

    def _trace_cluster_state(self):
        """Report every running node's role, term, vote, and collected votes."""
        if not _harness.isEnabledFor(logging.INFO):
            return
        parts, nodes = [], {}
        for node_id in self.member_ids:
            node = self.nodes.get(node_id)
            if node is None:
                parts.append(f"{node_id} down")
                continue
            part = f"{node_id} {node.role.value} t{node.current_term} v{node.voted_for or 0}"
            if node.candidacy is not None:
                part += f" votes{sorted(node.candidacy.votes_granted)}"
            parts.append(part)
            nodes[node_id] = {
                "role": node.role.value,
                "term": node.current_term,
                "vote": node.voted_for,
            }
        trace_step("state", "%s", " | ".join(parts), event={"name": "ClusterState", "nodes": nodes})


def _assert_role_record_matches(node):
    if node.role is Role.CANDIDATE:
        assert node.candidacy.term == node.current_term and node.leadership is None
    elif node.role is Role.LEADER:
        assert node.leadership.term == node.current_term and node.candidacy is None
    else:
        assert node.candidacy is None and node.leadership is None
