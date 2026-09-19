"""Tier 2 election tests across whole clusters, with no network and no timers.

Every node is a real DurableNodeState with its own SQLite file. Messages
are handed from node to node by direct calls, and the test itself decides
whose election timeout fires, which messages arrive, in what order, and
which are lost, repeated, or held back. Each voter's vote is read back
from its file before its answer is handed on, so every scenario also
checks that no vote is ever answered before it is on disk (PERSIST-1,
PERSIST-2).

Scenarios: a cold start electing exactly one Leader; split votes in three
and four nodes resolved in the next term; a restarted voter refusing a
second Candidate in the same term, with a negative control showing that
two Leaders appear in one term if votes are not persisted; a Candidate
missing a committed entry losing (ELECT-9, ELECT-10); a delayed grant from
an earlier election not creating a second Leader; stepping down on a
higher term; and randomized schedules — crashes, restarts, delays,
duplicates, and losses — never electing two Leaders in one term
(ELECT-8, ELECT-11, ELECT-12), after which a quiet network always elects
a Leader.

With `pytest --trace-elections`, every step the harness takes is reported
alongside the nodes' own log lines and trace events, and written to
test-traces/elections/ (see tests/conftest.py).
"""

import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import pytest

from raftkv.consensus import Cluster, RequestVoteRequest, RequestVoteResponse, Role
from raftkv.persistence import DurableNodeState, SqliteStore
from tests.divergent_logs import make_log
from tests.election_traces.recorder import HARNESS
from tests.persistence.store_doubles import reload

_harness = logging.getLogger(HARNESS)


def report(source, message, *args, event=None):
    """Report one harness step to the election trace, if one is being recorded.

    Args:
        source: What kind of step it is: "net", "clock", "crash", "disk",
            or "state".
        message: A %-style format string describing the step.
        *args: Its arguments.
        event: The step as a structured event, for the `.jsonl` trace.
    """
    if _harness.isEnabledFor(logging.INFO):
        _harness.info(message, *args, extra={"trace_source": source, "trace_event": event})


@dataclass(frozen=True)
class InFlight:
    """A message somewhere in the network: a RequestVote on its way, or its answer coming back.

    Attributes:
        request: The Candidate's RequestVote.
        voter: The node the request is addressed to.
        response: None while the request is on its way to `voter`; the
            voter's answer once it is on its way back to the Candidate.
    """

    request: RequestVoteRequest
    voter: int
    response: Optional[RequestVoteResponse] = None

    def describe(self):
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


class InProcessCluster:
    """Real nodes, each with its own SQLite file, whose RPCs are direct calls.

    Nothing here happens on its own: the test decides whose election
    timeout fires and which messages are delivered. Messages can be
    delivered at once (`ask_for_vote`, `deliver_answer`, `run_election`)
    or left in flight (`send_requests`) to be delivered, duplicated, or
    dropped later, in any order. Every term in which a node has ever been
    seen as Leader is recorded, so a test can check that no term ever had
    two. Every step is reported to the election trace.
    """

    def __init__(self, directory, member_ids, check_votes_on_disk=True):
        self.cluster = Cluster(member_ids)
        self.paths = {n: str(directory / f"node-{n}.db") for n in member_ids}
        self.nodes = {}
        self.in_flight = []
        self.leaders_by_term = defaultdict(set)
        self._stores = {}
        self._check_votes_on_disk = check_votes_on_disk

    async def seed(self, node_id, log_terms=(), current_term=0, voted_for=None):
        """Write a starting log, term, and vote straight to a node's file."""
        async with SqliteStore(self.paths[node_id]) as store:
            await store.save_log_from(1, list(make_log(list(log_terms))))
            await store.save_term_and_vote(current_term, voted_for)

    async def start(self, node_id):
        store = SqliteStore(self.paths[node_id])
        await store.__aenter__()
        self._stores[node_id] = store
        self.nodes[node_id] = await DurableNodeState.load(node_id, store, self.cluster)

    async def stop(self, node_id):
        del self.nodes[node_id]
        await self._stores.pop(node_id).__aexit__(None, None, None)

    async def restart(self, node_id):
        """Crash a node and bring it back from nothing but its file."""
        report("crash", "node %d crashes: its memory is lost; it restarts from its file", node_id,
               event={"name": "Restart", "nid": node_id})
        await self.stop(node_id)
        await self.start(node_id)
        self.report_state()

    async def start_all(self):
        for node_id in sorted(self.cluster.members):
            await self.start(node_id)

    async def stop_all(self):
        for node_id in list(self.nodes):
            await self.stop(node_id)

    async def fire_election_timeout(self, candidate):
        """The candidate's election timeout fires: it starts an election."""
        report("clock", "election timeout fires on node %d", candidate,
               event={"name": "Timeout", "nid": candidate})
        request = await self.nodes[candidate].start_election()
        self.note_leaders()
        self.report_state()
        return request

    async def ask_for_vote(self, voter, request):
        """Deliver a RequestVote and return the answer, once its vote is on disk.

        The voter's file is read back before the answer is returned. When
        `check_votes_on_disk` is set, the test fails if the file does not
        hold the term and vote the voter answered with; when a trace is
        being recorded, the comparison is reported either way.
        """
        message = InFlight(request, voter)
        report("net", "deliver %s", message.describe(),
               event={"name": "Deliver", "msg": _message_fields(message)})
        response = await self.nodes[voter].handle_vote_request(request)
        if self._check_votes_on_disk or _harness.isEnabledFor(logging.INFO):
            node = self.nodes[voter]
            persisted = await reload(self.paths[voter])
            on_disk = (persisted.current_term, persisted.voted_for)
            in_memory = (node.current_term, node.voted_for)
            report(
                "disk",
                "node %d's file holds term %d, vote %d — %s",
                voter, on_disk[0], on_disk[1] or 0,
                "same as memory" if on_disk == in_memory
                else f"MEMORY HAS term {in_memory[0]}, vote {in_memory[1] or 0}",
                event={"name": "DiskCheck", "nid": voter, "ok": on_disk == in_memory,
                       "disk": {"term": on_disk[0], "vote": on_disk[1]},
                       "memory": {"term": in_memory[0], "vote": in_memory[1]}},
            )
            if self._check_votes_on_disk:
                assert on_disk == in_memory
        self.note_leaders()
        self.report_state()
        return response

    async def deliver_answer(self, candidate, voter, request, response):
        message = InFlight(request, voter, response)
        report("net", "deliver %s", message.describe(),
               event={"name": "Deliver", "msg": _message_fields(message)})
        became_leader = await self.nodes[candidate].handle_vote_response(
            voter, request.term, response
        )
        self.note_leaders()
        self.report_state()
        return became_leader

    def send_requests(self, request):
        """Put a Candidate's RequestVote in flight to every one of its peers."""
        for voter in sorted(self.nodes[request.candidate_id].peers):
            self.in_flight.append(InFlight(request, voter))

    async def deliver(self, message, keep_copy=False):
        """Deliver an in-flight message; a delivered request's answer goes in flight.

        Args:
            message: One of `in_flight`.
            keep_copy: If True, a copy stays in flight and can arrive
                again later, as a network that duplicates messages would
                allow.

        Returns:
            The voter's answer, if `message` was a request; None if it
            was an answer.
        """
        if keep_copy:
            report("net", "duplicate %s: a copy stays in flight", message.describe(),
                   event={"name": "Duplicate", "msg": _message_fields(message)})
        else:
            self.in_flight.remove(message)
        if message.response is None:
            response = await self.ask_for_vote(message.voter, message.request)
            self.in_flight.append(InFlight(message.request, message.voter, response))
            return response
        await self.deliver_answer(
            message.request.candidate_id, message.voter, message.request, message.response
        )
        return None

    def drop(self, message):
        """Lose an in-flight message."""
        self.in_flight.remove(message)
        report("net", "drop %s", message.describe(),
               event={"name": "Drop", "msg": _message_fields(message)})

    def report_state(self):
        """Report every running node's role, term, vote, and collected votes."""
        if not _harness.isEnabledFor(logging.INFO):
            return
        parts, nodes = [], {}
        for node_id in sorted(self.cluster.members):
            node = self.nodes.get(node_id)
            if node is None:
                parts.append(f"{node_id} down")
                continue
            part = f"{node_id} {node.role.value} t{node.current_term} v{node.voted_for or 0}"
            if node.candidacy is not None:
                part += f" votes{sorted(node.candidacy.votes_granted)}"
            parts.append(part)
            nodes[node_id] = {"role": node.role.value, "term": node.current_term,
                              "vote": node.voted_for}
        report("state", "%s", " | ".join(parts), event={"name": "ClusterState", "nodes": nodes})

    async def run_election(self, candidate, reachable=None):
        """Time out `candidate`, then deliver its requests and their answers at once.

        Requests go to every running peer, or only those in `reachable`.
        """
        request = await self.fire_election_timeout(candidate)
        for voter in sorted(self.nodes[candidate].peers):
            if voter not in self.nodes or (reachable is not None and voter not in reachable):
                continue
            response = await self.ask_for_vote(voter, request)
            await self.deliver_answer(candidate, voter, request, response)
        return request

    def leaders(self):
        return {n for n, node in self.nodes.items() if node.role is Role.LEADER}

    def note_leaders(self):
        for node_id, node in self.nodes.items():
            if node.role is Role.LEADER:
                self.leaders_by_term[node.current_term].add(node_id)

    def assert_at_most_one_leader_per_term(self):
        for term, leaders in self.leaders_by_term.items():
            assert len(leaders) == 1, f"term {term} had leaders {sorted(leaders)}"


def _message_fields(message):
    """An in-flight message as trace-event fields, in etcd's `msg` shape."""
    if message.response is None:
        return {"type": "RequestVote", "term": message.request.term,
                "from": message.request.candidate_id, "to": message.voter}
    return {"type": "RequestVoteResponse", "term": message.response.term,
            "from": message.voter, "to": message.request.candidate_id,
            "reject": not message.response.vote_granted}


@pytest.fixture
async def three_nodes(tmp_path):
    cluster = InProcessCluster(tmp_path, [1, 2, 3])
    await cluster.start_all()
    yield cluster
    await cluster.stop_all()


# --- A cold start -----------------------------------------------------------------


async def test_cold_start_elects_exactly_one_leader(three_nodes):
    # Every node starts as a Follower in term 0 (STATE-2). Node 1's timeout
    # fires first.
    assert three_nodes.leaders() == set()
    await three_nodes.run_election(1)

    assert three_nodes.leaders() == {1}
    leader = three_nodes.nodes[1]
    assert (leader.current_term, leader.leadership.followers) == (1, frozenset({2, 3}))
    for voter in (2, 3):
        node = three_nodes.nodes[voter]
        assert (node.role, node.current_term, node.voted_for) == (Role.FOLLOWER, 1, 1)
    three_nodes.assert_at_most_one_leader_per_term()


async def test_every_vote_of_the_election_survives_a_restart_of_the_whole_cluster(three_nodes):
    await three_nodes.run_election(1)
    for node_id in (1, 2, 3):
        await three_nodes.restart(node_id)
    for node_id in (1, 2, 3):
        node = three_nodes.nodes[node_id]
        assert (node.role, node.current_term, node.voted_for) == (Role.FOLLOWER, 1, 1)


# --- Split votes --------------------------------------------------------------------


async def test_a_three_way_split_vote_is_resolved_in_the_next_term(three_nodes):
    # All three time out at once: each votes for itself, so every request
    # arrives at a node that has already voted in term 1.
    requests = {n: await three_nodes.fire_election_timeout(n) for n in (1, 2, 3)}
    for candidate, request in requests.items():
        for voter in three_nodes.nodes[candidate].peers:
            response = await three_nodes.ask_for_vote(voter, request)
            assert response.vote_granted is False
            await three_nodes.deliver_answer(candidate, voter, request, response)
    assert three_nodes.leaders() == set()
    assert {node.role for node in three_nodes.nodes.values()} == {Role.CANDIDATE}

    # Randomized timeouts make it unlikely they collide again. Node 2 is first.
    await three_nodes.run_election(2)
    assert three_nodes.leaders() == {2}
    assert three_nodes.nodes[2].current_term == 2
    three_nodes.assert_at_most_one_leader_per_term()


async def test_a_two_two_split_in_four_nodes_elects_nobody_in_that_term(tmp_path):
    cluster = InProcessCluster(tmp_path, [1, 2, 3, 4])
    await cluster.start_all()
    try:
        first = await cluster.fire_election_timeout(1)
        second = await cluster.fire_election_timeout(2)
        for voter, (candidate, request) in ((3, (1, first)), (4, (2, second))):
            response = await cluster.ask_for_vote(voter, request)
            assert response.vote_granted
            await cluster.deliver_answer(candidate, voter, request, response)
        # Each Candidate has 2 votes of 4 — half, which is not a majority.
        assert cluster.nodes[1].candidacy.votes_granted == frozenset({1, 3})
        assert cluster.nodes[2].candidacy.votes_granted == frozenset({2, 4})
        assert cluster.leaders() == set()

        await cluster.run_election(3)
        assert cluster.leaders() == {3}
        cluster.assert_at_most_one_leader_per_term()
    finally:
        await cluster.stop_all()


# --- A restarted voter (PERSIST-2, PERSIST-5, ELECT-8) ---------------------------


async def test_a_restarted_voter_refuses_a_second_candidate_in_the_same_term(three_nodes):
    first = await three_nodes.fire_election_timeout(1)
    assert (await three_nodes.ask_for_vote(3, first)).vote_granted
    await three_nodes.restart(3)  # crashes right after answering
    assert three_nodes.nodes[3].voted_for == 1

    second = await three_nodes.fire_election_timeout(2)  # also term 1
    assert second.term == first.term == 1
    assert (await three_nodes.ask_for_vote(3, second)).vote_granted is False


@pytest.mark.negative_control
async def test_negative_control_unpersisted_votes_let_a_restart_elect_two_leaders(
    tmp_path, monkeypatch
):
    # Confirms the test above can fail: if votes are never written to disk,
    # node 3 forgets its term-1 vote for node 1 on restart and gives node 2 a
    # term-1 vote as well — and term 1 ends up with two Leaders.
    async def save_nothing(self, current_term, voted_for):
        return None

    monkeypatch.setattr(SqliteStore, "save_term_and_vote", save_nothing)
    cluster = InProcessCluster(tmp_path, [1, 2, 3], check_votes_on_disk=False)
    await cluster.start_all()
    try:
        first = await cluster.fire_election_timeout(1)
        await cluster.deliver_answer(1, 3, first, await cluster.ask_for_vote(3, first))
        await cluster.restart(3)
        second = await cluster.fire_election_timeout(2)
        await cluster.deliver_answer(2, 3, second, await cluster.ask_for_vote(3, second))
        assert cluster.leaders_by_term[1] == {1, 2}
    finally:
        await cluster.stop_all()


# --- ELECT-9, ELECT-10: a Candidate missing a committed entry cannot win ----------


async def test_a_candidate_missing_a_committed_entry_cannot_win(tmp_path):
    # Five nodes, all holding one entry from term 1. Node 1 led term 2 and
    # appended three entries to its own log only, then crashed. Node 2 led
    # term 3 and committed one entry on nodes 2, 3, and 4 — a majority — then
    # crashed. Node 5 missed it. Node 1 comes back and runs in term 4.
    cluster = InProcessCluster(tmp_path, [1, 2, 3, 4, 5])
    await cluster.seed(1, [1, 2, 2, 2], current_term=3)
    for node_id in (3, 4):
        await cluster.seed(node_id, [1, 3], current_term=3, voted_for=2)
    await cluster.seed(5, [1], current_term=3, voted_for=2)
    for node_id in (1, 3, 4, 5):  # node 2 stays down
        await cluster.start(node_id)
    try:
        # Node 1's log is the longest, but its last term (2) is older than the
        # committed entry's (3): nodes 3 and 4 refuse; only node 5 grants.
        # Two votes of five is not a majority.
        request = await cluster.run_election(1)
        assert request.term == 4
        assert cluster.nodes[1].candidacy.votes_granted == frozenset({1, 5})
        assert cluster.leaders() == set()

        # Node 3 holds the committed entry, and wins the next term.
        await cluster.run_election(3)
        assert cluster.leaders() == {3}
        assert cluster.nodes[3].current_term == 5
        assert cluster.nodes[3].log[1].term == 3  # the committed entry is on the Leader
        cluster.assert_at_most_one_leader_per_term()
    finally:
        await cluster.stop_all()


# --- Late and higher-term answers ----------------------------------------------------


async def test_a_delayed_grant_from_an_earlier_election_cannot_create_a_second_leader(
    three_nodes,
):
    # Term 1: node 1 asks node 2, which grants — but the answer is held up.
    stale_request = await three_nodes.fire_election_timeout(1)
    stale_grant = await three_nodes.ask_for_vote(2, stale_request)
    assert stale_grant.vote_granted

    # Node 3 (cut off from node 1) loses term 1, then wins term 2 with node 2.
    await three_nodes.run_election(3, reachable={2})
    await three_nodes.run_election(3, reachable={2})
    assert three_nodes.leaders() == {3}
    assert three_nodes.nodes[3].current_term == 2

    # Node 1 has not heard of term 2. It runs in term 2 too; node 2 has
    # already voted for node 3 in term 2 and refuses.
    await three_nodes.run_election(1, reachable={2})
    assert three_nodes.nodes[1].role is Role.CANDIDATE
    assert three_nodes.nodes[1].current_term == 2

    # Now the term-1 grant finally arrives. Counting it would give node 1 two
    # votes of three in term 2 — and term 2 would have two Leaders.
    assert await three_nodes.deliver_answer(1, 2, stale_request, stale_grant) is False
    assert three_nodes.leaders() == {3}
    three_nodes.assert_at_most_one_leader_per_term()


async def test_a_candidate_answered_with_a_higher_term_steps_down(tmp_path):
    cluster = InProcessCluster(tmp_path, [1, 2, 3])
    await cluster.seed(3, current_term=5)
    await cluster.start_all()
    try:
        request = await cluster.run_election(1, reachable={3})
        assert request.term == 1
        node = cluster.nodes[1]
        assert (node.role, node.current_term, node.voted_for) == (Role.FOLLOWER, 5, None)
        persisted = await reload(cluster.paths[1])
        assert (persisted.current_term, persisted.voted_for) == (5, None)
    finally:
        await cluster.stop_all()


async def test_a_newer_election_replaces_an_old_leader_one_leader_per_term(three_nodes):
    await three_nodes.run_election(1)
    # Node 3 stops hearing from node 1 and wins term 2 with node 2's vote.
    # For a moment node 1 still believes it leads term 1: two nodes are
    # Leaders, but of different terms, which is allowed.
    await three_nodes.run_election(3, reachable={2})
    assert three_nodes.leaders() == {1, 3}
    assert three_nodes.leaders_by_term == {1: {1}, 2: {3}}

    # The first message from term 2 to reach node 1 ends its leadership.
    await three_nodes.ask_for_vote(1, await three_nodes.fire_election_timeout(2))
    assert three_nodes.nodes[1].role is Role.FOLLOWER
    assert three_nodes.nodes[1].leadership is None
    three_nodes.assert_at_most_one_leader_per_term()


# --- Randomized schedules -------------------------------------------------------------


def assert_role_records_match(node):
    """A Candidate has a Candidacy, a Leader a Leadership, each for its current term."""
    if node.role is Role.CANDIDATE:
        assert node.candidacy.term == node.current_term and node.leadership is None
    elif node.role is Role.LEADER:
        assert node.leadership.term == node.current_term and node.candidacy is None
    else:
        assert node.candidacy is None and node.leadership is None


@pytest.mark.parametrize("seed", range(25))
async def test_random_schedules_never_elect_two_leaders_in_one_term(tmp_path, seed):
    # Deterministic pseudo-random schedules over three or five nodes, each
    # starting from its own log and term: timeouts at any moment, crashes and
    # restarts, and messages delivered late, out of order, more than once, or
    # not at all. After every step: no term has had two Leaders, no node has
    # voted for two Candidates in one term — across restarts too — and every
    # node's role records match its role and term.
    rng = random.Random(seed)
    members = list(range(1, rng.choice([3, 5]) + 1))
    cluster = InProcessCluster(tmp_path, members)
    for node_id in members:
        log_terms = sorted(rng.choices([1, 2, 3], k=rng.randint(0, 4)))
        await cluster.seed(node_id, log_terms, current_term=max(log_terms, default=0))
    await cluster.start_all()
    votes_given = defaultdict(set)
    try:
        for _ in range(150):
            roll = rng.random()
            if roll < 0.15:
                ready = [n for n, node in cluster.nodes.items() if node.role is not Role.LEADER]
                if ready:
                    candidate = rng.choice(ready)
                    request = await cluster.fire_election_timeout(candidate)
                    votes_given[(candidate, request.term)].add(candidate)
                    cluster.send_requests(request)
            elif roll < 0.20:
                await cluster.restart(rng.choice(members))
            elif cluster.in_flight:
                message = rng.choice(cluster.in_flight)
                fate = rng.random()
                if fate < 0.1:
                    cluster.drop(message)
                else:
                    response = await cluster.deliver(message, keep_copy=fate < 0.3)
                    if response is not None and response.vote_granted:
                        votes_given[(message.voter, message.request.term)].add(
                            message.request.candidate_id
                        )

            cluster.assert_at_most_one_leader_per_term()
            for (voter, term), candidates in votes_given.items():
                assert len(candidates) == 1, f"node {voter} voted for {candidates} in term {term}"
            for node in cluster.nodes.values():
                assert_role_records_match(node)

        # Then the whole cluster restarts, so every node is a Follower
        # holding only what it persisted (STATE-2), and the network goes
        # quiet: nothing in flight, everyone reachable. Nodes take turns
        # timing out, those with the most up-to-date logs first, since only
        # they can win. The first one to try catches up to every other
        # node's term from their answers, so by its second try at the latest
        # its term is new to everyone and it wins.
        cluster.in_flight.clear()
        for node_id in members:
            await cluster.restart(node_id)
        by_log = sorted(
            members,
            key=lambda n: (cluster.nodes[n].log.last_term, cluster.nodes[n].log.last_index),
            reverse=True,
        )
        for candidate in by_log + by_log[:1]:
            await cluster.run_election(candidate)
            if cluster.leaders():
                break
        assert len(cluster.leaders()) == 1
        cluster.assert_at_most_one_leader_per_term()
    finally:
        await cluster.stop_all()
