"""Tier 2 elections across whole clusters of real nodes, driven step by step by the test.

ELECT-8 through ELECT-12, PERSIST-1, PERSIST-2, PERSIST-5, STATE-2. Every vote is read back
from its voter's file before its answer is handed on.
"""

import random

import pytest

from raftkv.consensus import Role
from raftkv.persistence import SqliteStore
from tests.persistence.store_doubles import term_and_vote_on_disk


@pytest.fixture
async def three_nodes(start_cluster):
    """Return a running cluster of fresh nodes 1, 2, and 3."""
    return await start_cluster([1, 2, 3])


# --- A cold start ---------------------------------------------------------------------


async def test_cold_start_elects_exactly_one_leader(three_nodes):
    # Every node starts as a Follower in term 0 (STATE-2); node 1's timeout fires first.
    assert three_nodes.leaders() == set()
    await three_nodes.run_election(1)

    assert three_nodes.leaders() == {1}
    leader = three_nodes.nodes[1]
    assert (leader.current_term, leader.leadership.followers) == (1, frozenset({2, 3}))
    for voter in (2, 3):
        node = three_nodes.nodes[voter]
        assert (node.role, node.current_term, node.voted_for) == (Role.FOLLOWER, 1, 1)
    three_nodes.assert_election_safety()


async def test_every_vote_of_the_election_survives_a_restart_of_the_whole_cluster(three_nodes):
    await three_nodes.run_election(1)
    for node_id in (1, 2, 3):
        await three_nodes.restart(node_id)
    for node_id in (1, 2, 3):
        node = three_nodes.nodes[node_id]
        assert (node.role, node.current_term, node.voted_for) == (Role.FOLLOWER, 1, 1)


# --- Split votes ----------------------------------------------------------------------


async def test_a_three_way_split_vote_is_resolved_in_the_next_term(three_nodes):
    # All three time out at once and vote for themselves, so every request reaches a node
    # that has already voted in term 1.
    requests = {n: await three_nodes.fire_election_timeout(n) for n in (1, 2, 3)}
    for candidate, request in requests.items():
        for voter in three_nodes.nodes[candidate].peers:
            response = await three_nodes.ask_for_vote(voter, request)
            assert response.vote_granted is False
            await three_nodes.deliver_vote_response(voter, request, response)
    assert three_nodes.leaders() == set()
    assert {node.role for node in three_nodes.nodes.values()} == {Role.CANDIDATE}

    # Randomized timeouts make another collision unlikely; node 2 times out first.
    await three_nodes.run_election(2)
    assert three_nodes.leaders() == {2}
    assert three_nodes.nodes[2].current_term == 2
    three_nodes.assert_election_safety()


async def test_a_two_two_split_in_four_nodes_elects_nobody_in_that_term(start_cluster):
    cluster = await start_cluster([1, 2, 3, 4])
    first = await cluster.fire_election_timeout(1)
    second = await cluster.fire_election_timeout(2)
    for voter, request in ((3, first), (4, second)):
        response = await cluster.ask_for_vote(voter, request)
        assert response.vote_granted
        await cluster.deliver_vote_response(voter, request, response)
    # Each Candidate holds 2 votes of 4: half, which is not a majority.
    assert cluster.nodes[1].candidacy.votes_granted == frozenset({1, 3})
    assert cluster.nodes[2].candidacy.votes_granted == frozenset({2, 4})
    assert cluster.leaders() == set()

    await cluster.run_election(3)
    assert cluster.leaders() == {3}
    cluster.assert_election_safety()


# --- A restarted voter (PERSIST-2, PERSIST-5, ELECT-8) --------------------------------


async def test_a_restarted_voter_refuses_a_second_candidate_in_the_same_term(three_nodes):
    first = await three_nodes.fire_election_timeout(1)
    assert (await three_nodes.ask_for_vote(3, first)).vote_granted
    await three_nodes.restart(3)  # crashes right after answering
    assert three_nodes.nodes[3].voted_for == 1

    second = await three_nodes.fire_election_timeout(2)
    assert second.term == first.term == 1
    assert (await three_nodes.ask_for_vote(3, second)).vote_granted is False


@pytest.mark.negative_control
async def test_negative_control_unpersisted_votes_let_a_restart_elect_two_leaders(
    start_cluster, monkeypatch
):
    # Proves the test above can fail: with votes never written to disk, node 3 forgets its
    # term-1 vote for node 1 on restart and gives node 2 a term-1 vote as well.
    async def save_nothing(self, current_term, voted_for):
        return None

    monkeypatch.setattr(SqliteStore, "save_term_and_vote", save_nothing)
    cluster = await start_cluster([1, 2, 3], check_votes_on_disk=False)

    await cluster.run_election(1, reachable={3})
    await cluster.restart(3)
    await cluster.run_election(2, reachable={3})

    assert cluster.leaders_by_term[1] == {1, 2}
    with pytest.raises(AssertionError, match=r"term 1 had leaders \[1, 2\]"):
        cluster.assert_election_safety()


# --- A Candidate missing a committed entry cannot win (ELECT-9, ELECT-10) -------------


async def test_a_candidate_missing_a_committed_entry_cannot_win(start_cluster):
    # Five nodes, all holding one entry from term 1. Node 1 led term 2 and appended three
    # entries to its own log only, then crashed. Node 2 led term 3 and committed one entry on
    # nodes 2, 3, and 4, a majority, then crashed and stays down. Node 5 missed that entry.
    # Node 1 comes back and runs in term 4.
    cluster = await start_cluster(
        [1, 2, 3, 4, 5],
        preload={
            1: {"log_terms": [1, 2, 2, 2], "current_term": 3},
            3: {"log_terms": [1, 3], "current_term": 3, "voted_for": 2},
            4: {"log_terms": [1, 3], "current_term": 3, "voted_for": 2},
            5: {"log_terms": [1], "current_term": 3, "voted_for": 2},
        },
        down=[2],
    )

    # Node 1's log is the longest, but its last term (2) is older than the committed entry's
    # (3): nodes 3 and 4 refuse, and only node 5 grants. Two votes of five is not a majority.
    request = await cluster.run_election(1)
    assert request.term == 4
    assert cluster.nodes[1].candidacy.votes_granted == frozenset({1, 5})
    assert cluster.leaders() == set()

    # Node 3 holds the committed entry and wins the next term.
    await cluster.run_election(3)
    assert cluster.leaders() == {3}
    assert cluster.nodes[3].current_term == 5
    assert cluster.nodes[3].log.term_at(2) == 3  # the committed entry is on the Leader
    cluster.assert_election_safety()


# --- Late and higher-term answers -----------------------------------------------------


async def test_a_delayed_grant_from_an_earlier_election_cannot_create_a_second_leader(
    three_nodes,
):
    # Term 1: node 1 asks node 2, which grants, but the answer is held up.
    stale_request = await three_nodes.fire_election_timeout(1)
    stale_grant = await three_nodes.ask_for_vote(2, stale_request)
    assert stale_grant.vote_granted

    # Node 3, cut off from node 1, loses term 1 and then wins term 2 with node 2's vote.
    await three_nodes.run_election(3, reachable={2})
    await three_nodes.run_election(3, reachable={2})
    assert three_nodes.leaders() == {3}
    assert three_nodes.nodes[3].current_term == 2

    # Node 1 has not heard of term 2 and times out into it too. No term-2 answer has reached it,
    # so node 2 has not answered this election yet.
    await three_nodes.fire_election_timeout(1)
    assert three_nodes.nodes[1].role is Role.CANDIDATE
    assert three_nodes.nodes[1].current_term == 2

    # The term-1 grant finally arrives; only its term marks it as stale. Counting it would give
    # node 1 two votes of three in term 2, and term 2 two Leaders.
    assert await three_nodes.deliver_vote_response(2, stale_request, stale_grant) is False
    assert three_nodes.nodes[1].candidacy.votes_granted == frozenset({1})
    assert three_nodes.leaders() == {3}
    three_nodes.assert_election_safety()


async def test_a_candidate_answered_with_a_higher_term_steps_down(start_cluster):
    cluster = await start_cluster([1, 2, 3], preload={3: {"current_term": 5}})
    request = await cluster.run_election(1, reachable={3})
    assert request.term == 1
    node = cluster.nodes[1]
    assert (node.role, node.current_term, node.voted_for) == (Role.FOLLOWER, 5, None)
    assert await term_and_vote_on_disk(cluster.paths[1]) == (5, None)


async def test_a_newer_election_replaces_an_old_leader_one_leader_per_term(three_nodes):
    await three_nodes.run_election(1)
    # Node 3 stops hearing from node 1 and wins term 2 with node 2's vote. Node 1 still
    # believes it leads term 1: two Leaders, but of different terms, which is allowed.
    await three_nodes.run_election(3, reachable={2})
    assert three_nodes.leaders() == {1, 3}
    assert three_nodes.leaders_by_term == {1: {1}, 2: {3}}

    # Node 2 times out into term 3. Its RequestVote, the first message from a later term to reach
    # node 1, ends node 1's leadership.
    await three_nodes.ask_for_vote(1, await three_nodes.fire_election_timeout(2))
    assert three_nodes.nodes[1].role is Role.FOLLOWER
    assert three_nodes.nodes[1].leadership is None
    three_nodes.assert_election_safety()


# --- Randomized schedules -------------------------------------------------------------

# How often a step fires a timeout, by cluster size. A five-node election needs twice the
# deliveries of a three-node one, so its timeouts fire less often, or few elections would finish.
TIMEOUT_PROBABILITY = {3: 0.15, 5: 0.06}
RESTART_PROBABILITY = 0.05


def random_starting_files(members, rng):
    """Return a random starting log for each member, and a term matching its last entry."""
    starting_files = {}
    for node_id in members:
        log_terms = sorted(rng.choices([1, 2, 3], k=rng.randint(0, 4)))
        starting_files[node_id] = {
            "log_terms": log_terms,
            "current_term": max(log_terms, default=0),
        }
    return starting_files


async def take_one_random_step(cluster, rng):
    """Take one random step: a timeout, a restart, or a message delivered, repeated, or lost."""
    timeout_probability = TIMEOUT_PROBABILITY[len(cluster.member_ids)]
    roll = rng.random()
    if roll < timeout_probability:
        ready = [n for n, node in cluster.nodes.items() if node.role is not Role.LEADER]
        if ready:
            cluster.send_requests(await cluster.fire_election_timeout(rng.choice(ready)))
    elif roll < timeout_probability + RESTART_PROBABILITY:
        await cluster.restart(rng.choice(cluster.member_ids))
    elif cluster.in_flight:
        message = rng.choice(cluster.in_flight)
        fate = rng.random()
        if fate < 0.1:
            cluster.drop(message)
        else:
            await cluster.deliver(message, keep_copy=fate < 0.3)


async def restart_all_and_elect_on_a_quiet_network(cluster):
    """Restart every node, empty the network, and run elections until one node wins.

    Every node restarts as a Follower holding only what it persisted (STATE-2). Nodes time out
    one at a time, most up-to-date log first, since only those can win. The first learns every
    node's term from their answers, so by its second try its term is new to all and it wins.
    Election safety is checked after every election.
    """
    cluster.in_flight.clear()
    for node_id in cluster.member_ids:
        await cluster.restart(node_id)
    by_log = sorted(
        cluster.member_ids,
        key=lambda n: (cluster.nodes[n].log.last_term, cluster.nodes[n].log.last_index),
        reverse=True,
    )
    for candidate in by_log + by_log[:1]:
        await cluster.run_election(candidate)
        cluster.assert_election_safety()
        if cluster.leaders():
            return


@pytest.mark.parametrize("seed", range(25), ids=lambda seed: f"seed={seed}")
async def test_random_schedules_never_elect_two_leaders_in_one_term(start_cluster, seed):
    # Three or five nodes, each starting from its own log and term; timeouts at any moment,
    # crashes and restarts, and messages delivered late, out of order, twice, or never. Most
    # seeds elect Leaders during the random steps, not only in the quiet phase.
    rng = random.Random(seed)
    members = list(range(1, rng.choice([3, 5]) + 1))
    cluster = await start_cluster(members, preload=random_starting_files(members, rng))

    for _ in range(150):
        await take_one_random_step(cluster, rng)
        cluster.assert_election_safety()

    await restart_all_and_elect_on_a_quiet_network(cluster)
    assert len(cluster.leaders()) == 1
    cluster.assert_election_safety()
