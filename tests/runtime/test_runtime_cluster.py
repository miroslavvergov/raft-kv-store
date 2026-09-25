"""Tier 2 tests of whole clusters that run themselves; only clock, links, and crashes are scripted.

After every tick no node has failed, no term has two Leaders, and no committed entry has changed
(RuntimeCluster.check). ELECT-2, ELECT-11, ELECT-12, REPL-9, REPL-10, APPLY-4, CLIENT-6, FAIL-2,
FAIL-3, DD-9, DD-30.
"""

import random

import pytest

from raftkv.consensus import NotLeaderError, Role
from raftkv.kvstore import KeyValueStore
from raftkv.runtime import Timing
from tests.support.waiting import eventually

# NOTE: an election timeout is 10 to 19 ticks; a split vote costs another, so 100 ticks leaves
# room for several rounds.
ELECTION_BOUND = 100


def put(key, value):
    return KeyValueStore.put_command(key, value)


async def elect(cluster):
    """Tick until one Leader is known to every running node; return its ID."""
    await cluster.run_until(cluster.has_one_leader_known_to_all, within=ELECTION_BOUND)
    return cluster.leader()


@pytest.mark.parametrize("size", [1, 3, 5])
async def test_a_fresh_cluster_elects_one_leader_that_every_node_knows(start_runtime_cluster, size):
    cluster = await start_runtime_cluster(list(range(1, size + 1)))
    leader = await elect(cluster)
    leading = {n for n, node in cluster.nodes.items() if node.durable.role is Role.LEADER}
    assert leading == {leader}
    assert len({node.durable.current_term for node in cluster.nodes.values()}) == 1


async def test_a_command_given_to_the_leader_reaches_every_nodes_map(start_runtime_cluster):
    cluster = await start_runtime_cluster([1, 2, 3])
    leader = await elect(cluster)

    await cluster.append_command(leader, put("x", "1"))
    await cluster.run_until(cluster.all_caught_up, within=2)

    assert cluster.maps() == {n: {"x": "1"} for n in (1, 2, 3)}


async def test_a_follower_refuses_a_command(start_runtime_cluster):
    cluster = await start_runtime_cluster([1, 2, 3])
    leader = await elect(cluster)
    follower = next(n for n in cluster.nodes if n != leader)
    with pytest.raises(NotLeaderError):
        await cluster.append_command(follower, put("x", "1"))


async def test_a_quiet_cluster_keeps_its_leader(start_runtime_cluster):
    # REPL-10: heartbeats arrive every tick, well before any election timeout.
    cluster = await start_runtime_cluster([1, 2, 3])
    leader = await elect(cluster)
    term = cluster.nodes[leader].durable.current_term

    await cluster.tick(200)

    assert cluster.leader() == leader
    assert {node.durable.current_term for node in cluster.nodes.values()} == {term}


async def test_a_crashed_leader_is_replaced_and_what_it_committed_survives(start_runtime_cluster):
    cluster = await start_runtime_cluster([1, 2, 3])
    old = await elect(cluster)
    await cluster.append_command(old, put("x", "1"))
    await cluster.run_until(cluster.all_caught_up, within=2)
    old_term = cluster.nodes[old].durable.current_term

    await cluster.stop(old)
    new = await elect(cluster)
    await cluster.append_command(new, put("y", "2"))
    await cluster.start(old)
    await cluster.run_until(cluster.all_caught_up, within=ELECTION_BOUND)

    assert cluster.nodes[new].durable.current_term > old_term
    assert cluster.maps() == {n: {"x": "1", "y": "2"} for n in (1, 2, 3)}


async def test_a_cut_off_leader_steps_down_on_return_and_loses_what_it_never_committed(
    start_runtime_cluster,
):
    cluster = await start_runtime_cluster([1, 2, 3])
    old = await elect(cluster)
    cluster.network.isolate(old, cluster.member_ids)
    await cluster.append_command(old, put("lost", "1"))  # reaches no Follower, never commits

    await cluster.run_until(lambda: cluster.leader() not in (None, old), within=ELECTION_BOUND)
    new = cluster.leader()
    await cluster.append_command(new, put("kept", "2"))
    cluster.network.heal()
    await cluster.run_until(cluster.all_caught_up, within=ELECTION_BOUND)

    assert cluster.nodes[old].durable.role is Role.FOLLOWER
    assert cluster.maps() == {n: {"kept": "2"} for n in (1, 2, 3)}


async def test_a_minority_elects_no_leader_and_commits_nothing(start_runtime_cluster):
    cluster = await start_runtime_cluster([1, 2, 3, 4, 5])
    leader = await elect(cluster)
    minority = [leader, next(n for n in cluster.nodes if n != leader)]
    majority = [n for n in cluster.nodes if n not in minority]
    for a in minority:
        for b in majority:
            cluster.network.cut(a, b)
    await cluster.append_command(leader, put("minority", "1"))

    await cluster.tick(ELECTION_BOUND)

    assert cluster.nodes[leader].durable.commit_index == 1  # only its empty entry
    majority_leader = cluster.leader()
    assert majority_leader in majority
    assert all(
        cluster.nodes[n].durable.role is not Role.LEADER
        or cluster.nodes[n].durable.current_term
        < cluster.nodes[majority_leader].durable.current_term
        for n in minority
    )


@pytest.mark.parametrize(
    ("members", "seed"),
    [
        *(pytest.param([1, 2, 3], seed, id=f"3-nodes-seed-{seed}") for seed in range(12)),
        *(pytest.param([1, 2, 3, 4, 5], seed, id=f"5-nodes-seed-{seed}") for seed in range(6)),
    ],
)
async def test_random_crashes_and_partitions_break_no_rule_and_the_cluster_recovers(
    start_runtime_cluster, members, seed
):
    rng = random.Random(seed)
    cluster = await start_runtime_cluster(members, seed=seed)
    written = 0
    for _ in range(150):
        roll = rng.random()
        if roll < 0.03:
            cluster.network.isolate(rng.choice(members), members)
        elif roll < 0.06:
            cluster.network.heal()
        elif roll < 0.08 and len(cluster.nodes) == len(members):
            await cluster.restart(rng.choice(members))
        elif roll < 0.25 and cluster.leader() is not None:
            written += 1
            await cluster.append_command(cluster.leader(), put(f"k{written}", str(seed)))
        await cluster.tick()

    cluster.network.heal()
    await cluster.run_until(
        lambda: cluster.has_one_leader_known_to_all() and cluster.all_caught_up(),
        within=2 * ELECTION_BOUND,
    )
    # Once healed, the cluster takes a new command everywhere; every entry ever committed
    # survives in the final Leader's log; and every node holds the same map.
    await cluster.append_command(cluster.leader(), put("after", "healing"))
    await cluster.run_until(cluster.all_caught_up, within=2)
    final_log = cluster.nodes[cluster.leader()].durable.log
    committed = cluster.files.committed
    assert all(final_log.entry_at(index) == entry for index, (entry, _) in committed.items())
    maps = list(cluster.maps().values())
    assert maps[0]["after"] == "healing"
    assert all(m == maps[0] for m in maps)


async def test_a_started_cluster_elects_and_replicates_on_its_own_clock(start_runtime_cluster):
    # Real time can depose a Leader before a command commits, as Raft allows, so the command
    # is given again to whichever node leads, as a client would; storing it twice is harmless.
    cluster = await start_runtime_cluster(
        [1, 2, 3], timing=Timing(tick_interval=0.005, heartbeat_ticks=1, election_ticks=10)
    )
    for node in cluster.nodes.values():
        node.start()

    def everywhere():
        return all(m == {"x": "1"} for m in cluster.maps().values())

    for _ in range(5):
        await eventually(cluster.has_one_leader_known_to_all)
        try:
            await cluster.nodes[cluster.leader()].append_command(put("x", "1"))
            await eventually(everywhere, timeout=1)
            break
        except (NotLeaderError, AssertionError):
            continue
    assert everywhere()
    cluster.check()


# --- The cluster's own checks can fail -------------------------------------------------


async def test_a_receiver_that_raises_fails_the_clusters_check(start_runtime_cluster, monkeypatch):
    cluster = await start_runtime_cluster([1, 2, 3])
    leader = await elect(cluster)
    follower = next(n for n in cluster.nodes if n != leader)

    async def raising(request):
        raise RuntimeError("broken receiver")

    monkeypatch.setattr(cluster.nodes[follower].durable, "handle_append_entries", raising)
    with pytest.raises(AssertionError, match="a receiver raised"):
        await cluster.tick()
    cluster.network.errors.clear()


async def test_a_node_that_fails_fails_the_clusters_check(start_runtime_cluster, monkeypatch):
    cluster = await start_runtime_cluster([1, 2, 3])
    leader = await elect(cluster)

    async def raising(max_entries=None):
        raise RuntimeError("broken state machine")

    monkeypatch.setattr(cluster.nodes[leader].durable, "apply_committed", raising)
    with pytest.raises(AssertionError, match="stopped on"):
        await cluster.append_command(leader, put("x", "1"))
