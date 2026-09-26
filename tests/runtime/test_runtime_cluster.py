"""Tier 2 tests of whole clusters that run themselves; only clock, links, and crashes are scripted.

After every tick no node has failed, no term has two Leaders, no committed entry has changed, no
cluster time has fallen, and no client request has taken effect twice (RuntimeCluster.check).
ELECT-2, ELECT-11, ELECT-12, REPL-9, REPL-10, APPLY-4, CLIENT-6, FAIL-2, FAIL-3, FAIL-6, DD-9,
DD-15, DD-30, DD-32.
"""

import random

import pytest

from raftkv.consensus import NotLeaderError, Role
from raftkv.kvstore import OpenSession, decode_command
from raftkv.runtime import Timing
from tests.support.kv_client import KvClient
from tests.support.waiting import eventually

# NOTE: an election timeout is 10 to 19 ticks; a split vote costs another, so 100 ticks leaves
# room for several rounds.
ELECTION_BOUND = 100


async def elect(cluster):
    """Tick until one Leader is known to every running node; return its ID."""
    await cluster.run_until(cluster.has_one_leader_known_to_all, within=ELECTION_BOUND)
    return cluster.leader()


async def elect_with_client(cluster):
    """Elect a Leader and open a session through it; return the Leader and the session's client."""
    leader = await elect(cluster)
    return leader, KvClient(await cluster.open_session(leader))


@pytest.mark.parametrize("size", [1, 3, 5])
async def test_a_fresh_cluster_elects_one_leader_that_every_node_knows(start_runtime_cluster, size):
    cluster = await start_runtime_cluster(list(range(1, size + 1)))
    leader = await elect(cluster)
    leading = {n for n, node in cluster.nodes.items() if node.durable.role is Role.LEADER}
    assert leading == {leader}
    assert len({node.durable.current_term for node in cluster.nodes.values()}) == 1


async def test_a_command_given_to_the_leader_reaches_every_nodes_map(start_runtime_cluster):
    cluster = await start_runtime_cluster([1, 2, 3])
    leader, client = await elect_with_client(cluster)

    await cluster.append_command(leader, client.put("x", "1"))
    await cluster.run_until(cluster.all_caught_up, within=2)

    assert cluster.maps() == {n: {"x": "1"} for n in (1, 2, 3)}


async def test_a_follower_refuses_a_command(start_runtime_cluster):
    cluster = await start_runtime_cluster([1, 2, 3])
    leader = await elect(cluster)
    follower = next(n for n in cluster.nodes if n != leader)
    with pytest.raises(NotLeaderError):
        await cluster.append_command(follower, OpenSession().encode())


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
    old, client = await elect_with_client(cluster)
    await cluster.append_command(old, client.put("x", "1"))
    await cluster.run_until(cluster.all_caught_up, within=2)
    old_term = cluster.nodes[old].durable.current_term

    await cluster.stop(old)
    new = await elect(cluster)
    await cluster.append_command(new, client.put("y", "2"))  # the session outlives its Leader
    await cluster.start(old)
    await cluster.run_until(cluster.all_caught_up, within=ELECTION_BOUND)

    assert cluster.nodes[new].durable.current_term > old_term
    assert cluster.maps() == {n: {"x": "1", "y": "2"} for n in (1, 2, 3)}


async def test_a_cut_off_leader_steps_down_on_return_and_loses_what_it_never_committed(
    start_runtime_cluster,
):
    cluster = await start_runtime_cluster([1, 2, 3])
    old, client = await elect_with_client(cluster)
    cluster.network.isolate(old, cluster.member_ids)
    await cluster.append_command(old, client.put("lost", "1"))  # reaches no Follower

    await cluster.run_until(lambda: cluster.leader() not in (None, old), within=ELECTION_BOUND)
    new = cluster.leader()
    await cluster.append_command(new, client.put("kept", "2"))
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
    await cluster.append_command(leader, OpenSession().encode())

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


# --- Cluster time and sessions (DD-15, DD-32) -------------------------------------------


async def test_cluster_time_counts_leader_ticks_and_carries_over_a_change_of_leader(
    start_runtime_cluster,
):
    cluster = await start_runtime_cluster([1, 2, 3])
    old, client = await elect_with_client(cluster)
    await cluster.tick(20)
    await cluster.append_command(old, client.put("x", "1"))
    stamped = cluster.nodes[old].durable.log.last_cluster_time
    assert stamped >= 20  # at least the twenty ticks it has led since

    await cluster.stop(old)
    new = await elect(cluster)

    # The new Leader resumes from the last time in its log: its empty entry carries that time,
    # and the election, which had no Leader, added nothing.
    log = cluster.nodes[new].durable.log
    assert log.entry_at(log.last_index).is_empty
    assert log.last_cluster_time == log.entry_at(log.last_index - 1).cluster_time == stamped


async def test_an_idle_session_expires_on_every_node_and_its_retry_is_refused(
    start_runtime_cluster,
):
    cluster = await start_runtime_cluster([1, 2, 3], session_timeout=30)
    leader, idle = await elect_with_client(cluster)
    await cluster.append_command(leader, idle.put("x", "1"))
    await cluster.tick(40)  # more than the timeout, counted on the Leader's clock

    busy = KvClient(await cluster.open_session(leader))  # applying this expires `idle`
    await cluster.append_command(leader, busy.put("x", "2"))
    await cluster.append_command(leader, idle.retry())  # applied, it would put x back to 1
    await cluster.run_until(cluster.all_caught_up, within=2)

    for kv in cluster.files.kv.values():
        assert kv.session(idle.client_id) is None
        assert kv.as_dict() == {"x": "2"}
    assert cluster.files.times_applied(idle.client_id, 1) == {1: 1, 2: 1, 3: 1}


@pytest.mark.parametrize(
    ("members", "seed"),
    [
        *(pytest.param([1, 2, 3], seed, id=f"3-nodes-seed-{seed}") for seed in range(12)),
        *(pytest.param([1, 2, 3, 4, 5], seed, id=f"5-nodes-seed-{seed}") for seed in range(6)),
    ],
)
async def test_retried_requests_take_effect_exactly_once_through_crashes_and_partitions(
    start_runtime_cluster, members, seed
):
    # Three clients each keep one request outstanding, on keys they share, and resend it, with
    # the same number, to whichever node leads, until they hear it was applied; half the time
    # an answer is lost and the client resends a request already applied. Meanwhile nodes are
    # cut off, healed, and restarted. No node may apply a request twice, or change its map for
    # a request that took no effect (RuntimeCluster.check). At the end every acknowledged
    # request took effect once on every node, and every map holds, for each key, the value of
    # the last put that took effect on it.
    rng = random.Random(seed)
    cluster = await start_runtime_cluster(members, seed=seed)
    leader = await elect(cluster)
    clients = [KvClient(await cluster.open_session(leader)) for _ in range(3)]
    outstanding = {}  # client -> its request awaiting an answer
    written = {}  # (client ID, request number) -> (key, value)
    acknowledged = set()
    for step in range(150):
        roll = rng.random()
        if roll < 0.03:
            cluster.network.isolate(rng.choice(members), members)
        elif roll < 0.06:
            cluster.network.heal()
        elif roll < 0.08 and len(cluster.nodes) == len(members):
            await cluster.restart(rng.choice(members))
        for client in clients:
            if client not in outstanding and rng.random() < 0.2:
                key, value = rng.choice("abc"), f"{client.client_id}:{step}"
                outstanding[client] = client.put(key, value)
                written[(client.client_id, client.last_seq)] = (key, value)
            leader = cluster.leader()
            if client in outstanding and leader is not None and rng.random() < 0.3:
                try:
                    await cluster.append_command(leader, outstanding[client])
                except NotLeaderError:
                    pass
        await cluster.tick()
        for client, command in list(outstanding.items()):
            request = decode_command(command)
            applied = cluster.files.has_applied(request.client_id, request.seq)
            if applied and rng.random() < 0.5:  # otherwise the answer is lost
                acknowledged.add((request.client_id, request.seq))
                del outstanding[client]

    cluster.network.heal()
    await cluster.run_until(
        lambda: cluster.has_one_leader_known_to_all() and cluster.all_caught_up(),
        within=2 * ELECTION_BOUND,
    )
    # Once healed, the cluster takes a new command everywhere; every entry ever committed
    # survives in the final Leader's log; every node holds the same map; and every
    # acknowledged request took effect exactly once on every node.
    last = KvClient(await cluster.open_session(cluster.leader()))
    await cluster.append_command(cluster.leader(), last.put("after", "healing"))
    written[(last.client_id, last.last_seq)] = ("after", "healing")
    await cluster.run_until(cluster.all_caught_up, within=2)
    final_log = cluster.nodes[cluster.leader()].durable.log
    committed = cluster.files.committed
    assert all(final_log.entry_at(index) == entry for index, (entry, _) in committed.items())
    maps = list(cluster.maps().values())
    assert maps[0]["after"] == "healing"
    assert all(m == maps[0] for m in maps)
    for node_id in cluster.nodes:
        effects = cluster.files.effects[node_id]
        assert all(effects.count(request) == 1 for request in acknowledged)
        expected = {}
        for request in effects:
            key, value = written[request]
            expected[key] = value
        assert cluster.files.kv[node_id].as_dict() == expected


async def test_a_started_cluster_elects_and_replicates_on_its_own_clock(start_runtime_cluster):
    # Real time can depose a Leader before a command commits, as Raft allows, so a request is
    # given again, with the same number, to whichever node leads, as a client would.
    cluster = await start_runtime_cluster(
        [1, 2, 3], timing=Timing(tick_interval=0.005, heartbeat_ticks=1, election_ticks=10)
    )
    for node in cluster.nodes.values():
        node.start()

    async def committed_as_sent(command):
        """Give `command` to the Leader until it commits in the term it was appended in."""
        for _ in range(5):
            await eventually(cluster.has_one_leader_known_to_all)
            node = cluster.nodes[cluster.leader()]
            try:
                position = await node.append_command(command)
                await eventually(
                    lambda node=node, index=position.index: node.durable.commit_index >= index,
                    timeout=1,
                )
            except (NotLeaderError, AssertionError):
                continue
            if node.durable.log.term_at(position.index) == position.term:
                return position
        raise AssertionError(f"{command} never committed")

    client = KvClient((await committed_as_sent(OpenSession().encode())).index)
    await committed_as_sent(client.put("x", "1"))
    await eventually(lambda: all(m == {"x": "1"} for m in cluster.maps().values()))
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
        await cluster.append_command(leader, OpenSession().encode())
