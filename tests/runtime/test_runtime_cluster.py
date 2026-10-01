"""Tier 2 tests of whole clusters that run themselves; only clock, links, and crashes are scripted.

After every tick no node has failed, no term has two Leaders, no committed entry has changed, no
cluster time has fallen, and no client request has taken effect twice (RuntimeCluster.check).
ELECT-2, ELECT-11, ELECT-12, REPL-9, REPL-10, APPLY-4, CLIENT-4, CLIENT-5, CLIENT-6, CLIENT-8,
CLIENT-9, CLIENT-10, FAIL-2, FAIL-3, FAIL-6, DD-9, DD-12, DD-15, DD-30, DD-32, DD-33, DD-34.
"""

import asyncio
import random
from dataclasses import dataclass

import pytest

from raftkv.consensus import NotLeaderError, Role
from raftkv.kvstore import OpenSession, PutApplied, decode_command
from raftkv.runtime import LeadershipLostError, NodeStoppedError, RaftNode, Timing
from tests.support.kv_client import KvClient
from tests.support.waiting import eventually, within_bound

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


async def disturb(cluster, rng, members):
    """Now and then cut a node off, restore every link, or restart a node.

    A restart takes a node with a proposal waiting, if any: restarting is what fails it.
    """
    roll = rng.random()
    if roll < 0.03:
        cluster.network.isolate(rng.choice(members), members)
    elif roll < 0.06:
        cluster.network.heal()
    elif roll < 0.08 and len(cluster.nodes) == len(members):
        waiting = [n for n in members if cluster.nodes[n].pending_proposals]
        await cluster.restart(rng.choice(waiting or members))


async def disturb_for_reads(cluster, rng, members):
    """Now and then cut a node off, split two nodes from the rest, restore links, or restart.

    Half the time the node cut off, or in the pair, is the Leader, so reads wait on a Leader that
    the others have left. Links come back within a few ticks more often than not, so a Leader cut
    off for less than an election timeout can rejoin with its reads still waiting. A restart takes
    a node with a read waiting, if any: restarting is what fails it with `NodeStoppedError`.
    """
    roll = rng.random()
    leader = cluster.leader()
    if roll < 0.04:
        victim = leader if leader is not None and rng.random() < 0.5 else rng.choice(members)
        cluster.network.isolate(victim, members)
    elif roll < 0.06 and len(members) > 3:
        pair = rng.sample(members, 2)  # in five nodes, a minority that may hold the Leader
        if leader is not None and leader not in pair and rng.random() < 0.5:
            pair[0] = leader
        for inside in pair:
            for outside in set(members) - set(pair):
                cluster.network.cut(inside, outside)
    elif roll < 0.16:
        cluster.network.heal()
    elif roll < 0.18 and len(cluster.nodes) == len(members):
        waiting = [n for n in members if cluster.nodes[n].pending_reads]
        await cluster.restart(rng.choice(waiting or members))


def previous_values(effects, written):
    """Return, for each request in `effects`, the value its key held just before it took effect.

    Args:
        effects: The requests, in the order they took effect.
        written: Each request's (key, value), by request.
    """
    held = {}  # key -> the value it holds after the requests so far
    previous = {}
    for request_id in effects:
        key, value = written[request_id]
        previous[request_id] = held.get(key)
        held[key] = value
    return previous


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


# --- Proposing (CLIENT-4, CLIENT-5, FAIL-6, DD-12, DD-33) -----------------------------------


async def test_a_proposal_through_the_leader_returns_its_result_once_the_leader_applies_it(
    start_runtime_cluster,
):
    cluster = await start_runtime_cluster([1, 2, 3])
    leader, client = await elect_with_client(cluster)

    first = await cluster.propose(leader, client.put("x", "1"))
    second = await cluster.propose(leader, client.put("x", "2"))

    assert (first, second) == (PutApplied(None), PutApplied("1"))
    await cluster.run_until(cluster.all_caught_up, within=2)
    assert cluster.maps() == {n: {"x": "2"} for n in (1, 2, 3)}
    assert all(node.pending_proposals == 0 for node in cluster.nodes.values())


async def test_a_proposal_lost_with_its_leader_fails_and_its_retry_applies_once(
    start_runtime_cluster,
):
    # The Leader is cut off before its entry reaches anyone, so the entry is overwritten by the
    # next Leader and the put never takes effect; the retry is its first application.
    cluster = await start_runtime_cluster([1, 2, 3])
    old_leader, client = await elect_with_client(cluster)
    cluster.network.isolate(old_leader, cluster.member_ids)

    proposal = asyncio.create_task(cluster.nodes[old_leader].propose(client.put("x", "1")))
    await cluster.wait_until_proposal_waits(old_leader)
    assert not proposal.done()

    await cluster.run_until(
        lambda: cluster.leader() not in (None, old_leader), within=ELECTION_BOUND
    )
    new_leader = cluster.leader()
    assert not proposal.done()  # the old Leader has not heard of the new one
    cluster.network.heal()
    await cluster.run_until(proposal.done, within=ELECTION_BOUND)

    with pytest.raises(LeadershipLostError):
        await proposal
    await cluster.run_until(cluster.all_caught_up, within=ELECTION_BOUND)
    assert cluster.files.times_applied(client.client_id, 1) == {1: 0, 2: 0, 3: 0}
    assert await cluster.propose(new_leader, client.retry()) == PutApplied(None)
    await cluster.run_until(cluster.all_caught_up, within=2)
    assert cluster.maps() == {n: {"x": "1"} for n in (1, 2, 3)}


async def test_a_proposal_that_fails_with_leadership_lost_may_still_take_effect(
    start_runtime_cluster,
):
    # The entry reaches one Follower, whose answer is lost, so the Leader cannot commit it. The
    # Leader is then cut off, that Follower wins the next election with the entry, and it commits
    # under the new Leader: the old Leader's caller is told leadership was lost, yet the put took
    # effect. Retrying with the same number answers with the first result and applies nothing,
    # though another client has written the key since.
    cluster = await start_runtime_cluster([1, 2, 3])
    old_leader, client = await elect_with_client(cluster)
    rival = KvClient(await cluster.open_session(old_leader))
    follower, other = (n for n in cluster.member_ids if n != old_leader)
    cluster.network.cut_one_way(follower, old_leader)
    cluster.network.cut(old_leader, other)

    proposal = asyncio.create_task(cluster.nodes[old_leader].propose(client.put("x", "1")))
    await cluster.wait_until_proposal_waits(old_leader)
    assert not proposal.done()
    logs = {n: cluster.nodes[n].durable.log.last_index for n in cluster.member_ids}
    assert logs[follower] == logs[old_leader]  # the entry reached the Follower...
    assert logs[other] < logs[old_leader]  # ...and not the other node

    cluster.network.isolate(old_leader, cluster.member_ids)
    await cluster.run_until(lambda: cluster.leader() == follower, within=ELECTION_BOUND)
    assert not proposal.done()  # the old Leader has not heard of the new one
    cluster.network.heal()
    await cluster.run_until(proposal.done, within=ELECTION_BOUND)

    with pytest.raises(LeadershipLostError):
        await proposal
    await cluster.run_until(cluster.all_caught_up, within=ELECTION_BOUND)
    assert cluster.files.times_applied(client.client_id, 1) == {1: 1, 2: 1, 3: 1}
    assert await cluster.propose(follower, rival.put("x", "2")) == PutApplied("1")
    assert await cluster.propose(follower, client.retry()) == PutApplied(None)  # the first result
    await cluster.run_until(cluster.all_caught_up, within=2)
    assert cluster.maps() == {n: {"x": "2"} for n in (1, 2, 3)}
    assert cluster.files.times_applied(client.client_id, 1) == {1: 1, 2: 1, 3: 1}


@pytest.mark.parametrize(
    ("members", "seed"),
    [
        *(pytest.param([1, 2, 3], seed, id=f"3-nodes-seed-{seed}") for seed in range(8)),
        *(pytest.param([1, 2, 3, 4, 5], seed, id=f"5-nodes-seed-{seed}") for seed in range(4)),
    ],
)
async def test_proposals_are_answered_with_their_own_results_through_crashes_and_partitions(
    start_runtime_cluster, members, seed
):
    # Three clients each keep one request outstanding, on keys they share, and propose it to the
    # Leader or, one time in five, to any node, which refuses unless it leads. A client gives up
    # on a proposal at random, and half the time a result it got is lost, so the request is
    # proposed again. Meanwhile nodes are cut off, healed, and restarted, a node with a proposal
    # waiting first. A proposal may fail only by leadership lost, a refusal, or a stop. Every
    # result must belong to that very request: the value its key held just before it took
    # effect, as the order in which requests took effect on the final Leader says. Once healed
    # and caught up, every proposal has ended.
    rng = random.Random(seed)
    cluster = await start_runtime_cluster(members, seed=seed)
    leader = await elect(cluster)
    clients = [KvClient(await cluster.open_session(leader)) for _ in range(3)]
    outstanding = {}  # client -> the request it has not yet heard answered
    attempts = {}  # client -> the proposal waiting now
    written = {}  # (client ID, request number) -> (key, value)
    results = {}  # (client ID, request number) -> the first result a proposal returned
    acknowledged = set()

    def collect(client, attempt):
        """Take what an ended proposal gave: a result, or one of the failures allowed."""
        error = attempt.exception()
        if error is not None:
            assert isinstance(error, NotLeaderError | LeadershipLostError | NodeStoppedError)
            return
        request = decode_command(outstanding[client])
        request_id = (request.client_id, request.seq)
        first = results.setdefault(request_id, attempt.result())
        assert attempt.result() == first, f"{request_id} was answered two different ways"
        if rng.random() < 0.5:  # otherwise the result is lost
            acknowledged.add(request_id)
            del outstanding[client]

    for step in range(150):
        await disturb(cluster, rng, members)
        for client in clients:
            if client not in outstanding and rng.random() < 0.2:
                key, value = rng.choice("abc"), f"{client.client_id}:{step}"
                outstanding[client] = client.put(key, value)
                written[(client.client_id, client.last_seq)] = (key, value)
            if client in outstanding and client not in attempts and rng.random() < 0.3:
                leader = cluster.leader()
                anywhere = leader is None or rng.random() < 0.2
                node = cluster.nodes[rng.choice(members) if anywhere else leader]
                attempts[client] = asyncio.create_task(node.propose(outstanding[client]))
        await cluster.tick()
        for client, attempt in list(attempts.items()):
            if not attempt.done():
                if rng.random() < 0.1:
                    attempt.cancel()  # the client gives up waiting
                    del attempts[client]
                continue
            del attempts[client]
            collect(client, attempt)

    cluster.network.heal()
    await cluster.run_until(
        lambda: cluster.has_one_leader_known_to_all() and cluster.all_caught_up(),
        within=2 * ELECTION_BOUND,
    )
    await cluster.tick(5)
    stuck = [client.client_id for client, attempt in attempts.items() if not attempt.done()]
    assert not stuck, f"proposals still waiting in a healed, caught-up cluster: clients {stuck}"
    for client, attempt in attempts.items():
        collect(client, attempt)
    assert all(node.pending_proposals == 0 for node in cluster.nodes.values())
    effects = cluster.files.effects[cluster.leader()]  # the order requests took effect in
    previous = previous_values(effects, written)
    assert all(effects.count(request_id) == 1 for request_id in acknowledged)
    for request_id, result in results.items():
        assert request_id in previous, f"{request_id} was answered but never took effect"
        assert result == PutApplied(previous[request_id]), f"the wrong answer to {request_id}"


# --- Reading (CLIENT-8, CLIENT-9, CLIENT-10, DD-12, DD-34) ---------------------------------


async def test_a_read_through_the_leader_reflects_a_write_the_leader_acknowledged(
    start_runtime_cluster,
):
    cluster = await start_runtime_cluster([1, 2, 3])
    leader, client = await elect_with_client(cluster)
    await cluster.propose(leader, client.put("x", "1"))

    index = await cluster.read_barrier(leader)

    assert index >= cluster.nodes[leader].durable.log.last_index
    assert cluster.maps()[leader]["x"] == "1"
    assert cluster.nodes[leader].pending_reads == 0


async def test_a_new_leader_serves_reads_that_reflect_what_the_old_leader_committed(
    start_runtime_cluster,
):
    cluster = await start_runtime_cluster([1, 2, 3])
    old_leader, client = await elect_with_client(cluster)
    await cluster.propose(old_leader, client.put("x", "1"))
    committed = cluster.nodes[old_leader].durable.log.last_index
    await cluster.run_until(cluster.all_caught_up, within=2)
    await cluster.stop(old_leader)
    new_leader = await elect(cluster)

    index = await cluster.read_barrier(new_leader)  # waits for the new Leader's empty entry

    assert index == committed + 1  # the old Leader's last entry, then this Leader's empty one


async def test_a_leader_cut_off_from_the_cluster_serves_no_read_once_a_newer_leader_committed(
    start_runtime_cluster,
):
    # The cut-off Leader still believes it leads. Serving a read from its own state would miss
    # what the new Leader commits, so its read waits for a majority that never answers, and
    # fails once the Leader hears of the new term (CLIENT-8).
    cluster = await start_runtime_cluster([1, 2, 3])
    old_leader, client = await elect_with_client(cluster)
    cluster.network.isolate(old_leader, cluster.member_ids)
    stale_read = asyncio.create_task(cluster.nodes[old_leader].read_barrier())
    await cluster.wait_until_read_waits(old_leader)

    await cluster.run_until(
        lambda: cluster.leader() not in (None, old_leader), within=ELECTION_BOUND
    )
    new_leader = cluster.leader()
    await cluster.propose(new_leader, client.put("x", "1"))  # commits under the new Leader
    committed = cluster.nodes[new_leader].durable.commit_index
    await cluster.tick(5)
    assert not stale_read.done()  # still cut off, still believing it leads, and serving nothing

    assert await cluster.read_barrier(new_leader) >= committed
    cluster.network.heal()
    await cluster.run_until(stale_read.done, within=ELECTION_BOUND)
    with pytest.raises(NotLeaderError):
        await stale_read


async def test_a_leader_cut_off_briefly_serves_its_waiting_read_once_reconnected(
    start_runtime_cluster,
):
    # Cut off for far less than an election timeout, the Leader still leads when its links
    # return, so a majority answers a request built after the read began, and it is served.
    cluster = await start_runtime_cluster([1, 2, 3])
    leader, _ = await elect_with_client(cluster)
    cluster.network.isolate(leader, cluster.member_ids)
    read = asyncio.create_task(cluster.nodes[leader].read_barrier())
    await cluster.wait_until_read_waits(leader)
    await cluster.tick(3)
    assert not read.done()
    assert cluster.leader() == leader

    cluster.network.heal()
    await cluster.tick(2)

    assert await within_bound(read) == cluster.nodes[leader].durable.commit_index


async def test_a_leader_with_only_one_follower_of_four_serves_no_read(start_runtime_cluster):
    # The Leader and one Follower are two of five: never a majority, however long the others
    # stay away. Once they return, the read is served or the Leader is deposed, never stale.
    cluster = await start_runtime_cluster([1, 2, 3, 4, 5])
    leader = await elect(cluster)
    commit_index = cluster.nodes[leader].durable.commit_index
    followers = [n for n in cluster.member_ids if n != leader]
    for node_id in followers[1:]:  # three of the four Followers are cut off from everyone
        cluster.network.isolate(node_id, cluster.member_ids)
    read = asyncio.create_task(cluster.nodes[leader].read_barrier())
    await cluster.wait_until_read_waits(leader)
    await cluster.tick(5)
    assert not read.done()

    cluster.network.heal()
    await cluster.run_until(read.done, within=ELECTION_BOUND)
    error = read.exception()
    if error is None:
        assert read.result() == commit_index  # a majority answered after the links returned
    else:
        assert isinstance(error, NotLeaderError)  # a node whose term had moved on deposed it


@dataclass
class ReadRecord:
    """A read a randomized test started.

    Attributes:
        floor: The highest commit index any node had when the read began.
        node: The RaftNode it ran on.
        ticks_waited: How many ticks it has been waiting.
    """

    floor: int
    node: RaftNode
    ticks_waited: int = 0


@pytest.mark.parametrize(
    ("members", "seed"),
    [
        *(pytest.param([1, 2, 3], seed, id=f"3-nodes-seed-{seed}") for seed in range(8)),
        *(pytest.param([1, 2, 3, 4, 5], seed, id=f"5-nodes-seed-{seed}") for seed in range(4)),
    ],
)
async def test_reads_never_fall_behind_earlier_commits_through_crashes_and_partitions(
    start_runtime_cluster, members, seed
):
    # Reads start on a Leader of any term or, one time in five, on any node, while a client keeps
    # writing and nodes are cut off, split, healed, and restarted. A read may fail only because
    # its node does not lead, or because the node was restarted. Every read served must reach at
    # least the highest commit index any node had when the read began: a Leader cut off from a
    # newer one would serve less, so this is what confirming through a majority prevents
    # (CLIENT-8). Once healed and caught up, every read has ended.
    rng = random.Random(seed)
    cluster = await start_runtime_cluster(members, seed=seed)
    leader = await elect(cluster)
    client = KvClient(await cluster.open_session(leader))
    reads = {}  # read -> its ReadRecord
    served = 0

    def collect(read, record):
        """Take what an ended read gave: an index at or above its floor, or an allowed failure."""
        nonlocal served
        error = read.exception()
        if error is not None:
            assert isinstance(error, NotLeaderError | NodeStoppedError)
            if isinstance(error, NodeStoppedError):
                assert cluster.nodes[record.node.node_id] is not record.node, "never restarted"
            return
        served += 1
        assert read.result() >= record.floor, f"served {read.result()}, behind {record.floor}"

    for _ in range(150):
        await disturb_for_reads(cluster, rng, members)
        leader = cluster.leader()
        if leader is not None and rng.random() < 0.4:
            try:
                await cluster.append_command(leader, client.put(rng.choice("abc"), "v"))
            except NotLeaderError:
                pass
        if rng.random() < 0.5:
            leaders = [
                n for n in sorted(cluster.nodes) if cluster.nodes[n].durable.role is Role.LEADER
            ]
            node_id = (
                rng.choice(leaders) if leaders and rng.random() >= 0.2 else rng.choice(members)
            )
            floor = max(node.durable.commit_index for node in cluster.nodes.values())
            reads[asyncio.create_task(cluster.nodes[node_id].read_barrier())] = ReadRecord(
                floor, cluster.nodes[node_id]
            )
        await cluster.tick()
        for read, record in list(reads.items()):
            if read.done():
                del reads[read]
                collect(read, record)
                continue
            record.ticks_waited += 1
            if record.ticks_waited >= 3 and rng.random() < 0.05:
                read.cancel()  # the client gives up waiting
                del reads[read]

    cluster.network.heal()
    await cluster.run_until(
        lambda: cluster.has_one_leader_known_to_all() and cluster.all_caught_up(),
        within=2 * ELECTION_BOUND,
    )
    await cluster.tick(5)
    stuck = sum(1 for read in reads if not read.done())
    assert not stuck, f"{stuck} reads still waiting in a healed, caught-up cluster"
    for read, record in reads.items():
        collect(read, record)
    assert all(node.pending_reads == 0 for node in cluster.nodes.values())
    assert served > 0


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
        await disturb(cluster, rng, members)
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

    async def raising(max_entries=None, **options):
        raise RuntimeError("broken state machine")

    monkeypatch.setattr(cluster.nodes[leader].durable, "apply_committed", raising)
    with pytest.raises(AssertionError, match="stopped on"):
        await cluster.append_command(leader, OpenSession().encode())
