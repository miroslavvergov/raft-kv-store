"""Tier 2 replication across whole clusters of real nodes, driven step by step by the test.

A Leader appends a command, replicates it, and commits it once a majority holds it; Followers
learn the commit from the next AppendEntries. After every step the harness checks that no
committed entry was ever changed or lost, and at teardown the trace checker re-checks the same
from the recorded trace alone. REPL-1 through REPL-8, REPL-13 through REPL-17, APPLY-1 through
APPLY-3, FAIL-1, FAIL-2.
"""

import random

import pytest

from raftkv.consensus import CommittedEntryConflictError, Log, LogEntry, LogPosition, Role
from tests.cluster.in_process_cluster import AppendEntriesInFlight


def commands(cluster, node_id):
    """Return a node's log as its commands, the empty entry shown as ''."""
    return [entry.command for entry in cluster.nodes[node_id].log]


def commit_indexes(cluster):
    """Return every running node's commit index."""
    return {n: node.commit_index for n, node in cluster.nodes.items()}


# --- A command reaches every node and is committed -------------------------------------


async def test_a_command_is_replicated_to_every_node_and_committed(three_nodes):
    position = await three_nodes.append_command(1, "x=5")
    assert position.index == 2  # after node 1's empty entry

    await three_nodes.replicate_to_all(1)

    assert commands(three_nodes, 2) == commands(three_nodes, 3) == ["", "x=5"]
    # The Leader committed as soon as node 2 confirmed. Node 2's request was built before
    # that, so it carried commit index 0; node 3's, built after, carried 2.
    assert commit_indexes(three_nodes) == {1: 2, 2: 0, 3: 2}

    await three_nodes.replicate_to_all(1)  # the next round, heartbeats, carries the commit
    assert commit_indexes(three_nodes) == {1: 2, 2: 2, 3: 2}


async def test_commands_are_committed_on_every_node_in_the_order_they_were_appended(three_nodes):
    for command in ("a=1", "b=2", "c=3"):
        await three_nodes.append_command(1, command)
    await three_nodes.replicate_to_all(1)
    await three_nodes.replicate_to_all(1)

    for node_id in (1, 2, 3):
        assert commands(three_nodes, node_id) == ["", "a=1", "b=2", "c=3"]
    assert set(commit_indexes(three_nodes).values()) == {4}


async def test_a_single_node_cluster_commits_each_command_at_once(start_cluster):
    cluster = await start_cluster([1])
    await cluster.fire_election_timeout(1)
    assert (await cluster.append_command(1, "x=5")).index == 2
    assert cluster.nodes[1].commit_index == 2


async def test_five_nodes_need_three_copies_to_commit(start_cluster):
    cluster = await start_cluster([1, 2, 3, 4, 5])
    await cluster.run_election(1)
    await cluster.append_command(1, "x=5")

    await cluster.replicate(1, 2)
    assert cluster.nodes[1].commit_index == 0  # 2 of 5
    await cluster.replicate(1, 3)
    assert cluster.nodes[1].commit_index == 2  # 3 of 5


# --- Committed entries survive crashes and a change of Leader --------------------------


async def test_a_committed_entry_survives_a_change_of_leader(three_nodes):
    await three_nodes.append_command(1, "x=5")
    await three_nodes.replicate_to_all(1)
    await three_nodes.stop(1)

    await three_nodes.run_election(2)

    assert three_nodes.leaders() == {2}
    # Node 2 kept x=5 and appended its own term-2 empty entry after it.
    assert three_nodes.nodes[2].log == Log(
        [LogEntry.empty(1), LogEntry(1, "x=5"), LogEntry.empty(2)]
    )
    await three_nodes.replicate(2, 3)
    assert three_nodes.nodes[2].commit_index == 3  # x=5 commits again, with node 2's entry


async def test_a_majority_commits_while_a_follower_is_down_and_it_catches_up_later(three_nodes):
    await three_nodes.stop(3)
    await three_nodes.append_command(1, "x=5")
    await three_nodes.replicate_to_all(1)
    assert three_nodes.nodes[1].commit_index == 2  # nodes 1 and 2

    await three_nodes.start(3)
    assert len(three_nodes.nodes[3].log) == 0  # stopped before it received anything
    await three_nodes.replicate(1, 3)
    assert commands(three_nodes, 3) == ["", "x=5"]


async def test_a_restarted_follower_relearns_the_commit_index_from_the_next_heartbeat(three_nodes):
    await three_nodes.append_command(1, "x=5")
    await three_nodes.replicate_to_all(1)
    await three_nodes.replicate_to_all(1)
    assert three_nodes.nodes[2].commit_index == 2

    await three_nodes.restart(2)
    assert three_nodes.nodes[2].commit_index == 0  # not persisted
    await three_nodes.replicate(1, 2)
    assert three_nodes.nodes[2].commit_index == 2


async def test_a_follower_with_conflicting_entries_is_walked_back_and_overwritten(start_cluster):
    # Nodes 1 and 2 hold entries of terms 1, 1, 3; node 3 holds entries of terms 1, 2, 2 from a
    # term-2 Leader that never committed them. Node 1 wins term 4 and appends its empty entry.
    cluster = await start_cluster(
        [1, 2, 3],
        preload={
            1: {"log_terms": [1, 1, 3], "current_term": 3},
            2: {"log_terms": [1, 1, 3], "current_term": 3},
            3: {"log_terms": [1, 2, 2], "current_term": 3},
        },
    )
    await cluster.run_election(1)
    assert cluster.leaders() == {1}

    rejections = await cluster.replicate(1, 3)

    # Rejected after index 3 (node 3's is from term 2, not 3) and after index 2 (term 2, not
    # 1); accepted after index 1, where both hold the same term-1 entry.
    assert rejections == 2
    assert cluster.nodes[3].log == cluster.nodes[1].log
    assert cluster.log_terms()[3] == [1, 1, 3, 4]


# --- Messages lost, repeated, and reordered (FAIL-1, FAIL-2) ---------------------------


async def test_a_lost_answer_leads_to_a_resend_that_changes_nothing(three_nodes):
    await three_nodes.append_command(1, "x=5")
    message = await three_nodes.send_append_entries(1, 2)
    await three_nodes.deliver(message)  # node 2 stores the entries; its answer goes in flight
    [answer] = [m for m in three_nodes.in_flight if m.response is not None]
    three_nodes.drop(answer)
    assert three_nodes.nodes[1].commit_index == 0  # the Leader never heard back

    await three_nodes.replicate(1, 2)  # it sends the same entries again

    assert commands(three_nodes, 2) == ["", "x=5"]  # stored once, not twice
    assert three_nodes.nodes[1].commit_index == 2


async def test_a_duplicated_old_append_entries_arriving_late_deletes_nothing(three_nodes):
    # A copy of the first AppendEntries stays in flight and arrives after node 2 has already
    # stored and committed a later entry. The copy's entries all match, so it deletes nothing.
    await three_nodes.append_command(1, "a=1")
    first = await three_nodes.send_append_entries(1, 2)
    await three_nodes.deliver(first, keep_copy=True)
    await three_nodes.append_command(1, "b=2")
    await three_nodes.replicate(1, 2)
    await three_nodes.replicate(1, 2)
    assert commands(three_nodes, 2) == ["", "a=1", "b=2"]
    assert three_nodes.nodes[2].commit_index == 3

    await three_nodes.deliver(first)  # the late copy: after index 0, entries 1 and 2

    assert commands(three_nodes, 2) == ["", "a=1", "b=2"]
    assert three_nodes.nodes[2].commit_index == 3


async def test_a_broken_merge_rule_is_stopped_before_a_committed_entry_is_lost(
    three_nodes, monkeypatch
):
    # Node 2 has committed entry 3 when a late copy of an earlier AppendEntries arrives. Under
    # blind truncation that copy would delete entry 3; the node refuses it by name instead.
    await three_nodes.append_command(1, "a=1")
    first = await three_nodes.send_append_entries(1, 2)
    await three_nodes.deliver(first, keep_copy=True)
    await three_nodes.append_command(1, "b=2")
    await three_nodes.replicate(1, 2)
    await three_nodes.replicate(1, 2)

    def blind_truncate(log, prev_log_index, entries):
        return Log(list(log)[:prev_log_index] + list(entries))

    monkeypatch.setattr(Log, "after_append_entries", blind_truncate)
    with pytest.raises(CommittedEntryConflictError):
        await three_nodes.deliver(first)
    assert commands(three_nodes, 2) == ["", "a=1", "b=2"]


@pytest.mark.negative_control
async def test_negative_control_a_node_committing_past_its_own_log_is_caught(three_nodes):
    # Proves the harness's "no node commits past the end of its log" check can fire. Real code
    # can no longer reach this state, so it is forged by reaching into _commit_index.
    await three_nodes.append_command(1, "a=1")
    await three_nodes.replicate(1, 2)
    node = three_nodes.nodes[2]
    node._commit_index = node.log.last_index + 1

    with pytest.raises(AssertionError, match="has commit index 3 but holds only 2"):
        three_nodes.assert_log_safety()


async def test_answers_arriving_out_of_order_never_lower_progress(three_nodes):
    early = await three_nodes.send_append_entries(1, 2)  # carries the empty entry
    await three_nodes.append_command(1, "x=5")
    await three_nodes.deliver(early)
    # Built before the first answer arrives, so it starts at index 0 too, one entry longer.
    later = await three_nodes.send_append_entries(1, 2)
    await three_nodes.deliver(later)
    answers = [m for m in three_nodes.in_flight if m.response is not None]
    assert [len(m.request.entries) for m in answers] == [1, 2]

    await three_nodes.deliver(answers[1])  # the later answer first
    await three_nodes.deliver(answers[0])

    assert three_nodes.nodes[1].leadership.match_index(2) == 2  # not lowered to 1
    assert three_nodes.nodes[1].commit_index == 2


async def test_a_duplicated_answer_is_counted_once(three_nodes):
    await three_nodes.append_command(1, "x=5")
    message = await three_nodes.send_append_entries(1, 2)
    await three_nodes.deliver(message)
    [answer] = [m for m in three_nodes.in_flight if isinstance(m, AppendEntriesInFlight)]
    await three_nodes.deliver(answer, keep_copy=True)
    await three_nodes.deliver(answer)
    assert three_nodes.nodes[1].leadership.match_index(2) == 2
    assert three_nodes.nodes[1].commit_index == 2


# --- A deposed Leader --------------------------------------------------------------------


async def test_a_deposed_leader_is_rejected_steps_down_and_loses_its_uncommitted_entry(
    three_nodes,
):
    # Node 1 is cut off. Nodes 2 and 3 elect node 2 in term 2. Node 1, still believing it
    # leads term 1, appends a command no one else will ever hold.
    await three_nodes.run_election(2, reachable={3})
    assert three_nodes.nodes[2].role is Role.LEADER
    await three_nodes.append_command(1, "stale")

    # Its AppendEntries reaches node 3, which rejects it with term 2, ending its leadership. A
    # rejection carrying a later term asks for no resend, so the loop stops at once.
    assert await three_nodes.replicate(1, 3) == 0
    assert (three_nodes.nodes[1].role, three_nodes.nodes[1].current_term) == (Role.FOLLOWER, 2)

    # Node 2 then repairs node 1. Neither of node 1's term-1 entries was ever replicated, let
    # alone committed, so both give way to node 2's term-2 empty entry and command.
    await three_nodes.append_command(2, "fresh")
    await three_nodes.replicate(2, 1)
    assert commands(three_nodes, 1) == commands(three_nodes, 2) == ["", "fresh"]
    assert three_nodes.log_terms()[1] == [2, 2]


@pytest.mark.negative_control
async def test_negative_control_a_leader_elected_without_a_committed_entry_fails_log_safety(
    three_nodes, monkeypatch
):
    # Proves the harness's check that a new Leader holds every committed entry can fire.
    # Nodes 1 and 2 commit x=5; node 3 never receives it. With the up-to-date rule removed,
    # node 3 can win term 2 without it, which is what that check exists to catch. Node 3 never
    # commits the index itself, so only the Leader loop can see it.
    await three_nodes.append_command(1, "x=5")
    await three_nodes.replicate(1, 2)
    assert three_nodes.nodes[1].commit_index == 2
    assert len(three_nodes.nodes[3].log) == 0

    monkeypatch.setattr(LogPosition, "is_at_least_as_up_to_date_as", lambda self, other: True)
    with pytest.raises(AssertionError, match="leader 3 of term 2 holds"):
        await three_nodes.run_election(3)


# --- Randomized schedules ----------------------------------------------------------------

TIMEOUT = 0.08
RESTART = 0.04
COMMAND = 0.12
SEND = 0.30


async def take_one_random_step(cluster, rng):
    """Take one random step: a timeout, a restart, a command, a send, or a message's fate."""
    roll = rng.random()
    leaders = sorted(cluster.leaders())
    if roll < TIMEOUT:
        ready = [n for n, node in cluster.nodes.items() if node.role is not Role.LEADER]
        if ready:
            cluster.send_requests(await cluster.fire_election_timeout(rng.choice(ready)))
    elif roll < TIMEOUT + RESTART:
        await cluster.restart(rng.choice(cluster.member_ids))
    elif roll < TIMEOUT + RESTART + COMMAND and leaders:
        await cluster.append_command(rng.choice(leaders), f"cmd-{rng.randrange(10**6)}")
    elif roll < TIMEOUT + RESTART + COMMAND + SEND and leaders:
        leader = rng.choice(leaders)
        await cluster.send_append_entries(leader, rng.choice(sorted(cluster.nodes[leader].peers)))
    elif cluster.in_flight:
        message = rng.choice(cluster.in_flight)
        fate = rng.random()
        if fate < 0.15:
            cluster.drop(message)
        else:
            await cluster.deliver(message, keep_copy=fate < 0.3)


async def settle(cluster):
    """Empty the network, elect one Leader, and replicate until every node matches it.

    A node that timed out into a later term during the random steps rejects the new Leader,
    which steps down on the answer; the next election then runs in that later term. So this
    elects and replicates until one Leader keeps its role and every log and commit index match.

    Returns:
        The Leader.
    """
    cluster.in_flight.clear()
    for _ in range(5):
        if not cluster.leaders():
            by_log = sorted(
                cluster.member_ids,
                key=lambda n: (cluster.nodes[n].log.last_term, cluster.nodes[n].log.last_index),
                reverse=True,
            )
            # NOTE: the first candidate is tried again at the end: its own failed attempt may
            # have raised every peer's term to one no earlier candidate could win in.
            for candidate in by_log + by_log[:1]:
                await cluster.run_election(candidate)
                if cluster.leaders():
                    break
        [leader] = cluster.leaders()
        await cluster.replicate_to_all(leader)
        await cluster.replicate_to_all(leader)
        lead = cluster.nodes[leader]
        if lead.role is Role.LEADER and all(
            node.log == lead.log and node.commit_index == lead.commit_index
            for node in cluster.nodes.values()
        ):
            return leader
    raise AssertionError(f"the cluster never settled: {cluster.log_terms()}")


@pytest.mark.parametrize(
    ("members", "seed"),
    [([1, 2, 3], seed) for seed in range(30)] + [([1, 2, 3, 4, 5], seed) for seed in range(10)],
    ids=lambda value: f"n={len(value)}" if isinstance(value, list) else f"seed={value}",
)
async def test_random_schedules_never_lose_a_committed_entry(start_cluster, members, seed):
    # Elections, crashes, commands, and AppendEntries at any moment, with every message
    # delivered late, out of order, twice, or never. After every step no committed entry may
    # have changed; at the end one Leader brings every node into line with it. Five nodes need
    # three copies to commit, and a new Leader's quorum may exclude every node that had them.
    rng = random.Random(seed)
    cluster = await start_cluster(members)

    for _ in range(200):
        await take_one_random_step(cluster, rng)
        cluster.assert_election_safety()
        cluster.assert_log_safety()

    leader = await settle(cluster)
    leader_log = cluster.nodes[leader].log
    for node in cluster.nodes.values():
        assert node.log == leader_log
        assert node.commit_index == cluster.nodes[leader].commit_index
    # Everything committed during the random steps is still in the final, agreed log.
    for index, (entry, _) in cluster.committed.items():
        assert leader_log.entry_at(index) == entry
