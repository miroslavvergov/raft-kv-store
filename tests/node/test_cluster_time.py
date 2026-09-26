"""Tier 2 tests for cluster time in DurableNodeState: a Leader's clock, stamped on its entries.

DD-32: cluster time counts a Leader's ticks, resumes from the cluster time of the log's last entry,
and reaches every other node, and the disk, only inside the entries it is stamped on.
"""

import pytest

from raftkv.consensus import Cluster, LogEntry, NotLeaderError, Role
from raftkv.node import DurableNodeState
from raftkv.storage import SqliteStore
from tests.support.append_entries_messages import append_entries, heartbeat
from tests.support.store_doubles import reload, win_election

NODE_ID = 7
THREE_NODES = Cluster([7, 8, 9])


async def seed_entries(path, *cluster_times, term=1):
    """Write a log of one entry per cluster time, all of `term`, and that term, to a file."""
    async with SqliteStore(path) as store:
        entries = [LogEntry(term, f"c{i}", time) for i, time in enumerate(cluster_times, 1)]
        await store.replace_log_from(1, entries)
        await store.save_term_and_vote(term, None)


def advance(node, ticks):
    for _ in range(ticks):
        node.advance_cluster_time()


async def test_a_new_leader_stamps_its_empty_entry_with_its_logs_last_cluster_time(db_path):
    await seed_entries(db_path, 5, 40)
    async with SqliteStore(db_path) as store:
        leader = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await win_election(leader)

        assert leader.log.entry_at(3) == LogEntry.empty(term=2, cluster_time=40)
        assert leader.leadership.cluster_time == 40


async def test_each_leader_tick_is_counted_in_the_time_on_the_next_command(db_path):
    async with SqliteStore(db_path) as store:
        leader = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await win_election(leader)
        advance(leader, 3)
        await leader.append_command("a")
        advance(leader, 2)
        await leader.append_command("b")

        assert [entry.cluster_time for entry in leader.log] == [0, 3, 5]
    assert [entry.cluster_time for entry in (await reload(db_path)).log] == [0, 3, 5]


async def test_ticks_count_only_while_leader(db_path):
    async with SqliteStore(db_path) as store:
        node = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        advance(node, 5)  # as Follower
        await node.start_election()
        assert node.role is Role.CANDIDATE
        advance(node, 5)  # as Candidate

        await win_election(node)

        assert node.log.entry_at(1).cluster_time == 0
        assert node.leadership.cluster_time == 0


async def test_a_followers_ticks_give_it_no_clock_and_no_way_to_append(db_path):
    async with SqliteStore(db_path) as store:
        node = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await node.handle_append_entries(heartbeat(term=3, leader=8))
        advance(node, 5)

        assert node.leadership is None
        with pytest.raises(NotLeaderError):
            await node.append_command("a")


async def test_a_new_leadership_resumes_from_the_log_not_from_an_earlier_leaderships_clock(
    db_path,
):
    # Ticks a Leader counted but stamped on no entry are not kept: whoever leads next resumes
    # from the last time in its log. Sessions can only seem younger for it, never older.
    async with SqliteStore(db_path) as store:
        node = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await win_election(node)  # its empty entry, at cluster time 0
        advance(node, 10)
        await node.handle_append_entries(
            heartbeat(term=5, leader=8, prev_log_index=1, prev_log_term=1)
        )
        assert node.leadership is None

        await win_election(node)

        assert node.log.entry_at(2) == LogEntry.empty(term=6, cluster_time=0)


async def test_a_follower_keeps_the_cluster_times_it_is_sent_across_a_restart(db_path):
    async with SqliteStore(db_path) as store:
        follower = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await follower.handle_append_entries(
            append_entries(term=3, leader=8, entries=[LogEntry(3, "a", 7), LogEntry(3, "b", 9)])
        )

    async with SqliteStore(db_path) as store:
        restarted = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        assert [entry.cluster_time for entry in restarted.log] == [7, 9]
