"""Tier 2 component tests for DurableNodeState (DD-8, DD-19, DD-22): every
state change persisted before its method returns, memory never ahead of
disk, the persisted log always equal to the in-memory one across all six
Figure 7 scenarios, and — using a store that holds a write in flight until
the test releases it — no second decision taken while a first write is
still pending.
"""

import asyncio

import pytest

from raftkv.consensus import FollowerProgress, IllegalTransition, LogEntry, Role
from raftkv.persistence import DurableNodeState, SqliteStore
from tests.figure_7 import FOLLOWER_TERMS, LEADER_TERMS, make_log

NODE_ID = 7


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "node.db")


async def reload(path):
    """Reopen the file from scratch, as a restarted node would, and load it."""
    async with SqliteStore(path) as store:
        return await store.load()


class RecordingStore(SqliteStore):
    """A real SqliteStore that also records every write it is asked to make."""

    def __init__(self, path):
        super().__init__(path)
        self.writes = []

    async def save_term_and_vote(self, current_term, voted_for):
        self.writes.append(("term_and_vote", current_term, voted_for))
        await super().save_term_and_vote(current_term, voted_for)

    async def save_log_from(self, index, entries):
        self.writes.append(("log_from", index, [e.term for e in entries]))
        await super().save_log_from(index, entries)


class FailingStore(SqliteStore):
    """A store whose every write fails, as a full or broken disk would."""

    async def save_term_and_vote(self, current_term, voted_for):
        raise OSError("disk full")

    async def save_log_from(self, index, entries):
        raise OSError("disk full")


class GatedStore(SqliteStore):
    """A store that holds each term/vote write in flight until `release` is set.

    `entered` is set as soon as a write arrives, so a test knows exactly
    when one is pending instead of guessing with sleeps.
    """

    def __init__(self, path):
        super().__init__(path)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.writes = []

    async def save_term_and_vote(self, current_term, voted_for):
        self.writes.append((current_term, voted_for))
        self.entered.set()
        await self.release.wait()
        await super().save_term_and_vote(current_term, voted_for)


async def seed_log(path, terms):
    async with SqliteStore(path) as store:
        await store.save_log_from(1, list(make_log(terms)))


async def reconcile(durable, leader_log):
    """Drive REPL-6/REPL-7's retry loop against a durable follower."""
    progress = FollowerProgress(next_index=leader_log.last_index + 1)
    while True:
        prev_log_index = progress.next_index - 1
        prev_log_term = leader_log[prev_log_index - 1].term if prev_log_index > 0 else 0
        entries = leader_log[prev_log_index:]
        if await durable.receive_entries(prev_log_index, prev_log_term, entries):
            return
        progress.record_rejection()


# --- Start-up (PERSIST-4/5/6, STATE-2) --------------------------------------


async def test_load_on_a_fresh_store_is_a_brand_new_follower(db_path):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store)
    assert durable.node_id == NODE_ID
    assert durable.role is Role.FOLLOWER
    assert (durable.current_term, durable.voted_for) == (0, None)
    assert len(durable.log) == 0


async def test_restart_after_becoming_candidate_comes_back_as_follower(db_path):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store)
        await durable.become_candidate()
    async with SqliteStore(db_path) as store:
        restarted = await DurableNodeState.load(NODE_ID, store)
    assert restarted.role is Role.FOLLOWER
    assert (restarted.current_term, restarted.voted_for) == (1, NODE_ID)


async def test_restart_reloads_the_log(db_path):
    await seed_log(db_path, [1, 1, 2])
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store)
    assert durable.log == make_log([1, 1, 2])


# --- Term and vote are persisted before each method returns ----------------


async def test_become_candidate_persists_term_and_self_vote(db_path):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store)
        await durable.become_candidate()
        assert (durable.current_term, durable.voted_for) == (1, NODE_ID)
    persisted = await reload(db_path)
    assert (persisted.current_term, persisted.voted_for) == (1, NODE_ID)


async def test_handle_observed_term_writes_only_when_it_fires(db_path):
    async with RecordingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store)
        assert await durable.handle_observed_term(0) is False
        assert store.writes == []
        assert await durable.handle_observed_term(3) is True
        assert store.writes == [("term_and_vote", 3, None)]
    persisted = await reload(db_path)
    assert (persisted.current_term, persisted.voted_for) == (3, None)


async def test_leader_observing_higher_term_steps_down_and_persists(db_path):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store)
        await durable.become_candidate()
        await durable.become_leader()
        assert await durable.handle_observed_term(5) is True
        assert durable.role is Role.FOLLOWER
    persisted = await reload(db_path)
    assert (persisted.current_term, persisted.voted_for) == (5, None)


async def test_become_leader_writes_nothing(db_path):
    async with RecordingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store)
        await durable.become_candidate()
        writes_before = list(store.writes)
        await durable.become_leader()
        assert durable.role is Role.LEADER
        assert store.writes == writes_before


async def test_illegal_transition_changes_nothing_in_memory_or_on_disk(db_path):
    async with RecordingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store)
        with pytest.raises(IllegalTransition):
            await durable.become_leader()
        assert durable.role is Role.FOLLOWER
        assert store.writes == []


# --- DD-22: memory is never ahead of disk -----------------------------------


async def test_failed_write_leaves_term_vote_and_role_unchanged(db_path):
    async with FailingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store)
        with pytest.raises(OSError):
            await durable.become_candidate()
        assert durable.role is Role.FOLLOWER
        assert (durable.current_term, durable.voted_for) == (0, None)
        with pytest.raises(OSError):
            await durable.handle_observed_term(4)
        assert durable.current_term == 0


async def test_failed_write_leaves_the_log_unchanged(db_path):
    await seed_log(db_path, [1, 1])
    async with FailingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store)
        with pytest.raises(OSError):
            await durable.receive_entries(2, 1, [LogEntry(term=2, command="x")])
        assert durable.log == make_log([1, 1])


# --- The log: REPL-5 rejection, and disk always equal to memory -------------


async def test_rejected_entries_write_nothing(db_path):
    async with RecordingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store)
        accepted = await durable.receive_entries(3, 1, [LogEntry(term=1, command="x")])
        assert accepted is False
        assert store.writes == []


async def test_only_the_changed_suffix_is_rewritten(db_path):
    await seed_log(db_path, [1, 1, 2, 2])
    async with RecordingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store)
        await durable.receive_entries(2, 1, [LogEntry(term=3, command="new")])
        assert store.writes == [("log_from", 3, [3])]


async def test_heartbeat_that_changes_nothing_writes_nothing(db_path):
    await seed_log(db_path, [1, 1])
    async with RecordingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store)
        assert await durable.receive_entries(2, 1, []) is True
        assert store.writes == []


@pytest.mark.parametrize("label", ["a", "b", "e", "f"])
async def test_figure_7_repair_is_persisted_exactly(db_path, label):
    await seed_log(db_path, FOLLOWER_TERMS[label])
    leader_log = make_log(LEADER_TERMS)
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store)
        await reconcile(durable, leader_log)
        assert durable.log == leader_log
    assert (await reload(db_path)).log == leader_log


@pytest.mark.parametrize("label", ["c", "d"])
async def test_figure_7_stale_tail_survives_a_heartbeat_on_disk_too(db_path, label):
    await seed_log(db_path, FOLLOWER_TERMS[label])
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store)
        assert await durable.receive_entries(10, 6, []) is True
    assert (await reload(db_path)).log == make_log(FOLLOWER_TERMS[label])

    # Once the leader writes a genuinely conflicting entry 11, the whole
    # stale tail goes — in memory and on disk alike.
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store)
        new_entry = LogEntry(term=8, command="new-write")
        assert await durable.receive_entries(10, 6, [new_entry]) is True
        in_memory = durable.log
    persisted = (await reload(db_path)).log
    assert persisted == in_memory
    assert [e.term for e in persisted] == LEADER_TERMS + [8]


# --- DD-19: no second decision while a first write is in flight -------------


async def test_second_decision_waits_while_first_write_is_in_flight(db_path):
    async with GatedStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store)

        first = asyncio.create_task(durable.become_candidate())
        await store.entered.wait()  # the first write is now held in flight

        second = asyncio.create_task(durable.handle_observed_term(99))
        for _ in range(10):
            await asyncio.sleep(0)  # give the second task every chance to run

        try:
            # The second decision has not reached the store, and the first
            # one has not been installed before its own write completes.
            assert store.writes == [(1, NODE_ID)]
            assert durable.current_term == 0
        finally:
            store.release.set()
            await asyncio.gather(first, second)

        # Once released, the second decision runs against the first one's
        # result: Candidate in term 1 observes term 99 and steps down.
        assert store.writes == [(1, NODE_ID), (99, None)]
        assert (durable.current_term, durable.voted_for) == (99, None)
        assert durable.role is Role.FOLLOWER

    persisted = await reload(db_path)
    assert (persisted.current_term, persisted.voted_for) == (99, None)


async def test_vote_cast_in_flight_is_not_forgotten_by_a_concurrent_observation(db_path):
    # The second task observes term 1 — the very term the first task is
    # becoming Candidate in. Evaluated against the state before the first
    # write, term 1 would look new and clear the vote; the node would
    # forget it already voted for itself in term 1 and could vote again
    # in that term.
    async with GatedStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store)

        first = asyncio.create_task(durable.become_candidate())
        await store.entered.wait()

        second = asyncio.create_task(durable.handle_observed_term(1))
        for _ in range(10):
            await asyncio.sleep(0)

        try:
            assert store.writes == [(1, NODE_ID)]
        finally:
            store.release.set()
            results = await asyncio.gather(first, second)

        assert results == [None, False]  # term 1 was not higher than term 1
        assert store.writes == [(1, NODE_ID)]
        assert (durable.current_term, durable.voted_for) == (1, NODE_ID)
        assert durable.role is Role.CANDIDATE

    persisted = await reload(db_path)
    assert (persisted.current_term, persisted.voted_for) == (1, NODE_ID)
