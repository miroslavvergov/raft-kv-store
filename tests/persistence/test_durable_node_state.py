"""Tier 2 component tests for DurableNodeState (DD-8, DD-19, DD-22): every
state change persisted before its method returns, memory never ahead of
disk, the persisted log always equal to the in-memory one while six
diverged followers are repaired, and — using a store that holds a write in
flight until the test releases it — no second decision taken while a first
write is still pending.
"""

import asyncio
from dataclasses import dataclass

import pytest

from raftkv.consensus import FollowerProgress, IllegalTransition, LogEntry, Role
from raftkv.persistence import DurableNodeState, SqliteStore
from tests.divergent_logs import FOLLOWER_TERMS, LEADER_TERMS, make_log

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
    """A store that holds each write in flight until `release` is set.

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

    async def save_log_from(self, index, entries):
        self.writes.append(("log_from", index))
        self.entered.set()
        await self.release.wait()
        await super().save_log_from(index, entries)


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


async def test_cancelled_caller_still_installs_what_it_persisted(db_path):
    # A write can complete on the store's background thread after the task
    # that issued it is cancelled. The change must still be installed, or
    # disk would hold a vote the node does not know it cast.
    async with GatedStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store)
        caller = asyncio.create_task(durable.become_candidate())
        await store.entered.wait()

        caller.cancel()
        for _ in range(10):
            await asyncio.sleep(0)
        assert not caller.done()  # still waiting for its write, lock still held

        store.release.set()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert (durable.current_term, durable.voted_for) == (1, NODE_ID)
        assert durable.role is Role.CANDIDATE

    persisted = await reload(db_path)
    assert (persisted.current_term, persisted.voted_for) == (1, NODE_ID)


async def test_cancelled_log_write_still_installs_the_new_log(db_path):
    await seed_log(db_path, [1, 1])
    async with GatedStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store)
        caller = asyncio.create_task(
            durable.receive_entries(2, 1, [LogEntry(term=2, command="x")])
        )
        await store.entered.wait()

        caller.cancel()
        store.release.set()
        with pytest.raises(asyncio.CancelledError):
            await caller
        in_memory = durable.log

    assert [e.term for e in in_memory] == [1, 1, 2]
    assert (await reload(db_path)).log == in_memory


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


@pytest.mark.parametrize(
    "follower",
    [
        "missing_last_entry",
        "missing_last_six_entries",
        "conflicts_from_index_6",
        "conflicts_from_index_4",
    ],
)
async def test_repaired_log_is_persisted_exactly(db_path, follower):
    await seed_log(db_path, FOLLOWER_TERMS[follower])
    leader_log = make_log(LEADER_TERMS)
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store)
        await reconcile(durable, leader_log)
        assert durable.log == leader_log
    assert (await reload(db_path)).log == leader_log


@pytest.mark.parametrize("follower", ["one_extra_stale_entry", "two_extra_stale_entries"])
async def test_stale_extra_entries_survive_a_heartbeat_on_disk_too(db_path, follower):
    await seed_log(db_path, FOLLOWER_TERMS[follower])
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store)
        assert await durable.receive_entries(10, 6, []) is True
    assert (await reload(db_path)).log == make_log(FOLLOWER_TERMS[follower])

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


@dataclass
class RaceOutcome:
    """What a race between become_candidate and handle_observed_term produced."""

    writes_while_held: list
    term_while_held: int
    results: list
    final: tuple
    persisted: tuple


class NoLock:
    """Stands in for DD-8's lock in the negative control: never blocks anyone."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


async def race_candidacy_against_observed_term(db_path, observed_term):
    """Hold become_candidate's write in flight, then start handle_observed_term.

    The first write is held by GatedStore until released, and the second
    task is given ten event-loop turns to run before the release — no real
    time passes, so the outcome does not depend on timing.
    """
    async with GatedStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store)

        first = asyncio.create_task(durable.become_candidate())
        await store.entered.wait()
        second = asyncio.create_task(durable.handle_observed_term(observed_term))
        for _ in range(10):
            await asyncio.sleep(0)

        writes_while_held = list(store.writes)
        term_while_held = durable.current_term
        store.release.set()
        results = await asyncio.gather(first, second)
        final = (durable.current_term, durable.voted_for, durable.role)

    persisted = await reload(db_path)
    return RaceOutcome(
        writes_while_held,
        term_while_held,
        results,
        final,
        (persisted.current_term, persisted.voted_for),
    )


async def test_second_decision_waits_while_first_write_is_in_flight(db_path):
    outcome = await race_candidacy_against_observed_term(db_path, observed_term=99)
    # While the first write was held, the second decision never reached the
    # store, and the first was not installed before its own write finished.
    assert outcome.writes_while_held == [(1, NODE_ID)]
    assert outcome.term_while_held == 0
    # Once released, the second ran against the first one's result: a
    # Candidate in term 1 observing term 99 steps down.
    assert outcome.final == (99, None, Role.FOLLOWER)
    assert outcome.persisted == (99, None)


async def test_vote_cast_in_flight_is_not_forgotten_by_a_concurrent_observation(db_path):
    # The second task observes term 1 — the very term the first is becoming
    # Candidate in. Against the state before the first write, term 1 would
    # look new and clear the vote, and the node could vote again in term 1.
    outcome = await race_candidacy_against_observed_term(db_path, observed_term=1)
    assert outcome.writes_while_held == [(1, NODE_ID)]
    assert outcome.results == [None, False]  # term 1 is not higher than term 1
    assert outcome.final == (1, NODE_ID, Role.CANDIDATE)
    assert outcome.persisted == (1, NODE_ID)


async def test_negative_control_without_the_lock_the_vote_is_lost(db_path, monkeypatch):
    # Confirms the race tests above can fail: with DD-8's lock replaced by a
    # no-op, the same scenario lets the second decision reach the store while
    # the first is held, and the node's persisted term-1 self-vote is erased.
    original_init = DurableNodeState.__init__

    def init_without_lock(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._lock = NoLock()

    monkeypatch.setattr(DurableNodeState, "__init__", init_without_lock)

    outcome = await race_candidacy_against_observed_term(db_path, observed_term=1)
    assert outcome.writes_while_held == [(1, NODE_ID), (1, None)]
    assert outcome.persisted == (1, None)
