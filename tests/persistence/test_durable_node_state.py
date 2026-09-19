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

from raftkv.consensus import (
    Cluster,
    FollowerProgress,
    IllegalTransition,
    LogEntry,
    RequestVoteResponse,
    Role,
)
from raftkv.persistence import DurableNodeState, SqliteStore
from tests.divergent_logs import FOLLOWER_TERMS, LEADER_TERMS, make_log
from tests.persistence.store_doubles import (
    FailingStore,
    GatedStore,
    NoLock,
    RecordingStore,
    reload,
    seed_log,
)

NODE_ID = 7
CLUSTER = Cluster([7, 8, 9])


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "node.db")


async def win_election(durable):
    """Start an election and have peer 8 grant its vote: 2 of 3 is a majority."""
    request = await durable.start_election()
    granted = RequestVoteResponse(term=request.term, vote_granted=True)
    assert await durable.handle_vote_response(8, request.term, granted) is True
    return request


async def reconcile(durable, leader_log):
    """Drive REPL-6/REPL-7's retry loop against a durable follower.

    A probe at index 1 always passes, so needing more rejections than the
    leader has entries fails the test instead of looping forever.
    """
    progress = FollowerProgress(next_index=leader_log.last_index + 1)
    rejections = 0
    while True:
        prev_log_index = progress.next_index - 1
        prev_log_term = leader_log[prev_log_index - 1].term if prev_log_index > 0 else 0
        entries = leader_log[prev_log_index:]
        if await durable.receive_entries(prev_log_index, prev_log_term, entries):
            return
        progress.record_rejection()
        rejections += 1
        assert rejections <= leader_log.last_index, "never reached an index where the logs agree"


# --- Start-up (PERSIST-4/5/6, STATE-2) --------------------------------------


async def test_load_on_a_fresh_store_is_a_brand_new_follower(db_path):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, CLUSTER)
    assert durable.node_id == NODE_ID
    assert durable.role is Role.FOLLOWER
    assert (durable.current_term, durable.voted_for) == (0, None)
    assert len(durable.log) == 0


async def test_restart_after_becoming_candidate_comes_back_as_follower(db_path):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, CLUSTER)
        await durable.start_election()
    async with SqliteStore(db_path) as store:
        restarted = await DurableNodeState.load(NODE_ID, store, CLUSTER)
    assert restarted.role is Role.FOLLOWER
    assert (restarted.current_term, restarted.voted_for) == (1, NODE_ID)


async def test_restart_reloads_the_log(db_path):
    await seed_log(db_path, [1, 1, 2])
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, CLUSTER)
    assert durable.log == make_log([1, 1, 2])


# --- Term and vote are persisted before each method returns ----------------


async def test_start_election_persists_term_and_self_vote(db_path):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, CLUSTER)
        await durable.start_election()
        assert (durable.current_term, durable.voted_for) == (1, NODE_ID)
    persisted = await reload(db_path)
    assert (persisted.current_term, persisted.voted_for) == (1, NODE_ID)


async def test_handle_observed_term_writes_only_when_it_fires(db_path):
    async with RecordingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, CLUSTER)
        assert await durable.handle_observed_term(0) is False
        assert store.writes == []
        assert await durable.handle_observed_term(3) is True
        assert store.writes == [("term_and_vote", 3, None)]
    persisted = await reload(db_path)
    assert (persisted.current_term, persisted.voted_for) == (3, None)


async def test_leader_observing_higher_term_steps_down_and_persists(db_path):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, CLUSTER)
        await win_election(durable)
        assert await durable.handle_observed_term(5) is True
        assert durable.role is Role.FOLLOWER
        assert durable.leadership is None
    persisted = await reload(db_path)
    assert (persisted.current_term, persisted.voted_for) == (5, None)


async def test_becoming_leader_writes_nothing(db_path):
    async with RecordingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, CLUSTER)
        request = await durable.start_election()
        writes_before = list(store.writes)
        granted = RequestVoteResponse(term=request.term, vote_granted=True)
        assert await durable.handle_vote_response(8, request.term, granted) is True
        assert durable.role is Role.LEADER
        assert store.writes == writes_before


async def test_illegal_transition_changes_nothing_in_memory_or_on_disk(db_path):
    # Leader -> Candidate is not a STATE-3 edge.
    async with RecordingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, CLUSTER)
        await win_election(durable)
        writes_before = list(store.writes)
        with pytest.raises(IllegalTransition):
            await durable.start_election()
        assert (durable.role, durable.current_term, durable.voted_for) == (
            Role.LEADER,
            1,
            NODE_ID,
        )
        assert store.writes == writes_before


# --- DD-22: memory is never ahead of disk -----------------------------------


async def test_failed_write_leaves_term_vote_and_role_unchanged(db_path):
    async with FailingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, CLUSTER)
        with pytest.raises(OSError):
            await durable.start_election()
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
        durable = await DurableNodeState.load(NODE_ID, store, CLUSTER)
        caller = asyncio.create_task(durable.start_election())
        await store.wait_for_write()

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
        durable = await DurableNodeState.load(NODE_ID, store, CLUSTER)
        caller = asyncio.create_task(
            durable.receive_entries(2, 1, [LogEntry(term=2, command="x")])
        )
        await store.wait_for_write()

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
        durable = await DurableNodeState.load(NODE_ID, store, CLUSTER)
        with pytest.raises(OSError):
            await durable.receive_entries(2, 1, [LogEntry(term=2, command="x")])
        assert durable.log == make_log([1, 1])


# --- The log: REPL-5 rejection, and disk always equal to memory -------------


async def test_rejected_entries_write_nothing(db_path):
    async with RecordingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, CLUSTER)
        accepted = await durable.receive_entries(3, 1, [LogEntry(term=1, command="x")])
        assert accepted is False
        assert store.writes == []


async def test_only_the_changed_suffix_is_rewritten(db_path):
    await seed_log(db_path, [1, 1, 2, 2])
    async with RecordingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, CLUSTER)
        await durable.receive_entries(2, 1, [LogEntry(term=3, command="new")])
        assert store.writes == [("log_from", 3, [3])]


async def test_heartbeat_that_changes_nothing_writes_nothing(db_path):
    await seed_log(db_path, [1, 1])
    async with RecordingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, CLUSTER)
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
        durable = await DurableNodeState.load(NODE_ID, store, CLUSTER)
        await reconcile(durable, leader_log)
        assert durable.log == leader_log
    assert (await reload(db_path)).log == leader_log


@pytest.mark.parametrize("follower", ["one_extra_stale_entry", "two_extra_stale_entries"])
async def test_stale_extra_entries_survive_a_heartbeat_on_disk_too(db_path, follower):
    await seed_log(db_path, FOLLOWER_TERMS[follower])
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, CLUSTER)
        assert await durable.receive_entries(10, 6, []) is True
    assert (await reload(db_path)).log == make_log(FOLLOWER_TERMS[follower])

    # Once the leader writes a genuinely conflicting entry 11, the whole
    # stale tail goes — in memory and on disk alike.
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, CLUSTER)
        new_entry = LogEntry(term=8, command="new-write")
        assert await durable.receive_entries(10, 6, [new_entry]) is True
        in_memory = durable.log
    persisted = (await reload(db_path)).log
    assert persisted == in_memory
    assert [e.term for e in persisted] == LEADER_TERMS + [8]


# --- DD-19: no second decision while a first write is in flight -------------


@dataclass
class RaceOutcome:
    """What a race between start_election and handle_observed_term produced."""

    writes_while_held: list
    term_while_held: int
    results: list
    final: tuple
    persisted: tuple


async def race_candidacy_against_observed_term(db_path, observed_term):
    """Hold start_election's write in flight, then start handle_observed_term.

    The first write is held by GatedStore until released, and the second
    task is given ten event-loop turns to run before the release — no real
    time passes, so the outcome does not depend on timing.
    """
    async with GatedStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, CLUSTER)

        first = asyncio.create_task(durable.start_election())
        await store.wait_for_write()
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
    assert outcome.results[1] is False  # term 1 is not higher than term 1
    assert outcome.final == (1, NODE_ID, Role.CANDIDATE)
    assert outcome.persisted == (1, NODE_ID)


@pytest.mark.negative_control
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
