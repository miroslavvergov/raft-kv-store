"""Tier 2 tests for DurableNodeState: each change is on disk before it is installed or returned.

DD-8, DD-19, DD-22. Memory is never ahead of disk, and no second decision is taken while a first
write is in flight.
"""

import asyncio
from dataclasses import dataclass

import pytest

from raftkv.consensus import Cluster, IllegalTransitionError, LogEntry, Role
from raftkv.node import DurableNodeState
from raftkv.storage import SqliteStore
from tests.support.append_entries_messages import append_entries
from tests.support.divergent_logs import make_log
from tests.support.store_doubles import (
    CommitThenHoldStore,
    FailingStore,
    GatedStore,
    RecordingStore,
    let_other_tasks_run,
    reload,
    seed_log,
    seed_term_and_vote,
    term_and_vote_on_disk,
    win_election,
)
from tests.support.vote_messages import granted

NODE_ID = 7  # IDs from 7 up never look like the small terms and indexes these tests use.
THREE_NODES = Cluster([7, 8, 9])


# --- Start-up (PERSIST-4, PERSIST-5, PERSIST-6, STATE-2) ------------------------------


async def test_load_on_a_fresh_store_is_a_brand_new_follower(db_path):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
    assert durable.node_id == NODE_ID
    assert durable.role is Role.FOLLOWER
    assert (durable.current_term, durable.voted_for) == (0, None)
    assert len(durable.log) == 0


async def test_restart_after_becoming_candidate_comes_back_as_follower(db_path):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await durable.start_election()
    async with SqliteStore(db_path) as store:
        restarted = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
    assert restarted.role is Role.FOLLOWER
    assert (restarted.current_term, restarted.voted_for) == (1, NODE_ID)


async def test_restart_reloads_the_log(db_path):
    await seed_log(db_path, [1, 1, 2])
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
    assert durable.log == make_log([1, 1, 2])


# --- Term and vote are persisted before each method returns ---------------------------


async def test_start_election_persists_term_and_self_vote(db_path):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await durable.start_election()
        assert (durable.current_term, durable.voted_for) == (1, NODE_ID)
    assert await term_and_vote_on_disk(db_path) == (1, NODE_ID)


async def test_handle_observed_term_writes_only_when_it_fires(db_path):
    async with RecordingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        assert await durable.handle_observed_term(0) is False
        assert store.writes == []
        assert await durable.handle_observed_term(3) is True
        assert store.writes == [("term_and_vote", 3, None)]
    assert await term_and_vote_on_disk(db_path) == (3, None)


async def test_leader_observing_higher_term_steps_down_and_persists(db_path):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await win_election(durable)
        assert await durable.handle_observed_term(5) is True
        assert durable.role is Role.FOLLOWER
        assert durable.leadership is None
    assert await term_and_vote_on_disk(db_path) == (5, None)


async def test_becoming_leader_writes_only_its_empty_entry(db_path):
    # Role is not persisted; the empty entry a new Leader appends in its term is.
    async with RecordingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        request = await durable.start_election()
        writes_before = list(store.writes)
        response = granted(term=request.term)
        assert await durable.handle_vote_response(8, request.term, response) is True
        assert durable.role is Role.LEADER
        assert store.writes == [*writes_before, ("replace_log_from", 1, [1])]


async def test_illegal_transition_changes_nothing_in_memory_or_on_disk(db_path):
    # Leader -> Candidate is not a STATE-3 edge.
    async with RecordingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await win_election(durable)
        writes_before = list(store.writes)
        with pytest.raises(IllegalTransitionError):
            await durable.start_election()
        assert (durable.role, durable.current_term, durable.voted_for) == (
            Role.LEADER,
            1,
            NODE_ID,
        )
        assert store.writes == writes_before


# --- Memory is never ahead of disk (DD-22) --------------------------------------------


async def test_failed_write_leaves_term_vote_and_role_unchanged(db_path):
    async with FailingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        with pytest.raises(OSError):
            await durable.start_election()
        assert durable.role is Role.FOLLOWER
        assert (durable.current_term, durable.voted_for) == (0, None)
        with pytest.raises(OSError):
            await durable.handle_observed_term(4)
        assert durable.current_term == 0


async def test_cancelled_caller_still_installs_what_it_persisted(db_path):
    # A write can complete on the store's background thread after its task is cancelled. The
    # change must still be installed, or disk would hold a vote the node does not know it cast.
    async with GatedStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        caller = asyncio.create_task(durable.start_election())
        await store.wait_for_write()

        caller.cancel()
        await let_other_tasks_run()
        assert not caller.done()  # still waiting for its write, lock still held

        store.release.set()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert (durable.current_term, durable.voted_for) == (1, NODE_ID)
        assert durable.role is Role.CANDIDATE

    assert await term_and_vote_on_disk(db_path) == (1, NODE_ID)


async def test_a_write_task_cancelled_after_its_commit_is_still_installed(db_path):
    # The write itself is cancelled, not its caller, after the commit but before it reports
    # back. The store is read back, finds the new term and vote, and installs them.
    async with CommitThenHoldStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        caller = asyncio.create_task(durable.start_election())
        await store.wait_for_commit()

        store.write_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert (durable.role, durable.current_term, durable.voted_for) == (
            Role.CANDIDATE,
            1,
            NODE_ID,
        )

    assert await term_and_vote_on_disk(db_path) == (1, NODE_ID)


async def test_a_write_task_cancelled_before_its_commit_installs_nothing(db_path):
    async with GatedStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        caller = asyncio.create_task(durable.start_election())
        await store.wait_for_write()

        store.write_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert (durable.role, durable.current_term, durable.voted_for) == (Role.FOLLOWER, 0, None)
        assert durable.candidacy is None

    assert await term_and_vote_on_disk(db_path) == (0, None)


async def test_cancelled_log_write_still_installs_the_new_log(db_path):
    # Seeded at the Leader's term, so this call's only write is the log's.
    await seed_log(db_path, [1, 1])
    await seed_term_and_vote(db_path, 2)
    entry = LogEntry(term=2, command="x")
    async with GatedStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        caller = asyncio.create_task(
            durable.handle_append_entries(
                append_entries(term=2, prev_log_index=2, prev_log_term=1, entries=[entry])
            )
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
    await seed_term_and_vote(db_path, 2)
    entry = LogEntry(term=2, command="x")
    async with FailingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        with pytest.raises(OSError):
            await durable.handle_append_entries(
                append_entries(term=2, prev_log_index=2, prev_log_term=1, entries=[entry])
            )
        assert durable.log == make_log([1, 1])


# --- No second decision while a first write is in flight (DD-19) ----------------------


@dataclass
class RaceOutcome:
    """What racing handle_observed_term against a held start_election write produced.

    Attributes:
        writes_while_held: The writes the store received before the first was released.
        term_while_held: The node's current_term before the release.
        results: What start_election and handle_observed_term returned.
        final: The node's (current_term, voted_for, role) at the end.
        persisted: The (current_term, voted_for) on disk at the end.
    """

    writes_while_held: list
    term_while_held: int
    results: list
    final: tuple
    persisted: tuple


async def race_candidacy_against_observed_term(db_path, observed_term):
    """Hold start_election's write in flight, then start handle_observed_term(observed_term).

    The second task gets ten event-loop turns before the release; no real time passes, so the
    outcome does not depend on timing.
    """
    async with GatedStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)

        first = asyncio.create_task(durable.start_election())
        await store.wait_for_write()
        second = asyncio.create_task(durable.handle_observed_term(observed_term))
        await let_other_tasks_run()

        writes_while_held = list(store.writes)
        term_while_held = durable.current_term
        store.release.set()
        results = await asyncio.gather(first, second)
        final = (durable.current_term, durable.voted_for, durable.role)

    return RaceOutcome(
        writes_while_held,
        term_while_held,
        results,
        final,
        await term_and_vote_on_disk(db_path),
    )


async def test_second_decision_waits_while_first_write_is_in_flight(db_path):
    outcome = await race_candidacy_against_observed_term(db_path, observed_term=99)
    # While the first write was held, the second decision never reached the store, and the
    # first was not installed before its own write finished.
    assert outcome.writes_while_held == [("term_and_vote", 1, NODE_ID)]
    assert outcome.term_while_held == 0
    # Once released, the second ran against the first's result: a term-1 Candidate observing
    # term 99 steps down.
    assert outcome.final == (99, None, Role.FOLLOWER)
    assert outcome.persisted == (99, None)


async def test_vote_cast_in_flight_is_not_forgotten_by_a_concurrent_observation(db_path):
    # The second task observes term 1, the very term the first is becoming Candidate in.
    # Decided against the state before the first write, term 1 would look new and clear the
    # vote, and the node could vote again in term 1.
    outcome = await race_candidacy_against_observed_term(db_path, observed_term=1)
    assert outcome.writes_while_held == [("term_and_vote", 1, NODE_ID)]
    assert outcome.results[1] is False  # term 1 is not higher than term 1
    assert outcome.final == (1, NODE_ID, Role.CANDIDATE)
    assert outcome.persisted == (1, NODE_ID)


@pytest.mark.negative_control
async def test_negative_control_without_the_lock_the_vote_is_lost(db_path, without_the_lock):
    # Proves the race tests above can fail: without DD-8's lock, the second decision reaches
    # the store while the first write is held, and the persisted term-1 self-vote is erased.
    outcome = await race_candidacy_against_observed_term(db_path, observed_term=1)
    assert outcome.writes_while_held == [
        ("term_and_vote", 1, NODE_ID),
        ("term_and_vote", 1, None),
    ]
    assert outcome.persisted == (1, None)
