"""Tier 2 tests for DurableNodeState.handle_append_entries: the Follower's side of AppendEntries.

The term decision and the log change happen in one lock hold, and whatever they change is on
disk before the answer exists. REPL-5, REPL-8, REPL-13, STATE-4, STATE-5, STATE-6, STATE-7,
PERSIST-1, PERSIST-2, PERSIST-3, DD-7, DD-8.
"""

import asyncio

import pytest

from raftkv.consensus import Cluster, CommittedEntryConflictError, Log, LogEntry, Role
from raftkv.node import DurableNodeState
from raftkv.storage import SqliteStore
from tests.support.append_entries_messages import append_entries, append_entries_at, heartbeat
from tests.support.divergent_logs import FOLLOWER_TERMS, LEADER_TERMS, make_log
from tests.support.store_doubles import (
    FailingStore,
    GatedStore,
    RecordingStore,
    let_other_tasks_run,
    reload,
    seed_log,
    term_and_vote_on_disk,
    win_election,
)

NODE_ID = 7
THREE_NODES = Cluster([7, 8, 9])


async def follower(store, *, at_term=0):
    """Load the node under test, already caught up to `at_term` if one is given.

    Catching up first leaves the term out of what a later AppendEntries writes, so a test can
    assert on the log's write alone.
    """
    durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
    if at_term:
        await durable.handle_observed_term(at_term)
    return durable


# --- Step 1: the decision on the term -------------------------------------------------


async def test_an_outdated_leader_is_rejected_and_told_the_newer_term(db_path):
    async with RecordingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await durable.handle_observed_term(7)
        store.writes.clear()

        answer = await durable.handle_append_entries(append_entries(term=3, entries=[]))

        assert answer.success is False
        assert answer.term == 7  # the stale Leader learns it has been deposed
        assert store.writes == []  # a refusal changes nothing, so nothing is written
        assert durable.current_term == 7


async def test_a_follower_catches_up_to_a_higher_term_and_persists_it(db_path):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        answer = await durable.handle_append_entries(heartbeat(term=5))
        assert (answer.term, answer.success) == (5, True)
        assert (durable.current_term, durable.voted_for) == (5, None)
    assert await term_and_vote_on_disk(db_path) == (5, None)


async def test_a_candidate_steps_down_for_a_leader_of_its_own_term_keeping_its_self_vote(db_path):
    # STATE-7: no term rises, so nothing tells the node it is behind except the Leader's
    # existence. Its self-vote stays, or it could vote a second time in the same term.
    async with RecordingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await durable.start_election()
        assert (durable.role, durable.current_term) == (Role.CANDIDATE, 1)
        store.writes.clear()

        answer = await durable.handle_append_entries(heartbeat(term=1))

        assert answer.success is True
        assert durable.role is Role.FOLLOWER
        assert (durable.current_term, durable.voted_for) == (1, NODE_ID)
        assert durable.candidacy is None
        # Role is not persisted, and neither term nor vote changed, so there is nothing to write.
        assert store.writes == []


async def test_a_leader_ignores_an_append_entries_from_its_own_term(db_path):
    # Two Leaders cannot hold one term, so this can only be a stale or forged message. Accepting
    # it would let another node rewrite this Leader's log under its own term.
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await win_election(durable)

        answer = await durable.handle_append_entries(heartbeat(term=durable.current_term))

        assert answer.success is False
        assert durable.role is Role.LEADER
        assert durable.leadership is not None


async def test_a_leader_steps_down_for_a_higher_term(db_path):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await win_election(durable)

        answer = await durable.handle_append_entries(heartbeat(term=durable.current_term + 4))

        assert answer.success is True
        assert durable.role is Role.FOLLOWER
        assert durable.leadership is None
        assert durable.voted_for is None
    assert await term_and_vote_on_disk(db_path) == (5, None)


# --- Step 2: the log consistency check (REPL-5) ---------------------------------------


@pytest.mark.parametrize(
    ("prev_log_index", "prev_log_term", "why"),
    [
        (2, 4, "entry 2 is there, but from term 1, not term 4"),
        (3, 1, "just past the last entry"),
        (5, 4, "far past the last entry"),
    ],
)
async def test_entries_are_rejected_unless_the_previous_entry_matches(
    db_path, prev_log_index, prev_log_term, why
):
    await seed_log(db_path, [1, 1])
    async with RecordingStore(db_path) as store:
        durable = await follower(store, at_term=4)
        store.writes.clear()

        answer = await durable.handle_append_entries(
            append_entries(
                term=4,
                prev_log_index=prev_log_index,
                prev_log_term=prev_log_term,
                entries=[LogEntry(4, "x")],
            )
        )

        assert answer.success is False, why
        assert store.writes == []
        assert durable.log == make_log([1, 1])


async def test_a_rejection_still_keeps_the_term_the_rpc_brought(db_path):
    # The log check fails, but the node has still seen a newer term and must not forget it:
    # forgetting would let it vote again in a term it has already moved past.
    await seed_log(db_path, [1, 1])
    async with RecordingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)

        answer = await durable.handle_append_entries(
            append_entries(term=6, prev_log_index=9, prev_log_term=6, entries=[LogEntry(6, "x")])
        )

        assert (answer.term, answer.success) == (6, False)
        assert store.writes == [("term_and_vote", 6, None)]
        assert durable.log == make_log([1, 1])
    assert await term_and_vote_on_disk(db_path) == (6, None)


# --- Step 3: the entries and the commit index (REPL-8, REPL-13) -----------------------


async def test_a_term_catch_up_and_the_entries_are_one_transaction(db_path):
    # A crash between two separate writes would leave entries stored under a term the node
    # never recorded, or the term without them. One transaction has no such gap (DD-7).
    await seed_log(db_path, [1, 1])
    async with RecordingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)

        answer = await durable.handle_append_entries(
            append_entries(term=4, prev_log_index=2, prev_log_term=1, entries=[LogEntry(4, "x")])
        )

        assert answer.success is True
        assert store.writes == [("term_vote_and_log", 4, None, 3, [4])]
    persisted = await reload(db_path)
    assert (persisted.current_term, persisted.voted_for) == (4, None)
    assert [e.term for e in persisted.log] == [1, 1, 4]


async def test_a_conflicting_entry_rewrites_only_the_suffix_and_drops_the_tail_on_disk(db_path):
    # Entries 3 and 4 are from term 2; the Leader's entry 3 is from term 4. Entry 3 conflicts,
    # so it and entry 4 after it go, and only the new entry 3 is written.
    await seed_log(db_path, [1, 1, 2, 2])
    async with RecordingStore(db_path) as store:
        durable = await follower(store, at_term=4)
        store.writes.clear()

        await durable.handle_append_entries(
            append_entries(term=4, prev_log_index=2, prev_log_term=1, entries=[LogEntry(4, "x")])
        )

        # Term already current, so only the log is written, from the first conflict on.
        assert store.writes == [("replace_log_from", 3, [4])]
    assert [e.term for e in (await reload(db_path)).log] == [1, 1, 4]


@pytest.mark.parametrize(
    ("leader_commit", "expected"),
    [
        (0, 0),
        (2, 2),  # within what this RPC covers
        (3, 3),  # exactly the last entry carried
        (7, 3),  # beyond it: the rest has not arrived yet
    ],
)
async def test_commit_index_follows_the_leader_capped_by_what_arrived(
    db_path, leader_commit, expected
):
    await seed_log(db_path, [1, 1])
    async with SqliteStore(db_path) as store:
        durable = await follower(store, at_term=4)
        await durable.handle_append_entries(
            append_entries(
                term=4,
                prev_log_index=2,
                prev_log_term=1,
                entries=[LogEntry(4, "x")],
                leader_commit=leader_commit,
            )
        )
        assert durable.commit_index == expected


async def test_a_rejected_rpc_commits_nothing(db_path):
    await seed_log(db_path, [1, 1])
    async with SqliteStore(db_path) as store:
        durable = await follower(store, at_term=4)
        answer = await durable.handle_append_entries(
            heartbeat(term=4, prev_log_index=9, prev_log_term=4, commit=9)
        )
        assert answer.success is False
        assert durable.commit_index == 0


async def test_a_delayed_rpc_never_lowers_the_commit_index(db_path):
    await seed_log(db_path, [1, 1, 1])
    async with SqliteStore(db_path) as store:
        durable = await follower(store, at_term=4)
        await durable.handle_append_entries(
            heartbeat(term=4, prev_log_index=3, prev_log_term=1, commit=3)
        )
        assert durable.commit_index == 3

        # An older heartbeat, delayed in the network, covers only index 1.
        await durable.handle_append_entries(
            heartbeat(term=4, prev_log_index=1, prev_log_term=1, commit=3)
        )
        assert durable.commit_index == 3


async def test_a_failed_write_commits_nothing(db_path):
    # The entries never reached the disk, so nothing they contain may count as committed.
    await seed_log(db_path, [1, 1])
    async with FailingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        with pytest.raises(OSError):
            await durable.handle_append_entries(
                append_entries(
                    term=4,
                    prev_log_index=2,
                    prev_log_term=1,
                    entries=[LogEntry(4, "x")],
                    leader_commit=3,
                )
            )
        assert durable.commit_index == 0
        assert durable.current_term == 0
        assert durable.log == make_log([1, 1])


async def test_a_heartbeat_that_changes_nothing_writes_nothing_but_still_commits(db_path):
    await seed_log(db_path, [1, 1])
    async with RecordingStore(db_path) as store:
        durable = await follower(store, at_term=4)
        store.writes.clear()

        answer = await durable.handle_append_entries(
            heartbeat(term=4, prev_log_index=2, prev_log_term=1, commit=2)
        )

        assert answer.success is True
        assert store.writes == []
        assert durable.commit_index == 2


@pytest.mark.parametrize(
    "request_",
    [
        heartbeat(term=4, prev_log_index=50, prev_log_term=1, commit=50),
        append_entries(term=4, prev_log_index=50, prev_log_term=1, entries=[LogEntry(4, "x")]),
    ],
    ids=["heartbeat", "one new entry"],
)
async def test_an_rpc_compares_only_the_entries_it_carried_not_the_whole_log(
    db_path, monkeypatch, request_
):
    # A heartbeat arrives many times a second: comparing every held entry each time would
    # make its cost grow with the log.
    await seed_log(db_path, [1] * 50)
    async with SqliteStore(db_path) as store:
        durable = await follower(store, at_term=4)
        compared = []
        entry_equality = LogEntry.__eq__
        monkeypatch.setattr(
            LogEntry, "__eq__", lambda a, b: compared.append(a) or entry_equality(a, b)
        )

        answer = await durable.handle_append_entries(request_)

    assert answer.success is True
    assert compared == []


async def test_commitment_is_forgotten_across_a_restart_and_relearned(db_path):
    await seed_log(db_path, [1, 1])
    async with SqliteStore(db_path) as store:
        durable = await follower(store, at_term=4)
        await durable.handle_append_entries(
            heartbeat(term=4, prev_log_index=2, prev_log_term=1, commit=2)
        )
        assert durable.commit_index == 2

    async with SqliteStore(db_path) as store:
        restarted = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        # Commitment is not persisted: the entries are still there, only the knowledge that a
        # majority holds them is gone, and the next heartbeat brings it back.
        assert restarted.commit_index == 0
        assert restarted.log == make_log([1, 1])
        await restarted.handle_append_entries(
            heartbeat(term=4, prev_log_index=2, prev_log_term=1, commit=2)
        )
        assert restarted.commit_index == 2


# --- A late RPC never deletes what a later one appended (REPL-8, FAIL-2) --------------


FIRST_THREE = [LogEntry(4, "cmd1"), LogEntry(4, "cmd2"), LogEntry(4, "cmd3")]


async def accept_three_then_replay_an_earlier_rpc(durable):
    """Accept entries 1-3 with all three committed, then deliver an earlier, delayed RPC.

    The delayed RPC carried only entry 2 and was sent when nothing was committed yet, so it
    covers neither entry 3 nor the commit that followed.

    Returns:
        The answer to the delayed RPC.
    """
    await durable.handle_append_entries(
        append_entries(term=4, entries=FIRST_THREE, leader_commit=3)
    )
    assert durable.commit_index == 3
    late = append_entries(term=4, prev_log_index=1, prev_log_term=4, entries=FIRST_THREE[1:2])
    return await durable.handle_append_entries(late)


async def test_a_late_rpc_repeating_a_held_entry_keeps_the_committed_entry_after_it(db_path):
    async with RecordingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        answer = await accept_three_then_replay_an_earlier_rpc(durable)

        # Entry 2 matches, so nothing is deleted: entry 3 stays in memory, and the replay
        # writes nothing, so it stays on disk too.
        assert answer.success is True
        assert durable.log == Log(FIRST_THREE)
        assert durable.commit_index == 3
        assert store.writes == [("term_vote_and_log", 4, None, 1, [4, 4, 4])]  # the first RPC's
    assert (await reload(db_path)).log == Log(FIRST_THREE)


async def test_a_broken_merge_rule_is_stopped_before_it_loses_a_committed_entry(
    db_path, monkeypatch
):
    # With the merge rule replaced by blind truncation, the late RPC would delete committed
    # entry 3. The guard in handle_append_entries refuses it by name, with nothing changed in
    # memory or on disk, rather than letting the loss happen silently.
    def blind_truncate(log, prev_log_index, entries):
        return Log(list(log)[:prev_log_index] + list(entries))

    monkeypatch.setattr(Log, "after_append_entries", blind_truncate)
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        with pytest.raises(CommittedEntryConflictError, match="would change entry 3"):
            await accept_three_then_replay_an_earlier_rpc(durable)

        assert durable.log == Log(FIRST_THREE)
        assert durable.commit_index == 3
    assert (await reload(db_path)).log == Log(FIRST_THREE)


# --- Committed entries are never overwritten --------------------------------------------


async def test_an_append_entries_that_would_change_a_committed_entry_is_refused(db_path):
    # Only a broken election or commit rule could send this: a Leader of term 5 whose entry 2
    # differs from one this node has already committed. Nothing is installed, not even term 5.
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await durable.handle_append_entries(
            append_entries(term=4, entries=FIRST_THREE, leader_commit=3)
        )
        rewrite = append_entries(
            term=5, prev_log_index=1, prev_log_term=4, entries=[LogEntry(5, "other")]
        )

        with pytest.raises(CommittedEntryConflictError, match="committed through 3"):
            await durable.handle_append_entries(rewrite)

        assert (durable.log, durable.current_term) == (Log(FIRST_THREE), 4)
    persisted = await reload(db_path)
    assert (persisted.log, persisted.current_term) == (Log(FIRST_THREE), 4)


async def test_an_uncommitted_entry_just_past_the_commit_index_may_still_be_replaced(db_path):
    # The boundary: entries 1 and 2 are committed, entry 3 is not, so a new Leader may still
    # overwrite entry 3.
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await durable.handle_append_entries(
            append_entries(term=4, entries=FIRST_THREE, leader_commit=2)
        )
        replacement = append_entries(
            term=5, prev_log_index=2, prev_log_term=4, entries=[LogEntry(5, "other")]
        )

        assert (await durable.handle_append_entries(replacement)).success is True

        assert [e.command for e in durable.log] == ["cmd1", "cmd2", "other"]


# --- Repairing a divergent Follower through the real handler --------------------------


async def repair_through_the_handler(durable, leader_log, *, commit=0):
    """Send `durable` the Leader's entries, backing off one index per rejection, until accepted.

    A probe with prev_log_index 0 always succeeds, so more rejections than the Leader has
    entries fails the test instead of looping forever.

    Returns:
        How many AppendEntries the Follower rejected before accepting.
    """
    next_index = leader_log.last_index + 1
    rejections = 0
    while True:
        request = append_entries_at(leader_log, next_index, commit=commit)
        if (await durable.handle_append_entries(request)).success:
            return rejections
        next_index -= 1
        rejections += 1
        assert rejections <= leader_log.last_index, "never reached an index where the logs agree"


@pytest.mark.parametrize(
    "follower_name",
    [
        "missing_last_entry",
        "missing_last_six_entries",
        "conflicts_from_index_6",
        "conflicts_from_index_4",
    ],
)
async def test_any_divergent_follower_is_repaired_exactly_and_persisted(db_path, follower_name):
    await seed_log(db_path, FOLLOWER_TERMS[follower_name])
    leader_log = make_log(LEADER_TERMS)
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await repair_through_the_handler(durable, leader_log, commit=len(LEADER_TERMS))

        assert durable.log == leader_log
        assert durable.commit_index == leader_log.last_index
        # The Leader's term arrived with the entries and was persisted with them.
        assert durable.current_term == 8
    assert (await reload(db_path)).log == leader_log


@pytest.mark.parametrize("follower_name", ["one_extra_stale_entry", "two_extra_stale_entries"])
async def test_a_longer_follower_keeps_its_stale_tail_until_an_entry_conflicts(
    db_path, follower_name
):
    # The Leader's first probe matches at its own last index, so the Follower accepts an empty
    # AppendEntries and its extra entries survive: nothing has contradicted them yet. They are
    # uncommitted, so keeping them is safe, and commitment still stops at what the Leader holds.
    await seed_log(db_path, FOLLOWER_TERMS[follower_name])
    leader_log = make_log(LEADER_TERMS)
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        assert await repair_through_the_handler(durable, leader_log, commit=10) == 0

        assert [e.term for e in durable.log] == FOLLOWER_TERMS[follower_name]
        assert durable.commit_index == leader_log.last_index
        assert [e.term for e in (await reload(db_path)).log] == FOLLOWER_TERMS[follower_name]

        # A conflicting entry 11 from the Leader finally removes the whole stale tail.
        answer = await durable.handle_append_entries(
            append_entries(
                term=8, prev_log_index=10, prev_log_term=6, entries=[LogEntry(8, "new-write")]
            )
        )
        assert answer.success is True
        assert [e.term for e in durable.log] == LEADER_TERMS + [8]
    assert [e.term for e in (await reload(db_path)).log] == LEADER_TERMS + [8]


async def test_repairing_a_follower_that_is_already_up_to_date_rejects_nothing(db_path):
    await seed_log(db_path, LEADER_TERMS)
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        assert await repair_through_the_handler(durable, make_log(LEADER_TERMS)) == 0


# --- One lock hold: no term can slip in between the check and the write (DD-8) --------


async def test_a_higher_term_cannot_arrive_between_the_term_check_and_the_log_write(db_path):
    # The deposed Leader's AppendEntries holds the write. A newer Leader's term arrives while it
    # is in flight. The lock makes the second wait, so it decides against the installed log.
    await seed_log(db_path, [1, 1])
    async with GatedStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)

        deposed = asyncio.create_task(
            durable.handle_append_entries(
                append_entries(
                    term=4, prev_log_index=2, prev_log_term=1, entries=[LogEntry(4, "old")]
                )
            )
        )
        await store.wait_for_write()
        newer = asyncio.create_task(durable.handle_observed_term(9))
        await let_other_tasks_run()

        # While the first write is held, the second decision has not reached the store.
        assert store.writes == [("term_vote_and_log", 4, None, 3, [4])]
        assert durable.current_term == 0

        store.release.set()
        first, _ = await asyncio.gather(deposed, newer)

    assert first.success is True
    assert durable.current_term == 9
    assert [e.term for e in durable.log] == [1, 1, 4]


async def test_a_cancelled_caller_still_installs_both_the_term_and_the_entries(db_path):
    await seed_log(db_path, [1, 1])
    async with GatedStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        caller = asyncio.create_task(
            durable.handle_append_entries(
                append_entries(
                    term=4,
                    prev_log_index=2,
                    prev_log_term=1,
                    entries=[LogEntry(4, "x")],
                    leader_commit=3,
                )
            )
        )
        await store.wait_for_write()

        caller.cancel()
        store.release.set()
        with pytest.raises(asyncio.CancelledError):
            await caller

        # What the write persisted is installed, since the disk already holds it.
        assert durable.current_term == 4
        assert [e.term for e in durable.log] == [1, 1, 4]
        # No answer was returned, so the Leader will retry and commitment waits for that.
        assert durable.commit_index == 0

    persisted = await reload(db_path)
    assert (persisted.current_term, [e.term for e in persisted.log]) == (4, [1, 1, 4])


@pytest.mark.negative_control
async def test_negative_control_without_the_lock_the_log_is_written_under_a_stale_term(
    db_path, without_the_lock
):
    # Proves the lock test above can fail: with DD-8's lock removed, the second decision reaches
    # the store while the first write is still in flight.
    await seed_log(db_path, [1, 1])
    async with GatedStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        deposed = asyncio.create_task(
            durable.handle_append_entries(
                append_entries(
                    term=4, prev_log_index=2, prev_log_term=1, entries=[LogEntry(4, "old")]
                )
            )
        )
        await store.wait_for_write()
        newer = asyncio.create_task(durable.handle_observed_term(9))
        await let_other_tasks_run()

        assert store.writes == [
            ("term_vote_and_log", 4, None, 3, [4]),
            ("term_and_vote", 9, None),
        ]

        store.release.set()
        await asyncio.gather(deposed, newer)


# --- The answer's term is the node's own, so a Leader always learns the truth ---------


@pytest.mark.parametrize(
    ("own_term", "rpc_term", "expected_term", "expected_success"),
    [
        (0, 1, 1, True),  # a brand-new node accepts and reports the Leader's term
        (5, 9, 9, True),  # caught up
        (5, 5, 5, True),  # same term
        (9, 5, 9, False),  # the Leader is behind and is told so
    ],
)
async def test_the_answer_always_carries_the_nodes_own_term(
    db_path, own_term, rpc_term, expected_term, expected_success
):
    async with SqliteStore(db_path) as store:
        durable = await follower(store, at_term=own_term)
        answer = await durable.handle_append_entries(heartbeat(term=rpc_term))
        assert (answer.term, answer.success) == (expected_term, expected_success)
        assert durable.current_term == expected_term
