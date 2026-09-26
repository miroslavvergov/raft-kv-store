"""Tier 2 tests for DurableNodeState.apply_committed: what reaches the state machine, and when.

APPLY-4 (strictly in log order), APPLY-5 (never past what is committed), DD-12 (the KV Store
layer is reached only through the callback), DD-26 (a Leader's empty entry is not a command),
DD-28 (applying is volatile, synchronous, and rebuilt by replay), CLIENT-10 (a Leader must have
committed an entry of its own term before it may answer a read).
"""

import pytest

from raftkv.consensus import Cluster, LogEntry, Role
from raftkv.kvstore import KeyValueStore, OpenSession, Put
from raftkv.node import DurableNodeState
from raftkv.storage import SqliteStore
from tests.support.append_entries_messages import accepted, append_entries, heartbeat
from tests.support.store_doubles import seed_term_and_vote, win_election

NODE_ID = 7
THREE_NODES = Cluster([7, 8, 9])
ONE_NODE = Cluster([NODE_ID])


class RecordingStateMachine:
    """A state machine double that remembers the commands it was given, in order.

    Attributes:
        applied: Every command applied, in order.
        calls: Every call, as (index, cluster_time, command).
    """

    def __init__(self, fail_on=None):
        self.applied = []
        self.calls = []
        self.fail_on = fail_on

    def __call__(self, index, cluster_time, command):
        if command == self.fail_on:
            raise RuntimeError("this state machine refuses that command")
        self.applied.append(command)
        self.calls.append((index, cluster_time, command))


async def follower_with(store, recorder):
    """Load node 7 as a Follower whose state machine is `recorder`."""
    return await DurableNodeState.load(NODE_ID, store, THREE_NODES, apply=recorder)


def command_entries(*commands, term=4):
    """Return one entry per command, all of `term`."""
    return [LogEntry(term=term, command=command) for command in commands]


# --- What reaches the state machine, and in what order --------------------------------


async def test_committed_commands_are_applied_in_log_order(db_path):
    recorder = RecordingStateMachine()
    async with SqliteStore(db_path) as store:
        durable = await follower_with(store, recorder)
        await durable.handle_append_entries(
            append_entries(term=4, entries=command_entries("a", "b", "c"), leader_commit=3)
        )

        assert await durable.apply_committed() == 3

        assert recorder.applied == ["a", "b", "c"]
        assert durable.last_applied == 3


async def test_each_command_is_handed_over_with_its_index_and_cluster_time(db_path):
    # The KV Store layer names a new session after its index, and judges expiry by the time
    # on the entry, which every replica reads the same (DD-32).
    recorder = RecordingStateMachine()
    entries = [LogEntry(4, "a", 3), LogEntry(4, "b", 3), LogEntry(4, "c", 7)]
    async with SqliteStore(db_path) as store:
        durable = await follower_with(store, recorder)
        await durable.handle_append_entries(
            append_entries(term=4, entries=entries, leader_commit=3)
        )

        await durable.apply_committed()

        assert recorder.calls == [(1, 3, "a"), (2, 3, "b"), (3, 7, "c")]


async def test_nothing_past_the_commit_index_is_applied(db_path):
    recorder = RecordingStateMachine()
    async with SqliteStore(db_path) as store:
        durable = await follower_with(store, recorder)
        # Three entries arrive, but the Leader has only committed the first two.
        await durable.handle_append_entries(
            append_entries(term=4, entries=command_entries("a", "b", "c"), leader_commit=2)
        )

        assert await durable.apply_committed() == 2

        assert recorder.applied == ["a", "b"]
        assert (durable.last_applied, durable.commit_index) == (2, 2)


async def test_a_limited_call_applies_at_most_that_many_entries(db_path):
    recorder = RecordingStateMachine()
    async with SqliteStore(db_path) as store:
        durable = await follower_with(store, recorder)
        await durable.handle_append_entries(
            append_entries(term=4, entries=command_entries("a", "b", "c"), leader_commit=3)
        )

        assert await durable.apply_committed(max_entries=2) == 2
        assert recorder.applied == ["a", "b"]
        assert await durable.apply_committed(max_entries=2) == 1
        assert recorder.applied == ["a", "b", "c"]


async def test_a_later_commit_applies_only_what_is_newly_committed(db_path):
    recorder = RecordingStateMachine()
    async with SqliteStore(db_path) as store:
        durable = await follower_with(store, recorder)
        await durable.handle_append_entries(
            append_entries(term=4, entries=command_entries("a", "b", "c"), leader_commit=2)
        )
        await durable.apply_committed()

        await durable.handle_append_entries(
            heartbeat(term=4, prev_log_index=3, prev_log_term=4, commit=3)
        )
        assert await durable.apply_committed() == 1

        assert recorder.applied == ["a", "b", "c"]  # "a" and "b" are not applied twice


async def test_applying_when_already_caught_up_does_nothing(db_path):
    recorder = RecordingStateMachine()
    async with SqliteStore(db_path) as store:
        durable = await follower_with(store, recorder)
        await durable.handle_append_entries(
            append_entries(term=4, entries=command_entries("a"), leader_commit=1)
        )
        await durable.apply_committed()

        assert await durable.apply_committed() == 0
        assert recorder.applied == ["a"]


async def test_nothing_is_applied_before_anything_is_committed(db_path):
    recorder = RecordingStateMachine()
    async with SqliteStore(db_path) as store:
        durable = await follower_with(store, recorder)
        await durable.handle_append_entries(append_entries(term=4, entries=command_entries("a")))

        assert await durable.apply_committed() == 0
        assert (recorder.applied, durable.last_applied) == ([], 0)


# --- A Leader's empty entry is not a command (DD-26) ----------------------------------


async def test_the_empty_entry_advances_the_index_without_reaching_the_state_machine(db_path):
    recorder = RecordingStateMachine()
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, ONE_NODE, apply=recorder)
        await durable.start_election()  # alone, so its empty entry commits at once
        assert durable.commit_index == 1

        assert await durable.apply_committed() == 0  # no command was applied

        assert recorder.applied == []
        assert durable.last_applied == 1  # but the index moved past it


async def test_commands_around_an_empty_entry_are_applied_in_order(db_path):
    recorder = RecordingStateMachine()
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, ONE_NODE, apply=recorder)
        await durable.start_election()
        await durable.append_command("a")
        await durable.append_command("b")

        assert await durable.apply_committed() == 2

        assert recorder.applied == ["a", "b"]
        assert durable.last_applied == 3  # the empty entry plus both commands


# --- The callback is the only way to the KV Store layer (DD-12) -----------------------


async def test_a_node_without_a_state_machine_refuses_to_apply_a_command(db_path):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await durable.handle_append_entries(
            append_entries(term=4, entries=command_entries("a"), leader_commit=1)
        )
        with pytest.raises(RuntimeError, match="no state machine"):
            await durable.apply_committed()
        assert durable.last_applied == 0


async def test_a_state_machine_that_raises_stops_before_the_entry_that_failed(db_path):
    recorder = RecordingStateMachine(fail_on="b")
    async with SqliteStore(db_path) as store:
        durable = await follower_with(store, recorder)
        await durable.handle_append_entries(
            append_entries(term=4, entries=command_entries("a", "b", "c"), leader_commit=3)
        )

        with pytest.raises(RuntimeError):
            await durable.apply_committed()

        assert recorder.applied == ["a"]
        assert durable.last_applied == 1  # so "b" is retried, not skipped


# --- The state machine is rebuilt from the log after a restart ------------------------


async def test_a_restarted_node_replays_its_whole_log_into_a_fresh_state_machine(db_path):
    # Seeded with real commands, since a replay hands every entry to the state machine: a
    # session opened at index 1, which is its client ID, and two of its puts.
    async with SqliteStore(db_path) as store:
        await store.replace_log_from(
            1,
            command_entries(
                OpenSession().encode(),
                Put(1, 1, "x", "1").encode(),
                Put(1, 2, "y", "2").encode(),
                term=1,
            ),
        )
    await seed_term_and_vote(db_path, 1)
    kv = KeyValueStore()
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES, apply=kv.apply)
        await win_election(durable)  # term 2; its empty entry is index 4
        await durable.append_command(Put(1, 3, "x", "5").encode())
        await durable.handle_append_entries(heartbeat(term=9, leader=8, commit=0))  # steps down

    restarted_kv = KeyValueStore()
    async with SqliteStore(db_path) as store:
        restarted = await DurableNodeState.load(
            NODE_ID, store, THREE_NODES, apply=restarted_kv.apply
        )
        assert restarted.last_applied == 0  # not persisted
        # A Leader tells it how far to commit, and the whole log is replayed from the start.
        await restarted.handle_append_entries(
            heartbeat(term=9, leader=8, prev_log_index=5, prev_log_term=2, commit=5)
        )
        await restarted.apply_committed()

        assert restarted.last_applied == 5
        assert restarted_kv.as_dict() == {"x": "5", "y": "2"}  # replayed from index 1
        assert set(restarted_kv.sessions) == {1}


# --- CLIENT-10: has this Leader committed an entry of its own term? -------------------


async def test_a_leader_has_not_committed_in_its_term_until_a_follower_confirms(db_path):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await win_election(durable)
        assert durable.has_committed_in_current_term is False

        request = await durable.append_entries_request_for(8)
        await durable.handle_append_entries_response(8, request, accepted(term=1))

        assert durable.has_committed_in_current_term is True


async def test_a_leader_alone_in_its_cluster_has_committed_in_its_term_at_once(db_path):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, ONE_NODE)
        await durable.start_election()
        assert durable.has_committed_in_current_term is True


async def test_inherited_entries_alone_do_not_count_as_committed_in_this_term(db_path):
    # The node holds two committed term-1 entries and has just won term 2. Everything it has
    # committed is from the old term, so it may not answer a read yet.
    async with SqliteStore(db_path) as store:
        await store.replace_log_from(1, command_entries("a", "b", term=1))
    await seed_term_and_vote(db_path, 1)
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await durable.handle_append_entries(
            heartbeat(term=1, leader=8, prev_log_index=2, prev_log_term=1, commit=2)
        )
        assert durable.commit_index == 2
        await win_election(durable)

        assert durable.role is Role.LEADER
        assert durable.commit_index == 2
        assert durable.has_committed_in_current_term is False


async def test_an_async_state_machine_is_refused_when_the_node_is_built(db_path):
    # Calling one would build a coroutine and drop it: last_applied would run ahead of a state
    # machine that never saw the command. Awaiting it is no answer either, since the callback
    # runs under the node's lock. So it is refused at wiring time, not at the first commit.
    async def apply(index, cluster_time, command):
        raise AssertionError("never reached")

    async with SqliteStore(db_path) as store:
        with pytest.raises(TypeError, match="synchronous"):
            await DurableNodeState.load(NODE_ID, store, THREE_NODES, apply=apply)
