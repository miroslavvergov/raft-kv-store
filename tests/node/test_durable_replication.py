"""Tier 2 tests for the Leader's side of replication in DurableNodeState.

A Leader appends commands (append_command), builds each Follower's AppendEntries
(append_entries_request_for), records the answers (handle_append_entries_response), and commits
what a majority holds from its own term. A new Leader appends an empty entry first. REPL-1,
REPL-2, REPL-4, REPL-6, REPL-7, REPL-16, REPL-17, APPLY-1, APPLY-2, APPLY-3, CLIENT-6, PERSIST-3,
DD-26, DD-27.
"""

import asyncio
import dataclasses

import pytest

from raftkv.consensus import (
    AppendEntriesRequest,
    Cluster,
    Log,
    LogEntry,
    LogPosition,
    NotLeaderError,
    Role,
)
from raftkv.node import DurableNodeState
from raftkv.storage import SqliteStore
from tests.support.append_entries_messages import accepted, append_entries, rejected
from tests.support.store_doubles import (
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

NODE_ID = 7
THREE_NODES = Cluster([7, 8, 9])
EMPTY_1 = LogEntry.empty(1)


async def elected_leader(store, cluster=THREE_NODES):
    """Load node 7 and win it term 1; its log then holds only its empty entry."""
    durable = await DurableNodeState.load(NODE_ID, store, cluster)
    await win_election(durable)
    return durable


async def win_term_2(store):
    """Win term 2 for node 7, whose file was seeded with two term-1 entries at term 1.

    Its empty entry is index 3, so each Follower's next_index starts at 3 and rejections have
    room to walk it back.
    """
    durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
    await win_election(durable)
    assert (durable.current_term, durable.log.last_index) == (2, 3)
    return durable


async def confirm(leader, follower):
    """Build `follower`'s AppendEntries and deliver an acceptance of it; return the request."""
    request = await leader.append_entries_request_for(follower)
    assert (
        await leader.handle_append_entries_response(follower, request, accepted(term=request.term))
        is False
    )
    return request


class LogWritesFailStore(SqliteStore):
    """A real store whose log writes fail once `fail_log_writes` is set, as a disk filling up."""

    def __init__(self, path):
        super().__init__(path)
        self.fail_log_writes = False

    async def replace_log_from(self, index, entries):
        if self.fail_log_writes:
            raise OSError("disk full")
        await super().replace_log_from(index, entries)


# --- A new Leader appends an empty entry in its term (APPLY-3) -------------------------


async def test_a_new_leader_appends_an_empty_entry_in_its_term_before_returning(db_path):
    async with SqliteStore(db_path) as store:
        leader = await elected_leader(store)
        assert leader.log == Log([EMPTY_1])
    assert (await reload(db_path)).log == Log([EMPTY_1])


async def test_the_empty_entry_follows_every_entry_the_leader_already_had(db_path):
    await seed_log(db_path, [1, 1])
    await seed_term_and_vote(db_path, 1)
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await win_election(durable)  # term 2
        assert [(e.term, e.is_empty) for e in durable.log] == [(1, False), (1, False), (2, True)]


async def test_the_empty_entry_is_not_committed_until_a_follower_has_it(db_path):
    async with SqliteStore(db_path) as store:
        leader = await elected_leader(store)
        assert leader.commit_index == 0


async def test_each_followers_first_request_carries_the_empty_entry(db_path):
    # next_index was set before the empty entry was appended, so it points at it.
    async with SqliteStore(db_path) as store:
        leader = await elected_leader(store)
        request = await leader.append_entries_request_for(8)
        assert (request.prev_log_index, request.entries) == (0, (EMPTY_1,))


async def test_a_single_node_leader_commits_its_empty_entry_at_once(db_path):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, Cluster([NODE_ID]))
        await durable.start_election()
        assert durable.role is Role.LEADER
        assert (durable.log, durable.commit_index) == (Log([EMPTY_1]), 1)


async def test_a_leader_whose_empty_entry_write_fails_stays_leader_without_it(db_path):
    # Documented in start_election and handle_vote_response: the node leads, but nothing of its
    # own term is in its log, so earlier entries wait for the first client command.
    async with LogWritesFailStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        request = await durable.start_election()
        store.fail_log_writes = True
        with pytest.raises(OSError):
            await durable.handle_vote_response(8, request.term, granted(term=request.term))

        assert durable.role is Role.LEADER
        assert len(durable.log) == 0
        assert durable.commit_index == 0

        store.fail_log_writes = False
        position = await durable.append_command("x=5")
        assert position.index == 1  # the index the empty entry would have had
    assert [e.command for e in (await reload(db_path)).log] == ["x=5"]


async def test_a_leader_restarted_mid_replication_comes_back_a_follower_that_kept_its_log(db_path):
    async with SqliteStore(db_path) as store:
        leader = await elected_leader(store)
        await leader.append_command("x=5")
        await confirm(leader, 8)
        assert leader.commit_index == 2

    async with SqliteStore(db_path) as store:
        restarted = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        assert restarted.role is Role.FOLLOWER
        assert [e.command for e in restarted.log] == ["", "x=5"]
        assert restarted.commit_index == 0  # commitment is not persisted
        assert restarted.leadership is None
        # It can no longer act as Leader, so an answer to one of its old requests is refused.
        with pytest.raises(NotLeaderError):
            await restarted.append_entries_request_for(8)


# --- append_command: the Leader writes a command into its own log (REPL-1) -------------


async def test_a_command_is_appended_in_the_leaders_term_and_persisted_before_returning(db_path):
    async with RecordingStore(db_path) as store:
        leader = await elected_leader(store)
        writes_before = list(store.writes)

        position = await leader.append_command("x=5")

        assert position == LogPosition(term=1, index=2)
        assert leader.log.entry_at(2) == LogEntry(term=1, command="x=5")
        assert store.writes == [*writes_before, ("replace_log_from", 2, [1])]
    assert (await reload(db_path)).log.entry_at(2) == LogEntry(term=1, command="x=5")


async def test_consecutive_commands_get_consecutive_indexes(db_path):
    async with SqliteStore(db_path) as store:
        leader = await elected_leader(store)
        positions = [await leader.append_command(c) for c in ("a=1", "b=2", "c=3")]
        assert [position.index for position in positions] == [2, 3, 4]
        assert [e.command for e in leader.log] == ["", "a=1", "b=2", "c=3"]


async def test_a_command_on_the_leader_alone_is_not_committed(db_path):
    async with SqliteStore(db_path) as store:
        leader = await elected_leader(store)
        await leader.append_command("x=5")
        assert leader.commit_index == 0


async def test_a_single_node_leaders_command_is_committed_at_once(db_path):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, Cluster([NODE_ID]))
        await durable.start_election()
        assert await durable.append_command("x=5") == LogPosition(term=1, index=2)
        assert durable.commit_index == 2


async def test_a_follower_refuses_a_command_and_writes_nothing(db_path):
    async with RecordingStore(db_path) as store:
        follower = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        with pytest.raises(NotLeaderError):
            await follower.append_command("x=5")
        assert store.writes == []
        assert len(follower.log) == 0


async def test_a_candidate_refuses_a_command(db_path):
    async with SqliteStore(db_path) as store:
        candidate = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await candidate.start_election()
        with pytest.raises(NotLeaderError):
            await candidate.append_command("x=5")


async def test_an_empty_command_is_refused_because_it_marks_a_leaders_empty_entry(db_path):
    async with SqliteStore(db_path) as store:
        leader = await elected_leader(store)
        with pytest.raises(ValueError, match="empty"):
            await leader.append_command("")
        assert leader.log == Log([EMPTY_1])


async def test_a_command_that_is_not_a_string_is_refused(db_path):
    async with SqliteStore(db_path) as store:
        leader = await elected_leader(store)
        with pytest.raises(TypeError):
            await leader.append_command(5)
        assert leader.log == Log([EMPTY_1])


async def test_a_failed_write_appends_nothing_and_the_next_command_takes_the_same_index(db_path):
    async with LogWritesFailStore(db_path) as store:
        leader = await elected_leader(store)
        store.fail_log_writes = True
        with pytest.raises(OSError):
            await leader.append_command("lost")
        assert leader.log == Log([EMPTY_1])

        store.fail_log_writes = False
        assert (await leader.append_command("kept")).index == 2
    assert [e.command for e in (await reload(db_path)).log] == ["", "kept"]


async def test_a_cancelled_caller_still_installs_and_commits_the_command(db_path):
    # A single node commits a command at once. Cancelled during the write, the entry is still
    # installed, since it is on disk, and committed, or it would wait for the next command.
    async with GatedStore(db_path) as store:
        store.release.set()
        durable = await DurableNodeState.load(NODE_ID, store, Cluster([NODE_ID]))
        await durable.start_election()
        store.hold_next_write()

        caller = asyncio.create_task(durable.append_command("x=5"))
        await store.wait_for_write()
        caller.cancel()
        store.release.set()
        with pytest.raises(asyncio.CancelledError):
            await caller

        assert durable.log.entry_at(2).command == "x=5"
        assert durable.commit_index == 2
    assert (await reload(db_path)).log.entry_at(2).command == "x=5"


async def test_commands_racing_each_other_are_appended_one_after_the_other(db_path):
    # The second command waits for the first one's write, so it never reads a log without it
    # and both cannot take the same index.
    async with GatedStore(db_path) as store:
        store.release.set()
        leader = await elected_leader(store)
        store.hold_next_write()

        writes_before = len(store.writes)
        first = asyncio.create_task(leader.append_command("first"))
        await store.wait_for_write()
        second = asyncio.create_task(leader.append_command("second"))
        await let_other_tasks_run()
        # Only the first command's write has reached the store; the second waits for the lock.
        assert store.writes[writes_before:] == [("replace_log_from", 2, [1])]

        store.release.set()
        positions = await asyncio.gather(first, second)
        assert [position.index for position in positions] == [2, 3]
        assert [e.command for e in leader.log] == ["", "first", "second"]


# --- append_entries_request_for: what the Leader sends a Follower (REPL-2, REPL-4) -----


async def test_the_request_carries_new_commands_after_the_last_confirmed_entry(db_path):
    async with SqliteStore(db_path) as store:
        leader = await elected_leader(store)
        await confirm(leader, 8)  # 8 has the empty entry; it is committed
        await leader.append_command("x=5")

        request = await leader.append_entries_request_for(8)

        assert (request.term, request.leader_id) == (1, NODE_ID)
        assert (request.prev_log_index, request.prev_log_term) == (1, 1)
        assert request.entries == (LogEntry(term=1, command="x=5"),)
        assert request.leader_commit == 1


async def test_building_a_request_writes_and_changes_nothing(db_path):
    async with RecordingStore(db_path) as store:
        leader = await elected_leader(store)
        writes_before = list(store.writes)
        await leader.append_entries_request_for(8)
        assert store.writes == writes_before
        assert (leader.leadership.next_index(8), leader.leadership.match_index(8)) == (1, 0)


async def test_a_follower_cannot_build_append_entries(db_path):
    async with SqliteStore(db_path) as store:
        follower = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        with pytest.raises(NotLeaderError):
            await follower.append_entries_request_for(8)


async def test_a_request_for_a_node_that_is_not_a_follower_raises_key_error(db_path):
    async with SqliteStore(db_path) as store:
        leader = await elected_leader(store)
        with pytest.raises(KeyError):
            await leader.append_entries_request_for(42)


# --- handle_append_entries_response: the Follower's answer (REPL-6, REPL-16) -----------


async def test_a_success_raises_the_followers_progress_and_commits_on_a_majority(db_path):
    async with SqliteStore(db_path) as store:
        leader = await elected_leader(store)
        await leader.append_command("x=5")
        await confirm(leader, 8)
        assert (leader.leadership.next_index(8), leader.leadership.match_index(8)) == (3, 2)
        assert leader.commit_index == 2  # leader and node 8: 2 of 3


async def test_a_rejection_lowers_next_index_and_asks_for_a_resend(db_path):
    await seed_log(db_path, [1, 1])
    await seed_term_and_vote(db_path, 1)
    async with SqliteStore(db_path) as store:
        leader = await win_term_2(store)
        request = await leader.append_entries_request_for(9)
        assert (request.prev_log_index, len(request.entries)) == (2, 1)

        resend = await leader.handle_append_entries_response(9, request, rejected(term=2))

        assert resend is True
        assert leader.leadership.next_index(9) == 2
        resent = await leader.append_entries_request_for(9)
        assert (resent.prev_log_index, len(resent.entries)) == (1, 2)
        assert leader.commit_index == 0  # a rejection commits nothing


async def test_rejections_walk_back_to_the_start_where_the_whole_log_is_sent(db_path):
    await seed_log(db_path, [1, 1])
    await seed_term_and_vote(db_path, 1)
    async with SqliteStore(db_path) as store:
        leader = await win_term_2(store)
        sent_after = []
        request = await leader.append_entries_request_for(9)
        while await leader.handle_append_entries_response(9, request, rejected(term=2)):
            sent_after.append(request.prev_log_index)
            request = await leader.append_entries_request_for(9)
            assert len(sent_after) <= 3, "next_index never reached its floor"

        # Rejected after 2 and after 1, each time asking for a resend further back.
        assert sent_after == [2, 1]
        # The last request, after index 0, carries the whole log. A rejection of it asks for no
        # resend: next_index is at its floor of 1, and a resend would repeat the same request.
        assert (request.prev_log_index, request.entries) == (0, tuple(leader.log))
        assert leader.leadership.next_index(9) == 1


async def test_an_answer_with_a_higher_term_makes_the_leader_step_down_and_persist_it(db_path):
    async with SqliteStore(db_path) as store:
        leader = await elected_leader(store)
        await confirm(leader, 8)
        request = await leader.append_entries_request_for(9)

        resend = await leader.handle_append_entries_response(9, request, rejected(term=5))

        assert resend is False
        assert (leader.role, leader.current_term, leader.voted_for) == (Role.FOLLOWER, 5, None)
        assert leader.leadership is None
        assert leader.commit_index == 1  # what was committed stays committed
    assert await term_and_vote_on_disk(db_path) == (5, None)


async def test_an_answer_to_a_request_from_an_earlier_term_is_ignored(db_path):
    # A delayed answer to a request the node sent while leading term 1 arrives after it has
    # won term 3. It describes a log that may since have changed, so no progress moves.
    async with SqliteStore(db_path) as store:
        leader = await elected_leader(store)
        old_request = await leader.append_entries_request_for(8)
        await leader.handle_observed_term(2)
        await win_election(leader)  # term 3
        progress_before = (leader.leadership.next_index(8), leader.leadership.match_index(8))

        resend = await leader.handle_append_entries_response(8, old_request, accepted(term=1))

        assert resend is False
        assert (leader.leadership.next_index(8), leader.leadership.match_index(8)) == (
            progress_before
        )
        assert leader.commit_index == 0


async def test_a_rejection_of_a_request_from_an_earlier_term_is_ignored(db_path):
    # A rejection delayed from an earlier leadership describes a log that may have changed
    # since; counting it would back off against entries this leadership never sent.
    await seed_log(db_path, [1, 1])
    await seed_term_and_vote(db_path, 1)
    async with SqliteStore(db_path) as store:
        leader = await win_term_2(store)
        old_request = await leader.append_entries_request_for(9)
        await leader.handle_observed_term(4)
        await win_election(leader)  # term 5
        progress = (leader.leadership.next_index(9), leader.leadership.match_index(9))

        assert (
            await leader.handle_append_entries_response(9, old_request, rejected(term=2)) is False
        )

        assert (leader.leadership.next_index(9), leader.leadership.match_index(9)) == progress


async def test_a_duplicated_rejection_backs_off_only_once(db_path):
    # The same rejection delivered twice must not lower next_index twice (FAIL-1): the second
    # answers a probe the Leader has already moved on from.
    await seed_log(db_path, [1, 1])
    await seed_term_and_vote(db_path, 1)
    async with SqliteStore(db_path) as store:
        leader = await win_term_2(store)
        request = await leader.append_entries_request_for(9)

        assert await leader.handle_append_entries_response(9, request, rejected(term=2)) is True
        after_first = leader.leadership.next_index(9)
        assert await leader.handle_append_entries_response(9, request, rejected(term=2)) is False

        assert leader.leadership.next_index(9) == after_first


async def test_a_late_rejection_of_an_older_probe_leaves_next_index_alone(db_path):
    # A rejection of a probe the Leader has already moved past must not back off again: only
    # the probe now outstanding counts (FAIL-1).
    await seed_log(db_path, [1, 1])
    await seed_term_and_vote(db_path, 1)
    async with SqliteStore(db_path) as store:
        leader = await win_term_2(store)
        request = await leader.append_entries_request_for(9)  # prev_log_index 2, next_index 3
        older = dataclasses.replace(request, prev_log_index=1, prev_log_term=1)

        assert await leader.handle_append_entries_response(9, older, rejected(term=2)) is False

        assert leader.leadership.next_index(9) == 3


async def test_a_stale_term_rejection_from_a_node_that_is_not_a_follower_is_ignored(db_path):
    # The term is checked before the Follower is looked up, so an answer to another term's
    # request is ignored rather than raising, whoever sent it.
    await seed_log(db_path, [1, 1])
    await seed_term_and_vote(db_path, 1)
    async with SqliteStore(db_path) as store:
        leader = await win_term_2(store)
        stale = append_entries(term=1, leader=NODE_ID, prev_log_index=2, prev_log_term=1)

        assert await leader.handle_append_entries_response(42, stale, rejected(term=1)) is False


async def test_an_answer_reaching_a_node_that_is_no_longer_leader_is_ignored(db_path):
    async with SqliteStore(db_path) as store:
        leader = await elected_leader(store)
        request = await leader.append_entries_request_for(8)
        await leader.handle_append_entries(append_entries(term=2, leader=9))  # a new Leader
        assert leader.role is Role.FOLLOWER
        assert await leader.handle_append_entries_response(8, request, accepted(term=1)) is False
        assert leader.commit_index == 0


async def test_a_duplicated_success_changes_nothing(db_path):
    async with SqliteStore(db_path) as store:
        leader = await elected_leader(store)
        await leader.append_command("x=5")
        request = await confirm(leader, 8)
        progress = (leader.leadership.next_index(8), leader.leadership.match_index(8))
        await leader.handle_append_entries_response(8, request, accepted(term=1))
        assert (leader.leadership.next_index(8), leader.leadership.match_index(8)) == progress
        assert leader.commit_index == 2


async def test_a_late_shorter_success_never_lowers_progress_or_commitment(db_path):
    # Two requests to node 8: an early one carrying only the empty entry, and a later one
    # carrying two commands too. The later answer arrives first.
    async with SqliteStore(db_path) as store:
        leader = await elected_leader(store)
        early = await leader.append_entries_request_for(8)
        await leader.append_command("a=1")
        await leader.append_command("b=2")
        later = await leader.append_entries_request_for(8)

        await leader.handle_append_entries_response(8, later, accepted(term=1))
        await leader.handle_append_entries_response(8, early, accepted(term=1))

        assert leader.leadership.match_index(8) == 3  # REPL-17: not lowered to 1
        assert leader.commit_index == 3


async def test_an_answer_from_a_node_that_is_not_a_follower_raises_key_error(db_path):
    async with SqliteStore(db_path) as store:
        leader = await elected_leader(store)
        request = await leader.append_entries_request_for(8)
        with pytest.raises(KeyError):
            await leader.handle_append_entries_response(42, request, accepted(term=1))


# --- Only the Leader's own term commits by counting (APPLY-2, APPLY-3) -----------------


async def test_entries_from_an_earlier_term_commit_only_with_the_leaders_empty_entry(db_path):
    # The node held two term-1 entries when it won term 2, then appended its empty entry 3.
    await seed_log(db_path, [1, 1])
    await seed_term_and_vote(db_path, 1)
    async with SqliteStore(db_path) as store:
        leader = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await win_election(leader)
        assert leader.current_term == 2

        # Node 8 confirms entries 1 and 2 only: they are on a majority, but from term 1.
        only_old = AppendEntriesRequest(
            term=2, leader_id=NODE_ID, entries=tuple(leader.log.entries_from(1)[:2])
        )
        await leader.handle_append_entries_response(8, only_old, accepted(term=2))
        assert leader.leadership.match_index(8) == 2
        assert leader.commit_index == 0

        # Node 8 confirms the empty entry too: it is from term 2, and all three commit.
        await confirm(leader, 8)
        assert leader.commit_index == 3


# --- A Leader and a Follower, both real, talking through their handlers ----------------


async def test_a_leader_and_a_follower_end_up_with_the_same_log_and_commit_index(tmp_path):
    leader_path, follower_path = str(tmp_path / "leader.db"), str(tmp_path / "follower.db")
    # The Follower holds a stale, never-committed entry from a Leader of term 1.
    await seed_log(follower_path, [1])
    await seed_term_and_vote(follower_path, 1)
    await seed_term_and_vote(leader_path, 1)

    async with SqliteStore(leader_path) as leader_store, SqliteStore(follower_path) as f_store:
        leader = await DurableNodeState.load(NODE_ID, leader_store, THREE_NODES)
        follower = await DurableNodeState.load(8, f_store, THREE_NODES)
        await win_election(leader)  # term 2: empty entry 1 is from term 2
        await leader.append_command("x=5")
        await leader.append_command("y=6")

        async def replicate():
            rejections = 0
            while True:
                request = await leader.append_entries_request_for(8)
                response = await follower.handle_append_entries(request)
                if not await leader.handle_append_entries_response(8, request, response):
                    return rejections
                rejections += 1
                assert rejections <= 3

        assert await replicate() == 0  # index 0 always matches: no back-off needed here
        assert follower.log == leader.log
        assert leader.commit_index == 3
        assert follower.commit_index == 0  # it learns the commit on the next AppendEntries

        await replicate()  # a heartbeat
        assert follower.commit_index == 3
        assert (follower.current_term, follower.role) == (2, Role.FOLLOWER)

    assert (await reload(follower_path)).log == (await reload(leader_path)).log
