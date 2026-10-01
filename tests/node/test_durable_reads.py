"""Tier 2 tests for the Leader's read support in DurableNodeState.

`mark_and_read_index` and `read_confirmed`: a read starts at a read index and a mark; a majority
answering an AppendEntries built after the mark confirms it. CLIENT-8 (confirm through a
majority), CLIENT-10 (only once an entry of the Leader's own term is committed), DD-34.
"""

import pytest

from raftkv.consensus import Cluster, LogEntry, NotLeaderError
from raftkv.node import DurableNodeState
from raftkv.storage import SqliteStore
from tests.support.append_entries_messages import accepted, append_entries, rejected
from tests.support.store_doubles import win_election

NODE_ID = 7
THREE_NODES = Cluster([7, 8, 9])
FIVE_NODES = Cluster([7, 8, 9, 10, 11])
ONE_NODE = Cluster([NODE_ID])


async def leader_that_has_committed(store, cluster=THREE_NODES):
    """Load node 7, win it term 1, and commit its empty entry through just enough Followers."""
    durable = await DurableNodeState.load(NODE_ID, store, cluster)
    await win_election(durable)
    for follower in sorted(durable.peers)[: cluster.majority - 1]:
        await answer(durable, follower)
    assert durable.has_committed_in_current_term
    return durable


async def answer(durable, follower, response=None):
    """Build `follower`'s AppendEntries and answer it, with an acceptance unless told otherwise."""
    request = await durable.append_entries_request_for(follower)
    await durable.handle_append_entries_response(
        follower, request, response or accepted(term=request.term)
    )
    return request


# --- When a read may start (CLIENT-10) ------------------------------------------------------


async def test_a_follower_can_neither_start_a_read_nor_confirm_one(db_path):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)

        with pytest.raises(NotLeaderError):
            durable.mark_and_read_index()
        with pytest.raises(NotLeaderError):
            durable.read_confirmed(0)


async def test_a_leader_that_has_committed_nothing_in_its_term_cannot_start_a_read(db_path):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await win_election(durable)  # its empty entry is on no Follower yet

        with pytest.raises(RuntimeError, match="committed nothing of term 1"):
            durable.mark_and_read_index()


async def test_a_commit_index_learned_in_an_earlier_term_does_not_let_a_new_leader_start_a_read(
    db_path,
):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await durable.handle_append_entries(
            append_entries(term=1, leader=8, entries=(LogEntry(1, "a"),), leader_commit=1)
        )
        await win_election(durable)  # term 2; its empty entry, index 2, is on no Follower yet
        assert durable.commit_index == 1
        assert not durable.has_committed_in_current_term

        with pytest.raises(RuntimeError, match="committed nothing of term 2"):
            durable.mark_and_read_index()


async def test_a_read_starts_with_the_commit_index_and_the_number_of_requests_built(db_path):
    async with SqliteStore(db_path) as store:
        durable = await leader_that_has_committed(store)  # one request built, one entry committed

        mark, read_index = durable.mark_and_read_index()

        assert (mark, read_index) == (1, durable.commit_index) == (1, 1)


async def test_the_read_index_is_the_commit_index_not_the_last_entry(db_path):
    async with SqliteStore(db_path) as store:
        durable = await leader_that_has_committed(store)
        await durable.append_command("x")  # entry 2, on no Follower yet

        _, read_index = durable.mark_and_read_index()

        assert (durable.log.last_index, durable.commit_index) == (2, 1)
        assert read_index == 1


async def test_asking_for_a_mark_and_read_index_changes_nothing_on_the_node(db_path):
    async with SqliteStore(db_path) as store:
        durable = await leader_that_has_committed(store)
        before = (durable.commit_index, durable.last_applied, durable.log.last_index)

        durable.mark_and_read_index()
        durable.mark_and_read_index()

        assert (durable.commit_index, durable.last_applied, durable.log.last_index) == before


# --- Which answers confirm a read (CLIENT-8, DD-34) -----------------------------------------


async def test_a_read_is_confirmed_once_a_majority_answers_a_request_built_after_it(db_path):
    async with SqliteStore(db_path) as store:
        durable = await leader_that_has_committed(store)
        mark, _ = durable.mark_and_read_index()
        assert not durable.read_confirmed(mark)

        await answer(durable, 9)

        assert durable.read_confirmed(mark)  # this node and node 9 are two of three


async def test_five_nodes_need_two_followers_besides_the_leader_to_confirm_a_read(db_path):
    async with SqliteStore(db_path) as store:
        durable = await leader_that_has_committed(store, FIVE_NODES)
        mark, _ = durable.mark_and_read_index()

        await answer(durable, 8)
        assert not durable.read_confirmed(mark)  # this node and one Follower: two of five

        await answer(durable, 9)
        assert durable.read_confirmed(mark)  # three of five


async def test_a_rejection_of_the_log_check_confirms_a_read_too(db_path):
    # The Follower took this Leader's term as its own before rejecting: it recognizes it.
    async with SqliteStore(db_path) as store:
        durable = await leader_that_has_committed(store)
        mark, _ = durable.mark_and_read_index()

        await answer(durable, 9, rejected(term=durable.current_term))

        assert durable.read_confirmed(mark)


async def test_an_answer_to_a_request_built_before_the_read_began_does_not_confirm_it(db_path):
    async with SqliteStore(db_path) as store:
        durable = await leader_that_has_committed(store)
        earlier = await durable.append_entries_request_for(9)
        mark, _ = durable.mark_and_read_index()

        await durable.handle_append_entries_response(9, earlier, accepted(term=earlier.term))

        assert not durable.read_confirmed(mark)


async def test_an_answer_to_a_superseded_request_does_not_confirm_a_read(db_path):
    async with SqliteStore(db_path) as store:
        durable = await leader_that_has_committed(store)
        mark, _ = durable.mark_and_read_index()
        older = await durable.append_entries_request_for(9)
        await durable.append_entries_request_for(9)  # built after `older`, before its answer

        await durable.handle_append_entries_response(9, older, accepted(term=older.term))

        assert not durable.read_confirmed(mark)


async def test_an_answer_carrying_a_lower_term_than_its_request_does_not_confirm_a_read(db_path):
    async with SqliteStore(db_path) as store:
        durable = await leader_that_has_committed(store)
        mark, _ = durable.mark_and_read_index()

        await answer(durable, 9, accepted(term=durable.current_term - 1))

        assert not durable.read_confirmed(mark)


async def test_a_node_that_is_no_longer_leader_cannot_confirm_a_read(db_path):
    async with SqliteStore(db_path) as store:
        durable = await leader_that_has_committed(store)
        mark, _ = durable.mark_and_read_index()
        later_term = durable.current_term + 1

        await answer(durable, 9, accepted(term=later_term))  # this Leader steps down

        with pytest.raises(NotLeaderError):
            durable.read_confirmed(mark)


async def test_a_leader_alone_in_its_cluster_confirms_a_read_at_once(db_path):
    async with SqliteStore(db_path) as store:
        durable = await leader_that_has_committed(store, ONE_NODE)

        mark, read_index = durable.mark_and_read_index()

        assert durable.read_confirmed(mark)
        assert read_index == durable.commit_index
