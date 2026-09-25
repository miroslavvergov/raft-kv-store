"""Tier 2 tests for both sides of RequestVote through DurableNodeState.

ELECT-3 through ELECT-11, DD-8, DD-18, DD-19, DD-22. No request or answer exists until its term
and vote are on disk.
"""

import asyncio
from dataclasses import dataclass

import pytest

from raftkv.consensus import Cluster, IllegalTransitionError, LogEntry, Role
from raftkv.node import DurableNodeState
from raftkv.storage import SqliteStore
from tests.support.store_doubles import (
    FailingStore,
    GatedStore,
    RecordingStore,
    let_other_tasks_run,
    seed_log,
    seed_term_and_vote,
    term_and_vote_on_disk,
    win_election,
)
from tests.support.vote_messages import granted, refused, vote_request

NODE_ID = 7  # IDs from 7 up never look like the small terms and indexes these tests use.
THREE_NODES = Cluster([7, 8, 9])
FIVE_NODES = Cluster([7, 8, 9, 10, 11])


# --- The Candidate's side: start_election (ELECT-3 through ELECT-7) -------------------


async def test_the_request_carries_the_new_term_and_the_last_log_entry(db_path):
    await seed_log(db_path, [1, 1, 2])
    await seed_term_and_vote(db_path, 2)
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        request = await durable.start_election()
    assert request == vote_request(term=3, candidate=NODE_ID, last_log_term=2, last_log_index=3)


async def test_the_request_does_not_exist_until_the_self_vote_is_on_disk(db_path):
    async with GatedStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        election = asyncio.create_task(durable.start_election())
        await store.wait_for_write()
        await let_other_tasks_run()
        assert not election.done()  # nothing to send while the write is held
        assert durable.current_term == 0

        store.release.set()
        request = await election
        assert request.term == 1
    assert await term_and_vote_on_disk(db_path) == (1, NODE_ID)


async def test_an_election_starts_counting_with_the_candidates_own_vote(db_path):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, FIVE_NODES)
        await durable.start_election()
        assert durable.role is Role.CANDIDATE
        assert durable.candidacy.term == 1
        assert durable.candidacy.votes_granted == frozenset({NODE_ID})
        assert durable.leadership is None
        assert durable.peers == frozenset({8, 9, 10, 11})


async def test_a_new_election_discards_the_votes_of_the_previous_one(db_path):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, FIVE_NODES)
        first = await durable.start_election()
        first_grant = granted(term=first.term)
        await durable.handle_vote_response(8, first.term, first_grant)
        assert durable.candidacy.votes_granted == frozenset({NODE_ID, 8})

        second = await durable.start_election()  # split vote: try again, term 2
        assert durable.candidacy.term == 2
        assert durable.candidacy.votes_granted == frozenset({NODE_ID})

        # Late grants from the term-1 election count for nothing in term 2.
        for voter in (9, 10):
            assert await durable.handle_vote_response(voter, first.term, first_grant) is False
        assert durable.candidacy.votes_granted == frozenset({NODE_ID})
        assert durable.role is Role.CANDIDATE
        assert second.term == 2


async def test_a_single_node_cluster_wins_on_its_own_vote(db_path):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, Cluster([NODE_ID]))
        request = await durable.start_election()
        assert durable.role is Role.LEADER
        assert durable.leadership.term == request.term == 1
        assert durable.leadership.followers == frozenset()
        assert durable.candidacy is None


async def test_a_single_node_election_cancelled_mid_write_still_ends_as_leader(db_path):
    # Winning is decided before the write and installed with it, so no cancelled caller can
    # leave the node a Candidate that has already won.
    async with GatedStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, Cluster([NODE_ID]))
        election = asyncio.create_task(durable.start_election())
        await store.wait_for_write()
        election.cancel()
        store.release.set()
        with pytest.raises(asyncio.CancelledError):
            await election
        assert (durable.role, durable.current_term, durable.voted_for) == (Role.LEADER, 1, NODE_ID)
        assert durable.leadership.term == 1
        assert durable.candidacy is None
    assert await term_and_vote_on_disk(db_path) == (1, NODE_ID)


async def test_a_node_outside_the_cluster_cannot_be_loaded(db_path):
    async with SqliteStore(db_path) as store:
        with pytest.raises(ValueError):
            await DurableNodeState.load(NODE_ID, store, Cluster([1, 2, 3]))


async def test_an_election_no_longer_due_once_the_lock_is_held_changes_nothing(db_path):
    # The caller's own check runs under the lock, before anything is decided or written.
    async with RecordingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)

        assert await durable.start_election(still_due=lambda: False) is None

        assert (durable.role, durable.current_term, durable.voted_for) == (Role.FOLLOWER, 0, None)
        assert store.writes == []


# --- The Candidate's side: counting answers (ELECT-11, REPL-14, REPL-15) --------------


async def test_winning_starts_a_fresh_leadership_for_every_peer(db_path):
    await seed_log(db_path, [1, 1, 2])
    await seed_term_and_vote(db_path, 2)
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        request = await durable.start_election()
        grant = granted(term=request.term)
        assert await durable.handle_vote_response(9, request.term, grant) is True

        assert durable.role is Role.LEADER
        assert durable.candidacy is None
        leadership = durable.leadership
        assert leadership.term == request.term
        assert leadership.followers == frozenset({8, 9})
        for peer in (8, 9):
            assert (leadership.next_index(peer), leadership.match_index(peer)) == (4, 0)


async def test_five_nodes_become_leader_at_the_third_vote_and_not_before(db_path):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, FIVE_NODES)
        request = await durable.start_election()
        grant, refusal = granted(term=request.term), refused(term=request.term)
        assert await durable.handle_vote_response(8, request.term, grant) is False
        assert await durable.handle_vote_response(9, request.term, refusal) is False
        assert durable.role is Role.CANDIDATE
        assert await durable.handle_vote_response(10, request.term, grant) is True
        assert durable.role is Role.LEADER


async def test_one_voters_grant_delivered_three_times_is_one_vote(db_path):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, FIVE_NODES)
        request = await durable.start_election()
        grant = granted(term=request.term)
        for _ in range(3):
            assert await durable.handle_vote_response(8, request.term, grant) is False
        assert durable.role is Role.CANDIDATE
        assert durable.candidacy.votes_granted == frozenset({NODE_ID, 8})


async def test_an_answer_with_a_higher_term_ends_the_candidacy_and_is_persisted(db_path):
    async with RecordingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        request = await durable.start_election()
        assert await durable.handle_vote_response(8, request.term, refused(term=5)) is False
        assert (durable.role, durable.current_term, durable.voted_for) == (Role.FOLLOWER, 5, None)
        assert durable.candidacy is None
        assert store.writes[-1] == ("term_and_vote", 5, None)
        # A grant still in flight from the abandoned election changes nothing.
        late_grant = granted(term=request.term)
        assert await durable.handle_vote_response(9, request.term, late_grant) is False
        assert durable.role is Role.FOLLOWER
    assert await term_and_vote_on_disk(db_path) == (5, None)


async def test_answers_arriving_after_the_win_change_nothing(db_path):
    async with RecordingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, FIVE_NODES)
        request = await durable.start_election()
        grant, refusal = granted(term=request.term), refused(term=request.term)
        for voter in (8, 9):
            await durable.handle_vote_response(voter, request.term, grant)
        leadership = durable.leadership
        writes_before = list(store.writes)

        assert await durable.handle_vote_response(10, request.term, grant) is False
        assert await durable.handle_vote_response(11, request.term, refusal) is False
        assert durable.role is Role.LEADER
        assert durable.leadership is leadership
        assert store.writes == writes_before


async def test_winning_again_later_starts_a_new_leadership_from_scratch(db_path):
    # Progress confirmed while Leader in term 1 must not survive into a later leadership
    # (REPL-14, REPL-15), even for the same node and followers.
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        first = await win_election(durable)
        durable.leadership.record_success(
            8, sent_in_term=first.term, prev_log_index=0, entry_count=0
        )
        old_leadership = durable.leadership

        await durable.handle_observed_term(3)
        assert durable.leadership is None
        second = await win_election(durable)

        assert durable.leadership is not old_leadership
        assert durable.leadership.term == second.term == 4
        # One past the log it won with, which ends with term 1's empty entry; the new empty
        # entry is appended after, so next_index points at it.
        assert (durable.leadership.next_index(8), durable.leadership.match_index(8)) == (2, 0)
        assert durable.log.entry_at(2) == LogEntry.empty(4)


async def test_a_leader_cannot_start_an_election(db_path):
    async with RecordingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await win_election(durable)
        writes_before = list(store.writes)
        with pytest.raises(IllegalTransitionError):
            await durable.start_election()
        assert (durable.role, durable.current_term) == (Role.LEADER, 1)
        assert store.writes == writes_before


# --- The voter's side: persisted before answering (PERSIST-1, PERSIST-2) --------------


async def test_a_granted_vote_is_on_disk_when_the_answer_is_returned(db_path):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        response = await durable.handle_vote_request(vote_request(term=1, candidate=8))
        assert response == granted(term=1)
        # Read back before the answer is "sent".
        assert await term_and_vote_on_disk(db_path) == (1, 8)


async def test_the_answer_does_not_exist_until_the_vote_is_on_disk(db_path):
    async with GatedStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        answering = asyncio.create_task(
            durable.handle_vote_request(vote_request(term=1, candidate=8))
        )
        await store.wait_for_write()
        await let_other_tasks_run()
        assert not answering.done()
        assert durable.voted_for is None

        store.release.set()
        assert await answering == granted(term=1)
        assert durable.voted_for == 8


async def test_answers_that_change_nothing_write_nothing(db_path):
    async with RecordingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await durable.handle_vote_request(vote_request(term=2, candidate=8))
        writes_before = list(store.writes)

        repeat = vote_request(term=2, candidate=8)
        rival = vote_request(term=2, candidate=9)
        stale = vote_request(term=1, candidate=9)
        assert await durable.handle_vote_request(repeat) == granted(term=2)
        assert await durable.handle_vote_request(rival) == refused(term=2)
        assert await durable.handle_vote_request(stale) == refused(term=2)
        assert store.writes == writes_before


async def test_a_refusal_that_adopts_a_higher_term_persists_it(db_path):
    await seed_log(db_path, [1, 3])
    async with RecordingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        stale_log = vote_request(term=4, candidate=8, last_log_term=2, last_log_index=5)
        assert await durable.handle_vote_request(stale_log) == refused(term=4)
        assert store.writes == [("term_and_vote", 4, None)]


async def test_a_leader_asked_for_a_vote_in_a_higher_term_steps_down(db_path):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await win_election(durable)
        # The Candidate's log matches the Leader's, whose last entry is its term-1 empty entry.
        up_to_date = vote_request(term=2, candidate=9, last_log_term=1, last_log_index=1)
        response = await durable.handle_vote_request(up_to_date)
        assert response == granted(term=2)
        assert (durable.role, durable.current_term, durable.voted_for) == (Role.FOLLOWER, 2, 9)
        assert durable.leadership is None
    assert await term_and_vote_on_disk(db_path) == (2, 9)


async def test_a_failed_write_gives_no_answer_and_no_vote(db_path):
    async with FailingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        with pytest.raises(OSError):
            await durable.handle_vote_request(vote_request(term=1, candidate=8))
        assert (durable.current_term, durable.voted_for) == (0, None)


async def test_a_cancelled_answer_still_installs_the_vote_it_persisted(db_path):
    async with GatedStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        answering = asyncio.create_task(
            durable.handle_vote_request(vote_request(term=1, candidate=8))
        )
        await store.wait_for_write()
        answering.cancel()
        store.release.set()
        with pytest.raises(asyncio.CancelledError):
            await answering
        # No answer was sent, but the vote is on disk, so it must be in memory too, or a second
        # Candidate could get this node's term-1 vote.
        assert (durable.current_term, durable.voted_for) == (1, 8)
        rival = vote_request(term=1, candidate=9)
        assert await durable.handle_vote_request(rival) == refused(term=1)
    assert await term_and_vote_on_disk(db_path) == (1, 8)


# --- Two Candidates asking at the same moment (ELECT-8, DD-8) -------------------------


@dataclass
class VoteRace:
    """What two Candidates asking for the same term's vote at once produced.

    Attributes:
        writes_while_held: The writes the store received before the first was released.
        answers: The answers to node 8 and to node 9.
        persisted: The (current_term, voted_for) on disk at the end.
    """

    writes_while_held: list
    answers: list
    persisted: tuple


async def race_two_candidates(db_path):
    """Hold the vote write for node 8's term-1 request in flight, then let node 9 ask for term 1.

    No real time passes, so the outcome does not depend on timing.
    """
    async with GatedStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        first = asyncio.create_task(durable.handle_vote_request(vote_request(term=1, candidate=8)))
        await store.wait_for_write()
        second = asyncio.create_task(durable.handle_vote_request(vote_request(term=1, candidate=9)))
        await let_other_tasks_run()
        writes_while_held = list(store.writes)
        store.release.set()
        answers = await asyncio.gather(first, second)
    return VoteRace(writes_while_held, answers, await term_and_vote_on_disk(db_path))


async def test_two_candidates_asking_at_once_never_both_get_the_vote(db_path):
    race = await race_two_candidates(db_path)
    assert race.writes_while_held == [("term_and_vote", 1, 8)]
    assert race.answers == [granted(term=1), refused(term=1)]
    assert race.persisted == (1, 8)


@pytest.mark.negative_control
async def test_negative_control_without_the_lock_both_candidates_get_the_vote(
    db_path, without_the_lock
):
    # Proves the test above can fail: without DD-8's lock, the second request is decided against
    # the state from before the first vote was installed, so the node votes twice in term 1.
    race = await race_two_candidates(db_path)
    assert race.writes_while_held == [("term_and_vote", 1, 8), ("term_and_vote", 1, 9)]
    assert race.answers == [granted(term=1), granted(term=1)]
    assert race.persisted == (1, 9)
