"""Tier 2 component tests for elections through DurableNodeState (DD-8,
DD-18, DD-19, DD-22): the RequestVote obtainable only once the new term and
self-vote are on disk (ELECT-5) and carrying the last log entry (ELECT-7);
a vote persisted before its answer exists (PERSIST-1, PERSIST-2); votes
counted to exactly a majority (ELECT-11) into a fresh Leadership (REPL-14,
REPL-15); stale, repeated, and higher-term answers; and — with a store
that holds a write in flight — two Candidates asking at once never both
getting the vote (ELECT-8), with a negative control showing that without
DD-8's lock they do.
"""

import asyncio

import pytest

from raftkv.consensus import (
    Cluster,
    IllegalTransition,
    RequestVoteRequest,
    RequestVoteResponse,
    Role,
)
from raftkv.persistence import DurableNodeState, SqliteStore
from tests.persistence.store_doubles import (
    FailingStore,
    GatedStore,
    NoLock,
    RecordingStore,
    reload,
    seed_log,
)

NODE_ID = 7
THREE_NODES = Cluster([7, 8, 9])
FIVE_NODES = Cluster([7, 8, 9, 10, 11])


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "node.db")


def vote_request(term, candidate, last_log_term=0, last_log_index=0):
    return RequestVoteRequest(
        term=term,
        candidate_id=candidate,
        last_log_index=last_log_index,
        last_log_term=last_log_term,
    )


def granted(term):
    return RequestVoteResponse(term=term, vote_granted=True)


def refused(term):
    return RequestVoteResponse(term=term, vote_granted=False)


# --- The Candidate's side: start_election (ELECT-3 to ELECT-7) -----------------


async def test_the_request_carries_the_new_term_and_the_last_log_entry(db_path):
    await seed_log(db_path, [1, 1, 2])
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        request = await durable.start_election()
    assert request == RequestVoteRequest(
        term=1, candidate_id=NODE_ID, last_log_index=3, last_log_term=2
    )


async def test_the_request_does_not_exist_until_the_self_vote_is_on_disk(db_path):
    async with GatedStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        election = asyncio.create_task(durable.start_election())
        await store.wait_for_write()
        for _ in range(10):
            await asyncio.sleep(0)
        assert not election.done()  # nothing to send while the write is held
        assert durable.current_term == 0

        store.release.set()
        request = await election
        assert request.term == 1
    persisted = await reload(db_path)
    assert (persisted.current_term, persisted.voted_for) == (1, NODE_ID)


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
        await durable.handle_vote_response(8, first.term, granted(first.term))
        assert durable.candidacy.votes_granted == frozenset({NODE_ID, 8})

        second = await durable.start_election()  # split vote: try again, term 2
        assert durable.candidacy.term == 2
        assert durable.candidacy.votes_granted == frozenset({NODE_ID})

        # Late grants from the term-1 election count for nothing in term 2.
        for voter in (9, 10):
            late = await durable.handle_vote_response(voter, first.term, granted(first.term))
            assert late is False
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


async def test_a_node_outside_the_cluster_cannot_be_loaded(db_path):
    async with SqliteStore(db_path) as store:
        with pytest.raises(ValueError):
            await DurableNodeState.load(NODE_ID, store, Cluster([1, 2, 3]))


# --- The Candidate's side: counting answers (ELECT-11, REPL-14, REPL-15) --------


async def test_winning_starts_a_fresh_leadership_for_every_peer(db_path):
    await seed_log(db_path, [1, 1, 2])
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        request = await durable.start_election()
        assert await durable.handle_vote_response(9, request.term, granted(request.term)) is True

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
        assert await durable.handle_vote_response(8, request.term, granted(1)) is False
        assert await durable.handle_vote_response(9, request.term, refused(1)) is False
        assert durable.role is Role.CANDIDATE
        assert await durable.handle_vote_response(10, request.term, granted(1)) is True
        assert durable.role is Role.LEADER


async def test_one_voters_grant_delivered_three_times_is_one_vote(db_path):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, FIVE_NODES)
        request = await durable.start_election()
        for _ in range(3):
            assert await durable.handle_vote_response(8, request.term, granted(1)) is False
        assert durable.role is Role.CANDIDATE
        assert durable.candidacy.votes_granted == frozenset({NODE_ID, 8})


async def test_an_answer_with_a_higher_term_ends_the_candidacy_and_is_persisted(db_path):
    async with RecordingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        request = await durable.start_election()
        assert await durable.handle_vote_response(8, request.term, refused(5)) is False
        assert (durable.role, durable.current_term, durable.voted_for) == (Role.FOLLOWER, 5, None)
        assert durable.candidacy is None
        assert store.writes[-1] == ("term_and_vote", 5, None)
        # A grant still in flight from the abandoned election changes nothing.
        assert await durable.handle_vote_response(9, request.term, granted(1)) is False
        assert durable.role is Role.FOLLOWER
    persisted = await reload(db_path)
    assert (persisted.current_term, persisted.voted_for) == (5, None)


async def test_answers_arriving_after_the_win_change_nothing(db_path):
    async with RecordingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, FIVE_NODES)
        request = await durable.start_election()
        for voter in (8, 9):
            await durable.handle_vote_response(voter, request.term, granted(1))
        leadership = durable.leadership
        writes_before = list(store.writes)

        assert await durable.handle_vote_response(10, request.term, granted(1)) is False
        assert await durable.handle_vote_response(11, request.term, refused(1)) is False
        assert durable.role is Role.LEADER
        assert durable.leadership is leadership
        assert store.writes == writes_before


async def test_winning_again_later_starts_a_new_leadership_from_scratch(db_path):
    # Progress confirmed while Leader in term 1 must not survive into a later
    # leadership (REPL-14, REPL-15), even for the same node and followers.
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        first = await durable.start_election()
        await durable.handle_vote_response(8, first.term, granted(first.term))
        durable.leadership.record_success(8, first.term, prev_log_index=0, entry_count=0)
        old_leadership = durable.leadership

        await durable.handle_observed_term(3)
        assert durable.leadership is None
        second = await durable.start_election()
        await durable.handle_vote_response(9, second.term, granted(second.term))

        assert durable.leadership is not old_leadership
        assert durable.leadership.term == second.term == 4
        assert (durable.leadership.next_index(8), durable.leadership.match_index(8)) == (1, 0)


async def test_a_leader_cannot_start_an_election(db_path):
    async with RecordingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        request = await durable.start_election()
        await durable.handle_vote_response(8, request.term, granted(1))
        writes_before = list(store.writes)
        with pytest.raises(IllegalTransition):
            await durable.start_election()
        assert (durable.role, durable.current_term) == (Role.LEADER, 1)
        assert store.writes == writes_before


# --- The voter's side: persisted before answering (PERSIST-1, PERSIST-2) --------


async def test_a_granted_vote_is_on_disk_when_the_answer_is_returned(db_path):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        response = await durable.handle_vote_request(vote_request(1, 8))
        assert response == granted(1)
        persisted = await reload(db_path)  # read back before the answer is "sent"
        assert (persisted.current_term, persisted.voted_for) == (1, 8)


async def test_the_answer_does_not_exist_until_the_vote_is_on_disk(db_path):
    async with GatedStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        answering = asyncio.create_task(durable.handle_vote_request(vote_request(1, 8)))
        await store.wait_for_write()
        for _ in range(10):
            await asyncio.sleep(0)
        assert not answering.done()
        assert durable.voted_for is None

        store.release.set()
        assert await answering == granted(1)
        assert durable.voted_for == 8


async def test_answers_that_change_nothing_write_nothing(db_path):
    async with RecordingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await durable.handle_vote_request(vote_request(1, 8))
        writes_before = list(store.writes)

        assert await durable.handle_vote_request(vote_request(1, 8)) == granted(1)  # repeat
        assert await durable.handle_vote_request(vote_request(1, 9)) == refused(1)  # rival
        assert await durable.handle_vote_request(vote_request(0, 9)) == refused(1)  # stale
        assert store.writes == writes_before


async def test_a_refusal_that_adopts_a_higher_term_persists_it(db_path):
    await seed_log(db_path, [1, 3])
    async with RecordingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        stale_log = vote_request(4, 8, last_log_term=2, last_log_index=5)
        assert await durable.handle_vote_request(stale_log) == refused(4)
        assert store.writes == [("term_and_vote", 4, None)]


async def test_a_leader_asked_for_a_vote_in_a_higher_term_steps_down(db_path):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        request = await durable.start_election()
        await durable.handle_vote_response(8, request.term, granted(1))
        assert await durable.handle_vote_request(vote_request(2, 9)) == granted(2)
        assert (durable.role, durable.current_term, durable.voted_for) == (Role.FOLLOWER, 2, 9)
        assert durable.leadership is None
    persisted = await reload(db_path)
    assert (persisted.current_term, persisted.voted_for) == (2, 9)


async def test_a_failed_write_gives_no_answer_and_no_vote(db_path):
    async with FailingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        with pytest.raises(OSError):
            await durable.handle_vote_request(vote_request(1, 8))
        assert (durable.current_term, durable.voted_for) == (0, None)


async def test_a_cancelled_answer_still_installs_the_vote_it_persisted(db_path):
    async with GatedStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        answering = asyncio.create_task(durable.handle_vote_request(vote_request(1, 8)))
        await store.wait_for_write()
        answering.cancel()
        store.release.set()
        with pytest.raises(asyncio.CancelledError):
            await answering
        # No answer was sent, but the vote is on disk — so it must be in
        # memory too, or a second Candidate could get this node's term-1 vote.
        assert (durable.current_term, durable.voted_for) == (1, 8)
        assert await durable.handle_vote_request(vote_request(1, 9)) == refused(1)
    persisted = await reload(db_path)
    assert (persisted.current_term, persisted.voted_for) == (1, 8)


# --- ELECT-8 under DD-8: two Candidates asking at the same moment --------------


async def race_two_candidates(db_path):
    """Hold the first Candidate's vote write in flight, then let the second one ask.

    Returns the writes that reached the store while the first was held,
    both answers, and the term and vote finally on disk. No real time
    passes, so the outcome does not depend on timing.
    """
    async with GatedStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        first = asyncio.create_task(durable.handle_vote_request(vote_request(1, 8)))
        await store.wait_for_write()
        second = asyncio.create_task(durable.handle_vote_request(vote_request(1, 9)))
        for _ in range(10):
            await asyncio.sleep(0)
        writes_while_held = list(store.writes)
        store.release.set()
        answers = await asyncio.gather(first, second)
    persisted = await reload(db_path)
    return writes_while_held, answers, (persisted.current_term, persisted.voted_for)


async def test_two_candidates_asking_at_once_never_both_get_the_vote(db_path):
    writes_while_held, answers, persisted = await race_two_candidates(db_path)
    assert writes_while_held == [(1, 8)]
    assert answers == [granted(1), refused(1)]
    assert persisted == (1, 8)


@pytest.mark.negative_control
async def test_negative_control_without_the_lock_both_candidates_get_the_vote(
    db_path, monkeypatch
):
    # Confirms the test above can fail: with DD-8's lock replaced by a no-op,
    # the second request is decided against the state from before the first
    # vote was installed, and the node votes twice in term 1.
    original_init = DurableNodeState.__init__

    def init_without_lock(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._lock = NoLock()

    monkeypatch.setattr(DurableNodeState, "__init__", init_without_lock)

    writes_while_held, answers, persisted = await race_two_candidates(db_path)
    assert writes_while_held == [(1, 8), (1, 9)]
    assert answers == [granted(1), granted(1)]
    assert persisted == (1, 9)
