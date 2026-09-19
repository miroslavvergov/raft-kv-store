"""Tier 1 unit tests for the voter's side of an election,
NodeState.handle_vote_request: at most one vote per term (ELECT-8), only
for a Candidate whose log is at least as up to date (ELECT-9, ELECT-10),
a request from an earlier term refused with the current term, a higher
term caught up to first (STATE-4, STATE-5, STATE-6), a repeated request
answered the same way (FAIL-1), and a randomized check that no node ever
grants two different Candidates a vote in the same term.
"""

import random
from collections import defaultdict

import pytest

from raftkv.consensus import (
    LogPosition,
    NodeState,
    RequestVoteRequest,
    RequestVoteResponse,
    Role,
)

EMPTY_LOG = LogPosition(term=0, index=0)


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


# --- ELECT-8: one vote per term ---------------------------------------------------


def test_a_fresh_node_grants_the_first_request_and_records_the_vote():
    node = NodeState(node_id=1)
    assert node.handle_vote_request(vote_request(1, 2), EMPTY_LOG) == granted(1)
    assert (node.role, node.current_term, node.voted_for) == (Role.FOLLOWER, 1, 2)


def test_a_second_candidate_in_the_same_term_is_refused():
    node = NodeState(node_id=1)
    node.handle_vote_request(vote_request(1, 2), EMPTY_LOG)
    assert node.handle_vote_request(vote_request(1, 3), EMPTY_LOG) == refused(1)
    assert node.voted_for == 2


def test_a_repeated_request_from_the_same_candidate_is_granted_again():
    # A retried request, or one whose answer was lost: answering it the same
    # way again gives no second vote.
    node = NodeState(node_id=1)
    node.handle_vote_request(vote_request(1, 2), EMPTY_LOG)
    assert node.handle_vote_request(vote_request(1, 2), EMPTY_LOG) == granted(1)
    assert (node.current_term, node.voted_for) == (1, 2)


def test_a_reloaded_vote_still_blocks_a_second_candidate():
    node = NodeState.reloaded(node_id=1, current_term=5, voted_for=2)
    assert node.handle_vote_request(vote_request(5, 3), EMPTY_LOG) == refused(5)
    assert node.voted_for == 2


# --- A request from an earlier term ------------------------------------------------


def test_a_request_from_an_earlier_term_is_refused_with_the_current_term():
    node = NodeState.reloaded(node_id=1, current_term=5, voted_for=None)
    assert node.handle_vote_request(vote_request(3, 2), EMPTY_LOG) == refused(5)
    assert (node.current_term, node.voted_for) == (5, None)


def test_an_earlier_term_is_refused_even_for_the_node_voted_for_now():
    # The node remembers only its term-5 vote. Whatever it did in term 3 is
    # gone, so it cannot know whether granting a term-3 vote would be its
    # second one in that term.
    node = NodeState.reloaded(node_id=1, current_term=5, voted_for=2)
    assert node.handle_vote_request(vote_request(3, 2), EMPTY_LOG) == refused(5)


# --- A request from a higher term (STATE-4, STATE-5, STATE-6) ---------------------


def test_a_higher_term_clears_the_old_vote_and_can_be_granted():
    node = NodeState.reloaded(node_id=1, current_term=5, voted_for=3)
    assert node.handle_vote_request(vote_request(6, 2), EMPTY_LOG) == granted(6)
    assert (node.current_term, node.voted_for) == (6, 2)


def test_a_higher_term_is_adopted_even_when_the_vote_is_refused():
    node = NodeState.reloaded(node_id=1, current_term=5, voted_for=3)
    own_log = LogPosition(term=5, index=4)
    stale = vote_request(6, 2, last_log_term=4, last_log_index=9)
    assert node.handle_vote_request(stale, own_log) == refused(6)
    assert (node.current_term, node.voted_for) == (6, None)


def test_a_candidate_steps_down_for_a_higher_term_and_can_vote_in_it():
    node = NodeState(node_id=1)
    node.become_candidate()  # term 1, voted for itself
    assert node.handle_vote_request(vote_request(2, 3), EMPTY_LOG) == granted(2)
    assert (node.role, node.current_term, node.voted_for) == (Role.FOLLOWER, 2, 3)


def test_a_leader_steps_down_for_a_higher_term():
    node = NodeState(node_id=1)
    node.become_candidate()
    node.become_leader()
    assert node.handle_vote_request(vote_request(2, 3), EMPTY_LOG) == granted(2)
    assert node.role is Role.FOLLOWER


def test_a_candidate_refuses_a_rival_in_its_own_term():
    node = NodeState(node_id=1)
    node.become_candidate()
    assert node.handle_vote_request(vote_request(1, 2), EMPTY_LOG) == refused(1)
    assert (node.role, node.voted_for) == (Role.CANDIDATE, 1)


def test_a_leader_refuses_a_request_in_its_own_term():
    node = NodeState(node_id=1)
    node.become_candidate()
    node.become_leader()
    assert node.handle_vote_request(vote_request(1, 2), EMPTY_LOG) == refused(1)
    assert node.role is Role.LEADER


# --- ELECT-9, ELECT-10: only for a log at least as up to date ----------------------


@pytest.mark.parametrize(
    "candidate_last, own_last, expected",
    [
        ((3, 2), (2, 9), True),  # later last term wins, even though shorter
        ((2, 9), (3, 2), False),  # longer, but its last term is older
        ((3, 5), (3, 4), True),  # same last term, longer
        ((3, 4), (3, 4), True),  # identical: at least as up to date
        ((3, 3), (3, 4), False),  # same last term, shorter
        ((0, 0), (0, 0), True),  # both empty
        ((0, 0), (1, 1), False),  # empty against non-empty
        ((1, 1), (0, 0), True),  # non-empty against empty
    ],
    ids=[
        "later-term-shorter",
        "older-term-longer",
        "same-term-longer",
        "identical",
        "same-term-shorter",
        "both-empty",
        "empty-candidate",
        "empty-voter",
    ],
)
def test_the_vote_follows_the_up_to_date_rule(candidate_last, own_last, expected):
    node = NodeState.reloaded(node_id=1, current_term=5, voted_for=None)
    request = vote_request(6, 2, last_log_term=candidate_last[0], last_log_index=candidate_last[1])
    own = LogPosition(term=own_last[0], index=own_last[1])
    response = node.handle_vote_request(request, own)
    assert response.vote_granted is expected
    assert node.voted_for == (2 if expected else None)


def test_refusing_a_stale_log_leaves_the_vote_free_for_another_candidate():
    node = NodeState.reloaded(node_id=1, current_term=5, voted_for=None)
    own_log = LogPosition(term=3, index=4)
    stale = vote_request(6, 2, last_log_term=2, last_log_index=9)
    current = vote_request(6, 3, last_log_term=3, last_log_index=4)
    assert node.handle_vote_request(stale, own_log) == refused(6)
    assert node.handle_vote_request(current, own_log) == granted(6)
    assert node.voted_for == 3


# --- Never two Candidates in one term -----------------------------------------------


@pytest.mark.parametrize("seed", range(50))
def test_no_term_ever_has_two_candidates_granted_a_vote(seed):
    # Deterministic pseudo-random requests — from several Candidates, across
    # earlier, current, and later terms, with logs of every kind, repeated
    # freely, and with the node now and then starting an election of its own
    # — checking after each one that the node never votes for two different
    # Candidates in the same term (itself included), and that its answers are
    # consistent with its own state.
    rng = random.Random(seed)
    node = NodeState.reloaded(node_id=1, current_term=rng.randint(0, 3), voted_for=None)
    own_log = LogPosition(term=rng.randint(0, 3), index=rng.randint(0, 6))
    granted_to = defaultdict(set)

    for _ in range(300):
        if rng.random() < 0.05:
            node.become_candidate()
            granted_to[node.current_term].add(node.node_id)
            continue
        term_before = node.current_term
        request = vote_request(
            term=max(0, term_before + rng.choice([-2, -1, 0, 0, 0, 1, 2])),
            candidate=rng.choice([2, 3, 4]),
            last_log_term=rng.randint(0, 4),
            last_log_index=rng.randint(0, 8),
        )
        response = node.handle_vote_request(request, own_log)

        assert node.current_term == max(term_before, request.term)
        assert response.term == node.current_term
        if response.vote_granted:
            assert request.term == node.current_term
            assert node.voted_for == request.candidate_id
            assert request.last_log_position.is_at_least_as_up_to_date_as(own_log)
            granted_to[request.term].add(request.candidate_id)
        assert all(len(candidates) == 1 for candidates in granted_to.values())
