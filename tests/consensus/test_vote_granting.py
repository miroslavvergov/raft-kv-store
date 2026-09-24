"""Tier 1 tests for the voter's side of RequestVote: NodeState.handle_vote_request.

ELECT-8, ELECT-9, ELECT-10, STATE-4, STATE-5, STATE-6, FAIL-1.
"""

import random
from collections import defaultdict

import pytest

from raftkv.consensus import LogPosition, NodeState, Role
from tests.support.vote_messages import granted, refused, vote_request

EMPTY_LOG = LogPosition(term=0, index=0)


# --- One vote per term (ELECT-8) ------------------------------------------------------


def test_a_fresh_node_grants_the_first_request_and_records_the_vote():
    node = NodeState(node_id=1)
    request = vote_request(term=1, candidate=2)
    assert node.handle_vote_request(request, EMPTY_LOG) == granted(term=1)
    assert (node.role, node.current_term, node.voted_for) == (Role.FOLLOWER, 1, 2)


def test_a_second_candidate_in_the_same_term_is_refused():
    node = NodeState(node_id=1)
    node.handle_vote_request(vote_request(term=1, candidate=2), EMPTY_LOG)
    rival = vote_request(term=1, candidate=3)
    assert node.handle_vote_request(rival, EMPTY_LOG) == refused(term=1)
    assert node.voted_for == 2


def test_a_repeated_request_from_the_same_candidate_is_granted_again():
    # A retried request, or one whose answer was lost: answering it the same way again gives
    # no second vote.
    node = NodeState(node_id=1)
    request = vote_request(term=1, candidate=2)
    node.handle_vote_request(request, EMPTY_LOG)
    assert node.handle_vote_request(request, EMPTY_LOG) == granted(term=1)
    assert (node.current_term, node.voted_for) == (1, 2)


def test_a_reloaded_vote_still_blocks_a_second_candidate():
    node = NodeState.reloaded(node_id=1, current_term=5, voted_for=2)
    rival = vote_request(term=5, candidate=3)
    assert node.handle_vote_request(rival, EMPTY_LOG) == refused(term=5)
    assert node.voted_for == 2


# --- A request from an earlier term ---------------------------------------------------


def test_a_request_from_an_earlier_term_is_refused_with_the_current_term():
    node = NodeState.reloaded(node_id=1, current_term=5, voted_for=None)
    stale = vote_request(term=3, candidate=2)
    assert node.handle_vote_request(stale, EMPTY_LOG) == refused(term=5)
    assert (node.current_term, node.voted_for) == (5, None)


def test_an_earlier_term_is_refused_even_for_the_node_voted_for_now():
    # The node remembers only its term-5 vote. Whatever it did in term 3 is gone, so it cannot
    # know whether a term-3 vote would be its second in that term.
    node = NodeState.reloaded(node_id=1, current_term=5, voted_for=2)
    stale = vote_request(term=3, candidate=2)
    assert node.handle_vote_request(stale, EMPTY_LOG) == refused(term=5)


# --- A request from a higher term (STATE-4, STATE-5, STATE-6) -------------------------


def test_a_higher_term_clears_the_old_vote_and_can_be_granted():
    node = NodeState.reloaded(node_id=1, current_term=5, voted_for=3)
    request = vote_request(term=6, candidate=2)
    assert node.handle_vote_request(request, EMPTY_LOG) == granted(term=6)
    assert (node.current_term, node.voted_for) == (6, 2)


def test_a_higher_term_is_adopted_even_when_the_vote_is_refused():
    node = NodeState.reloaded(node_id=1, current_term=5, voted_for=3)
    own_log = LogPosition(term=5, index=4)
    stale_log = vote_request(term=6, candidate=2, last_log_term=4, last_log_index=9)
    assert node.handle_vote_request(stale_log, own_log) == refused(term=6)
    assert (node.current_term, node.voted_for) == (6, None)


def test_a_candidate_steps_down_for_a_higher_term_and_can_vote_in_it():
    node = NodeState(node_id=1)
    node.become_candidate()  # term 1, voted for itself
    request = vote_request(term=2, candidate=3)
    assert node.handle_vote_request(request, EMPTY_LOG) == granted(term=2)
    assert (node.role, node.current_term, node.voted_for) == (Role.FOLLOWER, 2, 3)


def test_a_leader_steps_down_for_a_higher_term():
    node = NodeState(node_id=1)
    node.become_candidate()
    node.become_leader()
    request = vote_request(term=2, candidate=3)
    assert node.handle_vote_request(request, EMPTY_LOG) == granted(term=2)
    assert node.role is Role.FOLLOWER


def test_a_candidate_refuses_a_rival_in_its_own_term():
    node = NodeState(node_id=1)
    node.become_candidate()
    rival = vote_request(term=1, candidate=2)
    assert node.handle_vote_request(rival, EMPTY_LOG) == refused(term=1)
    assert (node.role, node.voted_for) == (Role.CANDIDATE, 1)


def test_a_leader_refuses_a_request_in_its_own_term():
    node = NodeState(node_id=1)
    node.become_candidate()
    node.become_leader()
    rival = vote_request(term=1, candidate=2)
    assert node.handle_vote_request(rival, EMPTY_LOG) == refused(term=1)
    assert node.role is Role.LEADER


# --- Only for a log at least as up to date (ELECT-9, ELECT-10) ------------------------


@pytest.mark.parametrize(
    "candidate_last, own_last, expected",
    [
        pytest.param((3, 2), (2, 9), True, id="later-term-shorter"),
        pytest.param((2, 9), (3, 2), False, id="older-term-longer"),
        pytest.param((3, 5), (3, 4), True, id="same-term-longer"),
        pytest.param((3, 4), (3, 4), True, id="identical"),
        pytest.param((3, 3), (3, 4), False, id="same-term-shorter"),
        pytest.param((0, 0), (0, 0), True, id="both-empty"),
        pytest.param((0, 0), (1, 1), False, id="empty-candidate"),
        pytest.param((1, 1), (0, 0), True, id="empty-voter"),
    ],
)
def test_the_vote_follows_the_up_to_date_rule(candidate_last, own_last, expected):
    # Each pair is a log's (last term, last index).
    node = NodeState.reloaded(node_id=1, current_term=5, voted_for=None)
    last_log_term, last_log_index = candidate_last
    request = vote_request(
        term=6, candidate=2, last_log_term=last_log_term, last_log_index=last_log_index
    )
    own = LogPosition(term=own_last[0], index=own_last[1])
    response = node.handle_vote_request(request, own)
    assert response.vote_granted is expected
    assert node.voted_for == (2 if expected else None)


def test_refusing_a_stale_log_leaves_the_vote_free_for_another_candidate():
    node = NodeState.reloaded(node_id=1, current_term=5, voted_for=None)
    own_log = LogPosition(term=3, index=4)
    stale = vote_request(term=6, candidate=2, last_log_term=2, last_log_index=9)
    current = vote_request(term=6, candidate=3, last_log_term=3, last_log_index=4)
    assert node.handle_vote_request(stale, own_log) == refused(term=6)
    assert node.handle_vote_request(current, own_log) == granted(term=6)
    assert node.voted_for == 3


# --- Never two Candidates in one term -------------------------------------------------


@pytest.mark.parametrize("seed", range(50), ids=lambda seed: f"seed={seed}")
def test_no_term_ever_has_two_candidates_granted_a_vote(seed):
    # Requests from several Candidates, in earlier, current, and later terms, with logs of every
    # kind, repeated freely, and the node now and then starting its own election. After each:
    # no term has two Candidates, itself included, holding its vote, and every answer agrees
    # with the node's own state.
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
