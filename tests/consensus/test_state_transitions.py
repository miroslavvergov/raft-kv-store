"""Tier 1 unit tests for the pure Follower/Candidate/Leader state machine
(STATE-1 through STATE-6) — every valid STATE-3 edge, several invalid ones,
and a forced higher-term observation confirming STATE-4/5/6 fire together.
"""

import pytest

from raftkv.consensus import IllegalTransition, NodeState, Role


def test_starts_as_follower_with_term_zero_and_no_vote():
    # STATE-1, STATE-2
    node = NodeState(node_id="n1")
    assert node.role is Role.FOLLOWER
    assert node.current_term == 0
    assert node.voted_for is None


def test_follower_to_candidate_is_legal_and_bumps_term_and_votes_self():
    # STATE-3 edge Follower->Candidate; ELECT-3, ELECT-4
    node = NodeState(node_id="n1")
    node.become_candidate()
    assert node.role is Role.CANDIDATE
    assert node.current_term == 1
    assert node.voted_for == "n1"


def test_candidate_to_candidate_is_legal_new_election_and_bumps_term_again():
    # STATE-3 edge Candidate->Candidate (a new election after a split vote)
    node = NodeState(node_id="n1")
    node.become_candidate()
    node.become_candidate()
    assert node.role is Role.CANDIDATE
    assert node.current_term == 2
    assert node.voted_for == "n1"


def test_candidate_to_leader_is_legal():
    # STATE-3 edge Candidate->Leader; ELECT-11
    node = NodeState(node_id="n1")
    node.become_candidate()
    node.become_leader()
    assert node.role is Role.LEADER


def test_follower_to_leader_is_illegal():
    node = NodeState(node_id="n1")
    with pytest.raises(IllegalTransition):
        node.become_leader()
    assert node.role is Role.FOLLOWER  # rejected transition leaves state untouched


def test_leader_to_candidate_is_illegal():
    node = NodeState(node_id="n1")
    node.become_candidate()
    node.become_leader()
    with pytest.raises(IllegalTransition):
        node.become_candidate()
    assert node.role is Role.LEADER


def test_leader_to_leader_is_illegal():
    node = NodeState(node_id="n1")
    node.become_candidate()
    node.become_leader()
    with pytest.raises(IllegalTransition):
        node.become_leader()
    assert node.role is Role.LEADER


def test_try_catch_up_to_term_does_nothing_for_a_lower_or_equal_term():
    node = NodeState(node_id="n1")
    node.become_candidate()  # term becomes 1
    fired = node.try_catch_up_to_term(1)
    assert fired is False
    assert node.role is Role.CANDIDATE
    assert node.current_term == 1
    assert node.voted_for == "n1"


def test_candidate_catches_up_to_higher_term_and_reverts_to_follower():
    # STATE-4 (role), STATE-5 (currentTerm), STATE-6 (votedFor) all firing
    # together off the same trigger, as one forced RPC.
    node = NodeState(node_id="n1")
    node.become_candidate()  # role=CANDIDATE, term=1, votedFor="n1"
    fired = node.try_catch_up_to_term(5)
    assert fired is True
    assert node.role is Role.FOLLOWER  # STATE-4
    assert node.current_term == 5  # STATE-5
    assert node.voted_for is None  # STATE-6


def test_leader_catches_up_to_higher_term_and_reverts_to_follower():
    node = NodeState(node_id="n1")
    node.become_candidate()
    node.become_leader()
    fired = node.try_catch_up_to_term(9)
    assert fired is True
    assert node.role is Role.FOLLOWER  # STATE-4
    assert node.current_term == 9  # STATE-5
    assert node.voted_for is None  # STATE-6


def test_follower_catches_up_to_higher_term_with_no_role_to_give_up():
    # STATE-4 is scoped to Candidate/Leader only — a plain Follower has no
    # role to give up, but STATE-5/STATE-6 still apply unconditionally.
    node = NodeState(node_id="n1")
    fired = node.try_catch_up_to_term(3)
    assert fired is True
    assert node.role is Role.FOLLOWER
    assert node.current_term == 3
    assert node.voted_for is None
