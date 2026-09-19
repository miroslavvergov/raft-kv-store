"""Tier 1 unit tests for the pure Follower/Candidate/Leader state machine
(STATE-1 through STATE-7) — every valid STATE-3 edge, several invalid ones,
a forced higher-term observation confirming STATE-4/5/6 fire together, and
a Candidate recognizing the Leader of its own term (STATE-7).
"""

import pytest

from raftkv.consensus import IllegalTransition, NodeState, Role


def test_starts_as_follower_with_term_zero_and_no_vote():
    # STATE-1, STATE-2
    node = NodeState(node_id=1)
    assert node.role is Role.FOLLOWER
    assert node.current_term == 0
    assert node.voted_for is None


def test_follower_to_candidate_is_legal_and_bumps_term_and_votes_self():
    # STATE-3 edge Follower->Candidate; ELECT-3, ELECT-4
    node = NodeState(node_id=1)
    node.become_candidate()
    assert node.role is Role.CANDIDATE
    assert node.current_term == 1
    assert node.voted_for == 1


def test_candidate_to_candidate_is_legal_new_election_and_bumps_term_again():
    # STATE-3 edge Candidate->Candidate (a new election after a split vote)
    node = NodeState(node_id=1)
    node.become_candidate()
    node.become_candidate()
    assert node.role is Role.CANDIDATE
    assert node.current_term == 2
    assert node.voted_for == 1


def test_candidate_to_leader_is_legal():
    # STATE-3 edge Candidate->Leader; ELECT-11
    node = NodeState(node_id=1)
    node.become_candidate()
    node.become_leader()
    assert node.role is Role.LEADER


def test_follower_to_leader_is_illegal():
    node = NodeState(node_id=1)
    with pytest.raises(IllegalTransition):
        node.become_leader()
    assert node.role is Role.FOLLOWER  # rejected transition leaves state untouched


def test_leader_to_candidate_is_illegal():
    node = NodeState(node_id=1)
    node.become_candidate()
    node.become_leader()
    with pytest.raises(IllegalTransition):
        node.become_candidate()
    assert node.role is Role.LEADER


def test_leader_to_leader_is_illegal():
    node = NodeState(node_id=1)
    node.become_candidate()
    node.become_leader()
    with pytest.raises(IllegalTransition):
        node.become_leader()
    assert node.role is Role.LEADER


def test_handle_observed_term_does_nothing_for_a_lower_or_equal_term():
    node = NodeState(node_id=1)
    node.become_candidate()  # term becomes 1
    fired = node.handle_observed_term(1)
    assert fired is False
    assert node.role is Role.CANDIDATE
    assert node.current_term == 1
    assert node.voted_for == 1


def test_candidate_handles_higher_observed_term_and_reverts_to_follower():
    # STATE-4 (role), STATE-5 (currentTerm), STATE-6 (votedFor) all firing
    # together off the same trigger, as one forced RPC.
    node = NodeState(node_id=1)
    node.become_candidate()  # role=CANDIDATE, term=1, votedFor=1
    fired = node.handle_observed_term(5)
    assert fired is True
    assert node.role is Role.FOLLOWER  # STATE-4
    assert node.current_term == 5  # STATE-5
    assert node.voted_for is None  # STATE-6


def test_leader_handles_higher_observed_term_and_reverts_to_follower():
    node = NodeState(node_id=1)
    node.become_candidate()
    node.become_leader()
    fired = node.handle_observed_term(9)
    assert fired is True
    assert node.role is Role.FOLLOWER  # STATE-4
    assert node.current_term == 9  # STATE-5
    assert node.voted_for is None  # STATE-6


def test_follower_handles_higher_observed_term_with_no_role_to_give_up():
    # STATE-4 is scoped to Candidate/Leader only — a plain Follower has no
    # role to give up, but STATE-5/STATE-6 still apply unconditionally.
    node = NodeState(node_id=1)
    fired = node.handle_observed_term(3)
    assert fired is True
    assert node.role is Role.FOLLOWER
    assert node.current_term == 3
    assert node.voted_for is None


# --- STATE-2 with PERSIST-4/5: reloading after a restart -------------------


def test_reloaded_node_starts_as_follower_with_its_persisted_term_and_vote():
    node = NodeState.reloaded(node_id=1, current_term=5, voted_for=2)
    assert node.role is Role.FOLLOWER
    assert node.current_term == 5
    assert node.voted_for == 2


def test_reloaded_candidate_comes_back_as_follower_not_candidate():
    # A node that crashed mid-candidacy persisted its own vote (ELECT-5) —
    # it comes back remembering that vote, but in the Follower role.
    node = NodeState.reloaded(node_id=1, current_term=5, voted_for=1)
    assert node.role is Role.FOLLOWER
    assert node.voted_for == 1


def test_reloaded_first_boot_matches_a_brand_new_node():
    reloaded = NodeState.reloaded(node_id=1, current_term=0, voted_for=None)
    fresh = NodeState(node_id=1)
    assert (reloaded.role, reloaded.current_term, reloaded.voted_for) == (
        fresh.role,
        fresh.current_term,
        fresh.voted_for,
    )


def test_reloaded_node_handles_observed_terms_like_any_other():
    node = NodeState.reloaded(node_id=1, current_term=5, voted_for=2)
    assert node.handle_observed_term(5) is False  # already in term 5
    assert node.voted_for == 2  # vote for term 5 kept
    assert node.handle_observed_term(6) is True
    assert node.current_term == 6
    assert node.voted_for is None


def test_reloaded_node_becomes_candidate_from_its_reloaded_term():
    # ELECT-3 increments from the reloaded term, never restarts from 0.
    node = NodeState.reloaded(node_id=1, current_term=5, voted_for=2)
    node.become_candidate()
    assert node.current_term == 6
    assert node.voted_for == 1


# --- STATE-7: recognizing the Leader of an AppendEntries' term ---------------


def test_candidate_recognizing_a_same_term_leader_reverts_to_follower_keeping_term_and_vote():
    # Two Candidates ran in term 1 and the other one won. Its AppendEntries
    # carries term 1 — not higher — so STATE-4 never fires; STATE-7 does. The
    # self-vote stays: clearing it would free this node to vote again in term 1.
    node = NodeState(node_id=1)
    node.become_candidate()
    assert node.recognize_leader(1) is True
    assert (node.role, node.current_term, node.voted_for) == (Role.FOLLOWER, 1, 1)


def test_follower_recognizes_a_same_term_leader_and_stays_follower():
    node = NodeState.reloaded(node_id=1, current_term=4, voted_for=2)
    assert node.recognize_leader(4) is True
    assert (node.role, node.current_term, node.voted_for) == (Role.FOLLOWER, 4, 2)


def test_leader_of_a_higher_term_is_recognized_after_catching_up():
    node = NodeState(node_id=1)
    node.become_candidate()
    assert node.recognize_leader(3) is True
    assert (node.role, node.current_term, node.voted_for) == (Role.FOLLOWER, 3, None)


def test_leader_of_an_earlier_term_is_not_recognized():
    node = NodeState.reloaded(node_id=1, current_term=5, voted_for=2)
    node.become_candidate()  # term 6
    assert node.recognize_leader(5) is False
    assert (node.role, node.current_term, node.voted_for) == (Role.CANDIDATE, 6, 1)


def test_leader_does_not_recognize_another_leader_in_its_own_term():
    node = NodeState(node_id=1)
    node.become_candidate()
    node.become_leader()
    assert node.recognize_leader(1) is False
    assert (node.role, node.current_term) == (Role.LEADER, 1)


def test_leader_recognizes_the_leader_of_a_higher_term_and_steps_down():
    node = NodeState(node_id=1)
    node.become_candidate()
    node.become_leader()
    assert node.recognize_leader(2) is True
    assert (node.role, node.current_term, node.voted_for) == (Role.FOLLOWER, 2, None)
