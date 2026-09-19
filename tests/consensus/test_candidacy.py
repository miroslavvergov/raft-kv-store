"""Tier 1 unit tests for Candidacy (ELECT-8, ELECT-11, ELECT-12): the
Candidate's own vote counted from the start, a majority reached at exactly
the right grant and not one before, refusals never counting, a voter's
repeated or changed answer counting once, answers from any other term
ignored — including a delayed grant from an earlier election — and a
randomized check against an independent model of the counting rules.
"""

import random

import pytest

from raftkv.consensus import Candidacy, Cluster

THREE_NODES = Cluster([1, 2, 3])
FIVE_NODES = Cluster([1, 2, 3, 4, 5])


# --- Starting point ------------------------------------------------------------


def test_starts_with_only_the_candidates_own_vote():
    candidacy = Candidacy(term=4, candidate_id=1, cluster=FIVE_NODES)
    assert candidacy.term == 4
    assert candidacy.votes_granted == frozenset({1})
    assert not candidacy.has_majority


def test_single_node_cluster_wins_with_its_own_vote():
    assert Candidacy(term=1, candidate_id=1, cluster=Cluster([1])).has_majority


def test_candidate_must_be_a_member_of_the_cluster():
    with pytest.raises(ValueError):
        Candidacy(term=1, candidate_id=9, cluster=THREE_NODES)


def test_term_and_votes_cannot_be_assigned():
    candidacy = Candidacy(term=4, candidate_id=1, cluster=THREE_NODES)
    with pytest.raises(AttributeError):
        candidacy.term = 5
    with pytest.raises(AttributeError):
        candidacy.votes_granted = frozenset({1, 2, 3})


# --- Reaching a majority (ELECT-11, ELECT-12) -----------------------------------


def test_three_nodes_win_on_the_first_grant_from_a_peer():
    candidacy = Candidacy(term=4, candidate_id=1, cluster=THREE_NODES)
    assert candidacy.record_vote(3, sent_in_term=4, granted=True) is True
    assert candidacy.votes_granted == frozenset({1, 3})
    assert candidacy.has_majority


def test_five_nodes_win_at_exactly_three_votes_and_not_one_before():
    candidacy = Candidacy(term=4, candidate_id=1, cluster=FIVE_NODES)
    candidacy.record_vote(2, sent_in_term=4, granted=True)
    assert not candidacy.has_majority  # 2 of 5
    candidacy.record_vote(4, sent_in_term=4, granted=True)
    assert candidacy.has_majority  # 3 of 5


def test_refusals_never_count_toward_a_majority():
    candidacy = Candidacy(term=4, candidate_id=1, cluster=FIVE_NODES)
    for voter in (2, 3, 4, 5):
        assert candidacy.record_vote(voter, sent_in_term=4, granted=False) is True
    assert candidacy.votes_granted == frozenset({1})
    assert candidacy.votes_refused == frozenset({2, 3, 4, 5})
    assert not candidacy.has_majority


# --- One answer per voter -------------------------------------------------------


def test_the_same_grant_delivered_again_counts_once():
    # A retried request, or a reply the network delivers twice, is still one
    # vote. Counting it again would let two nodes look like three.
    candidacy = Candidacy(term=4, candidate_id=1, cluster=FIVE_NODES)
    assert candidacy.record_vote(2, sent_in_term=4, granted=True) is True
    assert candidacy.record_vote(2, sent_in_term=4, granted=True) is False
    assert candidacy.record_vote(2, sent_in_term=4, granted=True) is False
    assert candidacy.votes_granted == frozenset({1, 2})
    assert not candidacy.has_majority


def test_a_voters_first_answer_is_the_one_kept():
    candidacy = Candidacy(term=4, candidate_id=1, cluster=THREE_NODES)
    candidacy.record_vote(2, sent_in_term=4, granted=False)
    assert candidacy.record_vote(2, sent_in_term=4, granted=True) is False
    assert candidacy.votes_granted == frozenset({1})


def test_the_candidate_cannot_count_itself_twice():
    candidacy = Candidacy(term=4, candidate_id=1, cluster=THREE_NODES)
    assert candidacy.record_vote(1, sent_in_term=4, granted=True) is False
    assert not candidacy.has_majority


# --- Only answers from this election count --------------------------------------


def test_a_delayed_grant_from_an_earlier_election_is_not_counted():
    # Term 4: node 1 asked node 2 for its vote, node 2 granted, and the grant
    # got stuck in the network. Term 5: node 1 runs again — and node 2 may
    # well have voted for node 3 this time. The old grant says nothing about
    # term 5; counting it would give node 1 a majority it does not have.
    candidacy = Candidacy(term=5, candidate_id=1, cluster=THREE_NODES)
    assert candidacy.record_vote(2, sent_in_term=4, granted=True) is False
    assert candidacy.votes_granted == frozenset({1})
    assert not candidacy.has_majority


def test_an_answer_from_any_other_term_is_ignored_not_only_an_earlier_one():
    candidacy = Candidacy(term=5, candidate_id=1, cluster=THREE_NODES)
    assert candidacy.record_vote(2, sent_in_term=6, granted=True) is False
    assert not candidacy.has_majority


def test_an_ignored_answer_does_not_use_up_the_voters_answer_for_this_term():
    candidacy = Candidacy(term=5, candidate_id=1, cluster=THREE_NODES)
    candidacy.record_vote(2, sent_in_term=4, granted=False)
    assert candidacy.record_vote(2, sent_in_term=5, granted=True) is True
    assert candidacy.has_majority


def test_unknown_voter_raises_in_this_term_but_is_ignored_in_another():
    candidacy = Candidacy(term=5, candidate_id=1, cluster=THREE_NODES)
    assert candidacy.record_vote(9, sent_in_term=4, granted=True) is False
    with pytest.raises(KeyError):
        candidacy.record_vote(9, sent_in_term=5, granted=True)
    assert candidacy.votes_granted == frozenset({1})


# --- Against an independent model -------------------------------------------------


@pytest.mark.parametrize("seed", range(50))
def test_counting_matches_a_simple_model_across_any_sequence_of_answers(seed):
    # Deterministic pseudo-random answers — from any member, for this term and
    # others, grants and refusals, repeated freely — checked after every one
    # against a direct restatement of the rules.
    rng = random.Random(seed)
    members = list(range(1, rng.randint(1, 7) + 1))
    cluster = Cluster(members)
    candidate = rng.choice(members)
    term = rng.randint(2, 9)
    candidacy = Candidacy(term=term, candidate_id=candidate, cluster=cluster)
    first_answers = {candidate: True}

    for _ in range(200):
        voter = rng.choice(members)
        sent_in_term = rng.choice([term - 1, term, term, term + 1])
        granted = rng.random() < 0.6

        recorded = candidacy.record_vote(voter, sent_in_term, granted)

        should_record = sent_in_term == term and voter not in first_answers
        assert recorded == should_record
        if should_record:
            first_answers[voter] = granted
        expected = frozenset(v for v, g in first_answers.items() if g)
        assert candidacy.votes_granted == expected
        assert candidacy.votes_refused == frozenset(first_answers) - expected
        assert candidacy.has_majority == (2 * len(expected) > len(members))
