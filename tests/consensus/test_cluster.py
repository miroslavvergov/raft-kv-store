"""Tier 1 tests for Cluster: membership, peers, and strict majorities, for every size 1 to 7.

ELECT-11, ELECT-12, NODE-8, DD-20.
"""

import itertools

import pytest

from raftkv.consensus import Cluster


def all_subsets(members):
    """Return every subset of `members`, the empty set and `members` itself included."""
    return [set(c) for r in range(len(members) + 1) for c in itertools.combinations(members, r)]


@pytest.mark.parametrize("size, majority", [(1, 1), (2, 2), (3, 2), (4, 3), (5, 3), (6, 4), (7, 4)])
def test_majority_is_the_smallest_count_above_half(size, majority):
    cluster = Cluster(range(1, size + 1))
    assert cluster.majority == majority
    assert 2 * cluster.majority > size
    assert 2 * (cluster.majority - 1) <= size


@pytest.mark.parametrize("size", range(1, 8))
def test_a_set_of_members_is_a_majority_exactly_when_it_reaches_that_count(size):
    members = list(range(1, size + 1))
    cluster = Cluster(members)
    for subset in all_subsets(members):
        assert cluster.is_majority(subset) == (len(subset) >= cluster.majority)


@pytest.mark.parametrize("size", range(1, 8))
def test_any_two_majorities_share_at_least_one_node(size):
    # The overlap every safety argument in Raft relies on, checked for every pair of majorities.
    members = list(range(1, size + 1))
    cluster = Cluster(members)
    majorities = [s for s in all_subsets(members) if cluster.is_majority(s)]
    assert majorities
    for first, second in itertools.product(majorities, repeat=2):
        assert first & second


def test_half_of_an_even_cluster_is_not_a_majority():
    # Two halves of four nodes need not share anyone, so neither may decide.
    cluster = Cluster([1, 2, 3, 4])
    assert not cluster.is_majority([1, 2])
    assert not cluster.is_majority([3, 4])


def test_a_repeated_id_counts_once():
    cluster = Cluster([1, 2, 3, 4, 5])
    assert not cluster.is_majority([1, 2, 2, 2, 2, 2])
    assert cluster.is_majority([1, 2, 3])


def test_ids_outside_the_cluster_count_for_nothing():
    cluster = Cluster([1, 2, 3])
    assert not cluster.is_majority([1, 4, 5, 6, 7])
    assert cluster.is_majority([1, 2, 4])


def test_peers_are_every_other_member():
    assert Cluster([1, 2, 3]).peers_of(2) == frozenset({1, 3})
    assert Cluster([5]).peers_of(5) == frozenset()


def test_peers_of_a_non_member_raises():
    with pytest.raises(ValueError):
        Cluster([1, 2, 3]).peers_of(4)


@pytest.mark.parametrize(
    "members",
    [
        [],
        [0, 1, 2],
        [-1, 2, 3],
        [1, 2, 2],
        [True, 2, 3],
        ["1", 2, 3],
        [1.0, 2, 3],
    ],
    ids=["empty", "zero", "negative", "duplicate", "bool", "string", "float"],
)
def test_invalid_membership_is_rejected(members):
    with pytest.raises(ValueError):
        Cluster(members)


def test_members_and_majority_cannot_be_assigned():
    cluster = Cluster([1, 2, 3])
    with pytest.raises(AttributeError):
        cluster.members = frozenset([1])
    with pytest.raises(AttributeError):
        cluster.majority = 1
