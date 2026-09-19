"""Tier 1 unit tests for Leadership (DD-25): every Follower's progress reset
when a leadership starts (REPL-14, REPL-15), replies counted only if they
answer an RPC sent in this leadership's term (REPL-16), each reply touching
only its own Follower, and one leadership repairing several diverged
Followers at once.
"""

import random

import pytest

from raftkv.consensus import Leadership
from tests.divergent_logs import FOLLOWER_TERMS, LEADER_TERMS, make_log

TERM = 7
FOLLOWERS = [2, 3, 4, 5]


def snapshot(leadership):
    """Every Follower's (next_index, match_index), for before/after comparisons."""
    return {
        f: (leadership.next_index(f), leadership.match_index(f))
        for f in leadership.followers
    }


def repair_all(leadership, leader_log, follower_logs):
    """Probe, back off, and confirm every Follower, taking turns, as a Leader would.

    Returns how many rejections each Follower took before its log was
    accepted. Each rejection lowers that Follower's nextIndex by one, and a
    probe at index 1 always succeeds, so no Follower can need more
    rejections than the Leader has entries; needing more fails the test
    instead of looping forever.
    """
    rejections = {f: 0 for f in follower_logs}
    pending = set(follower_logs)
    while pending:
        for follower in sorted(pending):
            prev_log_index = leadership.next_index(follower) - 1
            prev_log_term = leader_log[prev_log_index - 1].term if prev_log_index > 0 else 0
            if follower_logs[follower].matches(prev_log_index, prev_log_term):
                entries = leader_log[prev_log_index:]
                leadership.record_success(follower, leadership.term, prev_log_index, len(entries))
                pending.discard(follower)
            else:
                leadership.record_rejection(follower, leadership.term)
                rejections[follower] += 1
                assert rejections[follower] <= leader_log.last_index, (
                    f"follower {follower} never reached an index where the logs agree"
                )
    return rejections


# --- Starting a leadership resets every Follower (REPL-14, REPL-15) --------


def test_every_follower_starts_one_past_the_leaders_last_index_with_nothing_matched():
    leadership = Leadership(term=TERM, followers=FOLLOWERS, last_log_index=11)
    assert leadership.term == TERM
    assert leadership.followers == frozenset(FOLLOWERS)
    assert snapshot(leadership) == {f: (12, 0) for f in FOLLOWERS}


def test_an_empty_leader_log_starts_every_follower_at_index_1():
    leadership = Leadership(term=1, followers=FOLLOWERS, last_log_index=0)
    assert snapshot(leadership) == {f: (1, 0) for f in FOLLOWERS}


def test_a_single_node_cluster_has_no_followers():
    leadership = Leadership(term=1, followers=[], last_log_index=5)
    assert leadership.followers == frozenset()


def test_term_and_followers_cannot_be_assigned_directly():
    leadership = Leadership(term=TERM, followers=FOLLOWERS, last_log_index=11)
    with pytest.raises(AttributeError):
        leadership.term = 8
    with pytest.raises(AttributeError):
        leadership.followers = frozenset([9])


def test_follower_records_are_never_handed_out():
    # Nothing public exposes a FollowerProgress: the only way to change one
    # is record_success / record_rejection, which check the term first.
    leadership = Leadership(term=TERM, followers=FOLLOWERS, last_log_index=11)
    public = {name for name in dir(leadership) if not name.startswith("_")}
    assert public == {
        "term",
        "followers",
        "next_index",
        "match_index",
        "record_success",
        "record_rejection",
    }


def test_looking_up_an_unknown_follower_raises():
    leadership = Leadership(term=TERM, followers=FOLLOWERS, last_log_index=11)
    with pytest.raises(KeyError):
        leadership.next_index(99)
    with pytest.raises(KeyError):
        leadership.match_index(99)


# --- Each election starts fresh -------------------------------------------------


def test_winning_again_starts_fresh_even_if_this_node_led_before():
    # In term 5 this node led with 12 entries and confirmed follower 2 through
    # index 12. Its log was then cut back to 11 entries by another Leader, and
    # it won again in term 7. Nothing from term 5 carries over, and nextIndex
    # is recalculated from the log it holds now — 12, not 13.
    old = Leadership(term=5, followers=[2], last_log_index=12)
    old.record_success(2, sent_in_term=5, prev_log_index=12, entry_count=0)
    assert (old.next_index(2), old.match_index(2)) == (13, 12)

    new = Leadership(term=7, followers=[2], last_log_index=11)
    assert (new.next_index(2), new.match_index(2)) == (12, 0)


def test_a_late_reply_reaching_the_old_leadership_cannot_affect_the_new_one():
    old = Leadership(term=5, followers=[2], last_log_index=12)
    new = Leadership(term=7, followers=[2], last_log_index=11)
    old.record_success(2, sent_in_term=5, prev_log_index=9, entry_count=3)
    assert (new.next_index(2), new.match_index(2)) == (12, 0)


# --- Only replies from this leadership's term count (REPL-16) --------------


def test_success_from_this_term_is_counted_for_that_follower_only():
    leadership = Leadership(term=TERM, followers=FOLLOWERS, last_log_index=11)
    before = snapshot(leadership)
    assert leadership.record_success(3, TERM, prev_log_index=9, entry_count=2) is True
    after = snapshot(leadership)
    assert after[3] == (12, 11)
    assert {f: v for f, v in after.items() if f != 3} == {f: v for f, v in before.items() if f != 3}


def test_rejection_from_this_term_is_counted_for_that_follower_only():
    leadership = Leadership(term=TERM, followers=FOLLOWERS, last_log_index=11)
    before = snapshot(leadership)
    assert leadership.record_rejection(4, TERM) is True
    after = snapshot(leadership)
    assert after[4] == (11, 0)
    assert {f: v for f, v in after.items() if f != 4} == {f: v for f, v in before.items() if f != 4}


def test_success_sent_in_an_earlier_term_is_ignored():
    leadership = Leadership(term=TERM, followers=FOLLOWERS, last_log_index=11)
    before = snapshot(leadership)
    assert leadership.record_success(2, 5, prev_log_index=9, entry_count=3) is False
    assert snapshot(leadership) == before


def test_rejection_sent_in_an_earlier_term_is_ignored():
    leadership = Leadership(term=TERM, followers=FOLLOWERS, last_log_index=11)
    before = snapshot(leadership)
    assert leadership.record_rejection(2, 5) is False
    assert snapshot(leadership) == before


def test_reply_from_any_other_term_is_ignored_not_only_an_earlier_one():
    leadership = Leadership(term=TERM, followers=FOLLOWERS, last_log_index=11)
    before = snapshot(leadership)
    assert leadership.record_success(2, TERM + 1, prev_log_index=11, entry_count=0) is False
    assert leadership.record_rejection(2, TERM + 1) is False
    assert snapshot(leadership) == before


def test_a_stale_reply_is_ignored_before_the_follower_is_even_looked_up():
    leadership = Leadership(term=TERM, followers=FOLLOWERS, last_log_index=11)
    assert leadership.record_success(99, 5, prev_log_index=9, entry_count=3) is False
    assert leadership.record_rejection(99, 5) is False
    with pytest.raises(KeyError):
        leadership.record_success(99, TERM, prev_log_index=9, entry_count=3)
    with pytest.raises(KeyError):
        leadership.record_rejection(99, TERM)


def test_late_reply_from_an_earlier_leadership_is_not_counted():
    # The situation this rule exists for:
    #   term 5: this node, as Leader, sent entries 10-12 to follower 2, which
    #           accepted them — but the "success" reply got stuck in the
    #           network.
    #   term 6: another Leader overwrote entry 10 on this node (and left the
    #           follower's copies stale).
    #   term 7: this node is Leader again, with entry 10 from term 6 and a new
    #           entry 11 from term 7 — neither of which the follower has seen.
    leader_log = make_log([1] * 9 + [6, 7])
    follower_log = make_log([1] * 9 + [5, 5, 5])
    leadership = Leadership(term=7, followers=[2], last_log_index=leader_log.last_index)

    # The stuck term-5 reply finally arrives. It would claim the follower
    # matches through index 12 — past the end of the Leader's own log.
    assert 9 + 3 > leader_log.last_index
    assert leadership.record_success(2, sent_in_term=5, prev_log_index=9, entry_count=3) is False
    assert leadership.match_index(2) == 0

    # The follower is only counted once it genuinely takes entries 10-11 in
    # term 7: two rejections walk nextIndex back to where the logs agree.
    rejections = repair_all(leadership, leader_log, {2: follower_log})
    assert rejections == {2: 2}
    assert (leadership.match_index(2), leadership.next_index(2)) == (11, 12)
    assert follower_log.after_append_entries(9, leader_log[9:]) == leader_log


# --- Invariants under any mix of followers, terms, and replies --------------


@pytest.mark.parametrize("seed", range(50))
def test_invariants_hold_across_any_sequence_of_replies(seed):
    # Deterministic pseudo-random replies — from several followers, from this
    # term and others, successes and rejections, in any order — checking
    # every invariant after every one.
    rng = random.Random(seed)
    leadership = Leadership(term=TERM, followers=[2, 3, 4], last_log_index=rng.randint(0, 20))
    for _ in range(300):
        before = snapshot(leadership)
        follower = rng.choice([2, 3, 4])
        sent_in_term = rng.choice([TERM - 2, TERM - 1, TERM, TERM, TERM, TERM + 1])
        if rng.random() < 0.5:
            counted = leadership.record_success(
                follower, sent_in_term, rng.randint(0, 25), rng.randint(0, 5)
            )
        else:
            counted = leadership.record_rejection(follower, sent_in_term)
        after = snapshot(leadership)

        assert counted == (sent_in_term == TERM)
        for f, (next_index, match_index) in after.items():
            assert match_index < next_index
            assert match_index >= before[f][1]
            if f != follower or not counted:
                assert after[f] == before[f]


# --- One leadership, six diverged followers, repaired at once ------------------


def test_one_leadership_repairs_six_diverged_followers_taking_turns():
    names = [
        "missing_last_entry",
        "missing_last_six_entries",
        "one_extra_stale_entry",
        "two_extra_stale_entries",
        "conflicts_from_index_6",
        "conflicts_from_index_4",
    ]
    leader_log = make_log(LEADER_TERMS)
    follower_logs = {node: make_log(FOLLOWER_TERMS[name]) for node, name in enumerate(names, 2)}
    leadership = Leadership(term=8, followers=follower_logs, last_log_index=leader_log.last_index)

    rejections = repair_all(leadership, leader_log, follower_logs)

    assert rejections == dict(zip(range(2, 8), [1, 6, 0, 0, 5, 7]))
    assert snapshot(leadership) == {node: (11, 10) for node in follower_logs}
