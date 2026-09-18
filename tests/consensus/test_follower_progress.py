"""Tier 1 unit tests for FollowerProgress: nextIndex backoff (REPL-6, REPL-7)
and a matchIndex that only ever reflects acknowledged replication — what the
Leader will count toward APPLY-1's majority — including replies that arrive
late, twice, or out of order, and the full repair loop against six follower
logs that each diverge from the leader's in a different way.
"""

import random

import pytest

from raftkv.consensus import FollowerProgress
from tests.divergent_logs import FOLLOWER_TERMS, LEADER_TERMS, make_log


def repair(leader_log, follower_log):
    """Probe, back off, and confirm, as a Leader would.

    Returns the follower's progress once an AppendEntries is accepted, and
    how many rejections it took to get there.
    """
    progress = FollowerProgress(next_index=leader_log.last_index + 1)
    rejections = 0
    while True:
        prev_log_index = progress.next_index - 1
        prev_log_term = leader_log[prev_log_index - 1].term if prev_log_index > 0 else 0
        if follower_log.matches(prev_log_index, prev_log_term):
            entries = leader_log[prev_log_index:]
            progress.record_success(prev_log_index, len(entries))
            return progress, rejections
        progress.record_rejection()
        rejections += 1


# --- Starting state -----------------------------------------------------------


def test_starts_at_the_given_next_index_with_nothing_matched():
    progress = FollowerProgress(next_index=11)
    assert (progress.next_index, progress.match_index) == (11, 0)


def test_next_and_match_index_cannot_be_assigned_directly():
    # The only way to move matchIndex is to record an acknowledged success.
    progress = FollowerProgress(next_index=11)
    with pytest.raises(AttributeError):
        progress.match_index = 10
    with pytest.raises(AttributeError):
        progress.next_index = 5


def test_match_index_stays_zero_until_a_success_is_recorded():
    progress = FollowerProgress(next_index=11)
    for _ in range(5):
        progress.record_rejection()
    assert progress.match_index == 0


# --- record_success: confirmed, never assumed ---------------------------------


def test_success_confirms_through_the_last_entry_that_rpc_carried():
    progress = FollowerProgress(next_index=4)
    progress.record_success(prev_log_index=3, entry_count=7)
    assert (progress.match_index, progress.next_index) == (10, 11)


def test_heartbeat_success_confirms_through_prev_log_index():
    progress = FollowerProgress(next_index=11)
    progress.record_success(prev_log_index=10, entry_count=0)
    assert (progress.match_index, progress.next_index) == (10, 11)


def test_late_success_for_an_older_rpc_does_not_lower_match_index():
    progress = FollowerProgress(next_index=4)
    progress.record_success(prev_log_index=3, entry_count=7)  # newer reply: through 10
    progress.record_success(prev_log_index=3, entry_count=2)  # older reply, arriving late: through 5
    assert (progress.match_index, progress.next_index) == (10, 11)


def test_duplicate_success_changes_nothing():
    progress = FollowerProgress(next_index=4)
    progress.record_success(prev_log_index=3, entry_count=7)
    progress.record_success(prev_log_index=3, entry_count=7)
    assert (progress.match_index, progress.next_index) == (10, 11)


def test_success_never_lowers_next_index():
    # A new Leader's nextIndex starts optimistic (last index + 1). Confirming
    # a shorter prefix raises matchIndex without pulling nextIndex back.
    progress = FollowerProgress(next_index=11)
    progress.record_success(prev_log_index=3, entry_count=2)
    assert (progress.match_index, progress.next_index) == (5, 11)


# --- record_rejection: REPL-6 backoff, floored at match_index + 1 -----------


def test_rejection_lowers_next_index_by_one():
    progress = FollowerProgress(next_index=5)
    progress.record_rejection()
    assert progress.next_index == 4


def test_rejection_floors_at_one_when_nothing_is_matched():
    progress = FollowerProgress(next_index=1)
    progress.record_rejection()
    assert progress.next_index == 1


def test_late_rejection_never_goes_below_match_index_plus_one():
    progress = FollowerProgress(next_index=4)
    progress.record_success(prev_log_index=3, entry_count=2)  # confirmed through 5
    assert progress.next_index == 6
    progress.record_rejection()  # a late reply to an older probe
    assert (progress.match_index, progress.next_index) == (5, 6)


def test_backoff_then_success_resumes_just_past_the_confirmed_index():
    progress = FollowerProgress(next_index=11)
    progress.record_rejection()
    progress.record_rejection()
    assert progress.next_index == 9
    progress.record_success(prev_log_index=8, entry_count=2)
    assert (progress.match_index, progress.next_index) == (10, 11)


# --- Invariants under any order of replies ------------------------------------


@pytest.mark.parametrize("seed", range(50))
def test_invariants_hold_across_any_sequence_of_replies(seed):
    # Deterministic pseudo-random replies — successes for arbitrary RPCs and
    # rejections, in any order — checking every invariant after every one.
    rng = random.Random(seed)
    progress = FollowerProgress(next_index=rng.randint(1, 20))
    for _ in range(200):
        match_before, next_before = progress.match_index, progress.next_index
        if rng.random() < 0.5:
            prev_log_index, entry_count = rng.randint(0, 20), rng.randint(0, 5)
            progress.record_success(prev_log_index, entry_count)
            assert progress.match_index == max(match_before, prev_log_index + entry_count)
            assert progress.next_index >= next_before
        else:
            progress.record_rejection()
            assert progress.match_index == match_before
            assert progress.next_index in (next_before - 1, next_before)
        assert progress.match_index >= match_before
        assert progress.match_index < progress.next_index


# --- The whole repair loop, against six diverged followers ---------------------


@pytest.mark.parametrize(
    "follower, expected_rejections",
    [
        ("missing_last_entry", 1),
        ("missing_last_six_entries", 6),
        ("one_extra_stale_entry", 0),
        ("two_extra_stale_entries", 0),
        ("conflicts_from_index_6", 5),
        ("conflicts_from_index_4", 7),
    ],
)
def test_repair_confirms_the_leaders_whole_log(follower, expected_rejections):
    progress, rejections = repair(make_log(LEADER_TERMS), make_log(FOLLOWER_TERMS[follower]))
    assert rejections == expected_rejections
    assert (progress.match_index, progress.next_index) == (10, 11)


@pytest.mark.parametrize("follower", ["one_extra_stale_entry", "two_extra_stale_entries"])
def test_match_index_does_not_count_a_followers_extra_unconfirmed_entries(follower):
    # These followers hold 11 and 12 entries, but only the leader's 10 have
    # been confirmed to match — their stale extra entries don't count.
    follower_log = make_log(FOLLOWER_TERMS[follower])
    progress, _ = repair(make_log(LEADER_TERMS), follower_log)
    assert len(follower_log) > 10
    assert progress.match_index == 10
