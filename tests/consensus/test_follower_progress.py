"""Tier 1 tests for FollowerProgress, a Leader's nextIndex and matchIndex for one follower.

DD-25, REPL-6, REPL-7, REPL-15, REPL-16, REPL-17. Leadership decides which replies reach it.
"""

import random

import pytest

from raftkv.consensus import FollowerProgress
from tests.support.divergent_logs import FOLLOWER_TERMS, LEADER_TERMS, make_log, repair

# --- Starting state -------------------------------------------------------------------


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


# --- record_success: confirmed, never assumed -----------------------------------------


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
    progress.record_success(prev_log_index=3, entry_count=2)  # older, late reply: through 5
    assert (progress.match_index, progress.next_index) == (10, 11)


def test_duplicate_success_changes_nothing():
    progress = FollowerProgress(next_index=4)
    progress.record_success(prev_log_index=3, entry_count=7)
    progress.record_success(prev_log_index=3, entry_count=7)
    assert (progress.match_index, progress.next_index) == (10, 11)


def test_success_never_lowers_next_index():
    # A new Leader's nextIndex starts optimistic (last index + 1). Confirming a shorter prefix
    # raises matchIndex without pulling nextIndex back.
    progress = FollowerProgress(next_index=11)
    progress.record_success(prev_log_index=3, entry_count=2)
    assert (progress.match_index, progress.next_index) == (5, 11)


# --- record_rejection: back off, floored at match_index + 1 (REPL-6) ------------------


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


# --- Invariants under any order of replies --------------------------------------------


@pytest.mark.parametrize("seed", range(50), ids=lambda seed: f"seed={seed}")
def test_invariants_hold_across_any_sequence_of_replies(seed):
    # Successes for arbitrary RPCs and rejections, in any order, with every invariant checked
    # after each one.
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


# --- The whole repair loop, against six diverged followers ----------------------------


@pytest.mark.parametrize(
    "follower, expected_rejections",
    [
        pytest.param("missing_last_entry", 1, id="missing_last_entry"),
        pytest.param("missing_last_six_entries", 6, id="missing_last_six_entries"),
        pytest.param("one_extra_stale_entry", 0, id="one_extra_stale_entry"),
        pytest.param("two_extra_stale_entries", 0, id="two_extra_stale_entries"),
        pytest.param("conflicts_from_index_6", 5, id="conflicts_from_index_6"),
        pytest.param("conflicts_from_index_4", 7, id="conflicts_from_index_4"),
    ],
)
def test_repair_confirms_the_leaders_whole_log(follower, expected_rejections):
    result = repair(make_log(LEADER_TERMS), make_log(FOLLOWER_TERMS[follower]))
    assert result.rejections == expected_rejections
    assert (result.progress.match_index, result.progress.next_index) == (10, 11)


@pytest.mark.parametrize("follower", ["one_extra_stale_entry", "two_extra_stale_entries"])
def test_match_index_ignores_a_followers_extra_unconfirmed_entries(follower):
    # These followers hold 11 and 12 entries, but only the Leader's 10 are confirmed to match.
    follower_log = make_log(FOLLOWER_TERMS[follower])
    result = repair(make_log(LEADER_TERMS), follower_log)
    assert len(follower_log) > 10
    assert result.progress.match_index == 10
