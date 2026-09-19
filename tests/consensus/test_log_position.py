"""Tier 1 tests for LogPosition's up-to-date comparison, which decides votes (ELECT-10)."""

from raftkv.consensus import LogPosition


def test_later_last_log_term_wins_even_with_a_shorter_log():
    # A much shorter log whose last entry is from a later term still wins; "longest log wins"
    # alone would wrongly favor the voter here.
    candidate = LogPosition(term=5, index=3)
    voter = LogPosition(term=4, index=10)
    assert candidate.is_at_least_as_up_to_date_as(voter) is True


def test_earlier_last_log_term_loses_even_with_a_longer_log():
    candidate = LogPosition(term=4, index=10)
    voter = LogPosition(term=5, index=3)
    assert candidate.is_at_least_as_up_to_date_as(voter) is False


def test_equal_terms_fall_back_to_longer_log_wins():
    assert (
        LogPosition(term=5, index=10).is_at_least_as_up_to_date_as(LogPosition(term=5, index=9))
        is True
    )
    assert (
        LogPosition(term=5, index=9).is_at_least_as_up_to_date_as(LogPosition(term=5, index=10))
        is False
    )


def test_equal_terms_and_equal_length_is_up_to_date():
    # ELECT-9 grants to a Candidate "at least as" up to date: ties count.
    candidate = LogPosition(term=5, index=7)
    voter = LogPosition(term=5, index=7)
    assert candidate.is_at_least_as_up_to_date_as(voter) is True


def test_both_empty_logs_are_equally_up_to_date():
    empty = LogPosition(term=0, index=0)
    assert empty.is_at_least_as_up_to_date_as(empty) is True
