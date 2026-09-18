"""Tier 1 unit tests for the up-to-date vote comparison (ELECT-10), against
constructed LogPosition pairs — including the counterexample that shows
why "longest log wins" alone is the wrong rule.
"""

from raftkv.consensus import LogPosition


def test_later_last_log_term_wins_even_with_a_shorter_log():
    # The classic counterexample: a candidate whose log is much shorter but
    # whose last entry is from a later term must still win the comparison.
    # "Longest log wins" alone would (wrongly) favor the voter here.
    candidate = LogPosition(term=5, index=3)
    voter = LogPosition(term=4, index=10)
    assert candidate.is_at_least_as_up_to_date_as(voter) is True


def test_earlier_last_log_term_loses_even_with_a_longer_log():
    candidate = LogPosition(term=4, index=10)
    voter = LogPosition(term=5, index=3)
    assert candidate.is_at_least_as_up_to_date_as(voter) is False


def test_equal_terms_fall_back_to_longer_log_wins():
    assert LogPosition(term=5, index=10).is_at_least_as_up_to_date_as(
        LogPosition(term=5, index=9)
    ) is True
    assert LogPosition(term=5, index=9).is_at_least_as_up_to_date_as(
        LogPosition(term=5, index=10)
    ) is False


def test_equal_terms_and_equal_length_is_up_to_date():
    # ELECT-9 grants to a candidate "at least as" up to date — ties count.
    a = LogPosition(term=5, index=7)
    b = LogPosition(term=5, index=7)
    assert a.is_at_least_as_up_to_date_as(b) is True


def test_both_empty_logs_are_equally_up_to_date():
    empty = LogPosition(term=0, index=0)
    assert empty.is_at_least_as_up_to_date_as(empty) is True
