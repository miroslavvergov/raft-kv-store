"""Tier 1 unit tests for the up-to-date vote comparison (ELECT-10), against
constructed (term, length) pairs — including the counterexample that shows
why "longest log wins" alone is the wrong rule.
"""

from raftkv.consensus import is_log_up_to_date


def test_later_last_log_term_wins_even_with_a_shorter_log():
    # The classic counterexample: a candidate whose log is much shorter but
    # whose last entry is from a later term must still win the comparison.
    # "Longest log wins" alone would (wrongly) favor the voter here.
    assert is_log_up_to_date(
        candidate_last_term=5, candidate_last_index=3,
        voter_last_term=4, voter_last_index=10,
    ) is True


def test_earlier_last_log_term_loses_even_with_a_longer_log():
    assert is_log_up_to_date(
        candidate_last_term=4, candidate_last_index=10,
        voter_last_term=5, voter_last_index=3,
    ) is False


def test_equal_terms_fall_back_to_longer_log_wins():
    assert is_log_up_to_date(
        candidate_last_term=5, candidate_last_index=10,
        voter_last_term=5, voter_last_index=9,
    ) is True
    assert is_log_up_to_date(
        candidate_last_term=5, candidate_last_index=9,
        voter_last_term=5, voter_last_index=10,
    ) is False


def test_equal_terms_and_equal_length_is_up_to_date():
    # ELECT-9 grants to a candidate "at least as" up to date — ties count.
    assert is_log_up_to_date(
        candidate_last_term=5, candidate_last_index=7,
        voter_last_term=5, voter_last_index=7,
    ) is True


def test_both_empty_logs_are_equally_up_to_date():
    assert is_log_up_to_date(
        candidate_last_term=0, candidate_last_index=0,
        voter_last_term=0, voter_last_index=0,
    ) is True
