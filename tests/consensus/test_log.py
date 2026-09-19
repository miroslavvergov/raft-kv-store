"""Tier 1 tests for Log's 1-based access (entry_at, term_at, entries_from), its bounds, and repr."""

import pytest

from raftkv.consensus import Log, LogEntry

LOG = Log(
    [
        LogEntry(term=1, command="a"),
        LogEntry(term=1, command="b"),
        LogEntry(term=3, command="c"),
    ]
)


@pytest.mark.parametrize("index, command", [(1, "a"), (2, "b"), (3, "c")])
def test_entry_at_counts_from_one(index, command):
    assert LOG.entry_at(index).command == command


@pytest.mark.parametrize("index", [0, 4, -1])
def test_entry_at_outside_one_to_last_index_raises(index):
    with pytest.raises(IndexError):
        LOG.entry_at(index)


def test_entry_at_on_an_empty_log_raises():
    with pytest.raises(IndexError):
        Log().entry_at(1)


@pytest.mark.parametrize("index, term", [(0, 0), (1, 1), (3, 3)])
def test_term_at_is_the_entrys_term_and_zero_at_index_zero(index, term):
    assert LOG.term_at(index) == term


def test_term_at_zero_on_an_empty_log_is_zero():
    assert Log().term_at(0) == 0


@pytest.mark.parametrize("index", [4, -1])
def test_term_at_outside_zero_to_last_index_raises(index):
    with pytest.raises(IndexError):
        LOG.term_at(index)


@pytest.mark.parametrize("index, commands", [(1, ["a", "b", "c"]), (3, ["c"]), (4, [])])
def test_entries_from_runs_to_the_end_and_is_empty_just_past_it(index, commands):
    assert [e.command for e in LOG.entries_from(index)] == commands


def test_entries_from_one_on_an_empty_log_is_empty():
    assert Log().entries_from(1) == []


@pytest.mark.parametrize("index", [0, 5, -1])
def test_entries_from_outside_one_to_last_index_plus_one_raises(index):
    with pytest.raises(IndexError):
        LOG.entries_from(index)


def test_entries_from_returns_a_copy():
    LOG.entries_from(1).clear()
    assert LOG.last_index == 3


def test_matches_just_past_the_last_entry_is_false_rather_than_an_error():
    assert LOG.matches(prev_log_index=4, prev_log_term=3) is False


@pytest.mark.parametrize("term", ["1", 1.0, None, True], ids=["str", "float", "none", "bool"])
def test_log_entry_rejects_a_term_that_is_not_an_int(term):
    with pytest.raises(TypeError):
        LogEntry(term=term, command="x")


@pytest.mark.parametrize("term", [0, -1])
def test_log_entry_rejects_a_term_below_one(term):
    with pytest.raises(ValueError):
        LogEntry(term=term, command="x")


def test_repr_shows_the_entries():
    assert repr(Log([LogEntry(term=1, command="a")])) == "Log([LogEntry(term=1, command='a')])"
