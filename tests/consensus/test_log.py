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
    assert repr(Log([LogEntry(term=1, command="a")])) == (
        "Log([LogEntry(term=1, command='a', cluster_time=0)])"
    )


# --- The empty entry a new Leader appends ---------------------------------------------


def test_the_empty_entry_carries_its_term_and_no_command():
    entry = LogEntry.empty(5)
    assert (entry.term, entry.command, entry.is_empty) == (5, "", True)


def test_an_entry_with_a_command_is_not_marked_empty():
    assert LogEntry(term=5, command="x=5").is_empty is False


# --- Cluster time (DD-32) -------------------------------------------------------------


@pytest.mark.parametrize("cluster_time", [1.0, "3", True, None])
def test_log_entry_rejects_a_cluster_time_that_is_not_an_int(cluster_time):
    with pytest.raises(TypeError):
        LogEntry(term=1, command="a", cluster_time=cluster_time)


def test_log_entry_rejects_a_negative_cluster_time():
    with pytest.raises(ValueError):
        LogEntry(term=1, command="a", cluster_time=-1)


def test_an_entry_built_without_a_cluster_time_carries_zero():
    assert LogEntry(term=1, command="a").cluster_time == 0


def test_the_empty_entry_carries_the_cluster_time_it_is_given():
    assert LogEntry.empty(term=3, cluster_time=40).cluster_time == 40


def test_last_cluster_time_is_the_last_entrys_and_zero_for_an_empty_log():
    assert Log().last_cluster_time == 0
    assert Log([LogEntry(1, "a", 5), LogEntry(1, "b", 9)]).last_cluster_time == 9
