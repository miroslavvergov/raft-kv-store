"""Tier 1 tests for the Log Matching check, the overwrite rule, and the Leader's repair loop.

REPL-5 through REPL-8, DD-21.
"""

import pytest

from raftkv.consensus import Log, LogEntry
from tests.divergent_logs import FOLLOWER_TERMS, LEADER_TERMS, make_log, repair

# --- The consistency check (REPL-5) ---------------------------------------------------


def test_matches_is_true_at_index_zero():
    assert Log().matches(prev_log_index=0, prev_log_term=0) is True
    assert make_log([1, 1]).matches(prev_log_index=0, prev_log_term=0) is True


def test_matches_is_false_past_the_end_of_the_log():
    log = make_log([1, 1, 1])
    assert log.matches(prev_log_index=4, prev_log_term=1) is False


def test_matches_is_false_when_the_term_at_that_index_differs():
    log = make_log([1, 1, 4])
    assert log.matches(prev_log_index=3, prev_log_term=1) is False


def test_matches_is_true_when_index_and_term_agree():
    log = make_log([1, 1, 4])
    assert log.matches(prev_log_index=3, prev_log_term=4) is True


# --- Overwrite conflicting entries, leave matching ones alone (REPL-8) ----------------


def test_after_append_entries_appends_past_end_of_log():
    log = make_log([1, 1])
    new_entries = [LogEntry(term=2, command="x")]
    result = log.after_append_entries(prev_log_index=2, entries=new_entries)
    assert [e.term for e in result] == [1, 1, 2]


def test_after_append_entries_overwrites_conflicting_tail():
    log = make_log([1, 1, 2, 2])  # the follower's extra term-2 entries at 3 and 4
    new_entries = [LogEntry(term=3, command="y")]  # the Leader's entry 3 is from term 3
    result = log.after_append_entries(prev_log_index=2, entries=new_entries)
    assert [e.term for e in result] == [1, 1, 3]


def test_after_append_entries_leaves_matching_entries_untouched():
    log = make_log([1, 1, 3])
    same_entry = log.entry_at(3)
    result = log.after_append_entries(prev_log_index=2, entries=[same_entry])
    assert result == log
    assert result.entry_at(3) is same_entry  # the very object: not replaced


def test_a_late_rpc_repeating_held_entries_does_not_delete_the_entries_after_them():
    # A Leader sent "after 1, here is 2", then "after 2, here is 3". The second arrived first,
    # so the Follower holds 1-3; the first arrives late. Entry 2 matches, so nothing changes:
    # entry 3 stays, though no incoming entry covers it and it may already be committed.
    log = make_log([1, 1, 1])
    late = log.after_append_entries(prev_log_index=1, entries=[log.entry_at(2)])
    assert late == log


def test_a_late_rpc_repeating_several_held_entries_keeps_every_entry_after_them():
    log = make_log([1, 1, 2, 2, 2])
    late = log.after_append_entries(prev_log_index=1, entries=log.entries_from(2)[:2])
    assert late == log


# --- Six diverged followers, repaired against one Leader ------------------------------


@pytest.mark.parametrize(
    "follower",
    [
        "missing_last_entry",
        "missing_last_six_entries",
        "conflicts_from_index_6",
        "conflicts_from_index_4",
    ],
)
def test_a_follower_without_extra_entries_ends_up_identical_to_the_leader(follower):
    # Missing entries are appended and diverging ones overwritten, leaving nothing of the
    # follower's own. Followers with extra entries are covered by the heartbeat tests below.
    repaired_log = repair(make_log(LEADER_TERMS), make_log(FOLLOWER_TERMS[follower])).repaired_log
    assert [e.term for e in repaired_log] == LEADER_TERMS


@pytest.mark.parametrize(
    "follower, agreed_at",
    [
        pytest.param("missing_last_entry", 9, id="missing_last_entry"),
        pytest.param("conflicts_from_index_6", 5, id="conflicts_from_index_6"),
        pytest.param("conflicts_from_index_4", 3, id="conflicts_from_index_4"),
    ],
)
def test_repair_agrees_at_the_last_index_both_logs_share(follower, agreed_at):
    result = repair(make_log(LEADER_TERMS), make_log(FOLLOWER_TERMS[follower]))
    assert result.agreed_at == agreed_at


def test_one_extra_stale_entry_survives_a_plain_heartbeat():
    # A heartbeat covering only indices both logs hold must not touch the follower's extra,
    # never-committed entry 11: nothing conflicts with it, and entries are deleted only from
    # the first conflict on.
    follower_log = make_log(FOLLOWER_TERMS["one_extra_stale_entry"])

    assert follower_log.matches(prev_log_index=10, prev_log_term=6)
    heartbeat_result = follower_log.after_append_entries(prev_log_index=10, entries=[])
    assert [e.term for e in heartbeat_result] == FOLLOWER_TERMS["one_extra_stale_entry"]

    # The Leader, still in term 8, then accepts a client command as its entry 11.
    new_entry = LogEntry(term=8, command="new-write")
    written_result = heartbeat_result.after_append_entries(prev_log_index=10, entries=[new_entry])
    assert [e.term for e in written_result] == LEADER_TERMS + [8]


def test_two_extra_stale_entries_survive_a_plain_heartbeat():
    # As with one extra entry, but a conflict at 11 removes 12 too: everything from the first
    # conflict on goes, not just the conflicting entry.
    follower_log = make_log(FOLLOWER_TERMS["two_extra_stale_entries"])

    assert follower_log.matches(prev_log_index=10, prev_log_term=6)
    heartbeat_result = follower_log.after_append_entries(prev_log_index=10, entries=[])
    assert [e.term for e in heartbeat_result] == FOLLOWER_TERMS["two_extra_stale_entries"]

    new_entry = LogEntry(term=8, command="new-write")
    written_result = heartbeat_result.after_append_entries(prev_log_index=10, entries=[new_entry])
    assert [e.term for e in written_result] == LEADER_TERMS + [8]


# --- last_index and last_term ---------------------------------------------------------


def test_last_index_and_term_on_empty_and_nonempty_logs():
    assert Log().last_index == 0
    assert Log().last_term == 0
    log = make_log(LEADER_TERMS)
    assert log.last_index == 10
    assert log.last_term == 6


# --- A command is an opaque string (DD-21) --------------------------------------------


def test_log_entry_rejects_a_non_string_command():
    with pytest.raises(TypeError):
        LogEntry(term=1, command=5)
    with pytest.raises(TypeError):
        LogEntry(term=1, command=("put", "x", 1))


# --- first_differing_index: the part of a log that actually changed -------------------


def test_first_differing_index_is_none_for_identical_logs():
    assert make_log([1, 1, 2]).first_differing_index(make_log([1, 1, 2])) is None
    assert Log().first_differing_index(Log()) is None


def test_first_differing_index_finds_first_conflicting_position():
    assert make_log([1, 1, 2, 2]).first_differing_index(make_log([1, 1, 3])) == 3


def test_first_differing_index_when_one_log_is_a_prefix_of_the_other():
    assert make_log([1, 1]).first_differing_index(make_log([1, 1, 2])) == 3
    assert make_log([1, 1, 2]).first_differing_index(make_log([1, 1])) == 3
    assert Log().first_differing_index(make_log([1])) == 1


def test_first_differing_index_compares_commands_not_just_terms():
    ours = Log([LogEntry(term=1, command="a")])
    theirs = Log([LogEntry(term=1, command="b")])
    assert ours.first_differing_index(theirs) == 1


def test_first_differing_index_is_none_after_a_heartbeat_keeps_a_stale_entry():
    # A stale entry kept through a heartbeat is not a change, so nothing needs rewriting.
    follower_log = make_log(FOLLOWER_TERMS["one_extra_stale_entry"])
    after_heartbeat = follower_log.after_append_entries(prev_log_index=10, entries=[])
    assert follower_log.first_differing_index(after_heartbeat) is None


def test_first_differing_index_after_repairing_a_conflict_from_index_4_is_4():
    leader_log = make_log(LEADER_TERMS)
    follower_log = make_log(FOLLOWER_TERMS["conflicts_from_index_4"])
    repaired_log = repair(leader_log, follower_log).repaired_log
    assert follower_log.first_differing_index(repaired_log) == 4
