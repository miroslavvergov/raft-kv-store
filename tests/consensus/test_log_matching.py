"""Tier 1 unit tests for the Log Matching consistency check and repair loop
(REPL-5, REPL-6, REPL-7, REPL-8), against hand-constructed logs — including
six follower logs that each diverge from one leader's log in a different way.
"""

import pytest

from raftkv.consensus import FollowerProgress, Log, LogEntry
from tests.divergent_logs import FOLLOWER_TERMS, LEADER_TERMS, make_log


def reconcile(leader_log, follower_log):
    """Drive the REPL-6/REPL-7 retry loop against `follower_log`: starting
    from nextIndex just past the leader's last entry, decrement on rejection
    until REPL-5's check passes, then apply REPL-8's overwrite. Returns the
    reconciled follower log and the prevLogIndex the retry loop settled on.
    """
    progress = FollowerProgress(next_index=leader_log.last_index + 1)
    while True:
        prev_log_index = progress.next_index - 1
        prev_log_term = leader_log[prev_log_index - 1].term if prev_log_index > 0 else 0
        if follower_log.matches(prev_log_index, prev_log_term):
            entries = leader_log[prev_log_index:]
            progress.record_success(prev_log_index, len(entries))
            return (
                follower_log.after_append_entries(prev_log_index, entries),
                prev_log_index,
            )
        progress.record_rejection()


# --- REPL-5: the consistency check itself -----------------------------------


def test_log_matches_when_prev_log_index_is_zero():
    assert Log().matches(prev_log_index=0, prev_log_term=0) is True
    assert make_log([1, 1]).matches(prev_log_index=0, prev_log_term=0) is True


def test_log_matches_rejects_when_index_beyond_log():
    log = make_log([1, 1, 1])
    assert log.matches(prev_log_index=4, prev_log_term=1) is False


def test_log_matches_rejects_when_term_differs_at_that_index():
    log = make_log([1, 1, 4])
    assert log.matches(prev_log_index=3, prev_log_term=1) is False


def test_log_matches_accepts_when_index_and_term_agree():
    log = make_log([1, 1, 4])
    assert log.matches(prev_log_index=3, prev_log_term=4) is True


# --- REPL-8: overwrite conflicting entries, leave matching ones alone -------


def test_after_append_entries_appends_past_end_of_log():
    log = make_log([1, 1])
    new_entries = [LogEntry(term=2, command="x")]
    result = log.after_append_entries(prev_log_index=2, entries=new_entries)
    assert [e.term for e in result] == [1, 1, 2]


def test_after_append_entries_overwrites_conflicting_tail():
    log = make_log([1, 1, 2, 2])  # follower has extra term-2 entries at 3, 4
    new_entries = [LogEntry(term=3, command="y")]  # leader's real entry 3 is term 3
    result = log.after_append_entries(prev_log_index=2, entries=new_entries)
    assert [e.term for e in result] == [1, 1, 3]


def test_after_append_entries_leaves_matching_entries_untouched():
    log = make_log([1, 1, 3])
    same_entry = log[2]  # identical object: term 3, "cmd3"
    result = log.after_append_entries(prev_log_index=2, entries=[same_entry])
    assert result == log
    assert result[2] is same_entry  # not replaced, since it already matched


# --- Six diverged followers, repaired against one leader ---------------------


def test_followers_without_extra_entries_end_up_identical_to_the_leader():
    # Two followers are only missing entries, so there is nothing of their
    # own to conflict with. Two diverge partway through, but the leader's
    # log covers and overwrites every diverging entry. In all four cases one
    # repair leaves the follower identical to the leader. The two followers
    # holding extra entries are excluded here — the tests below show why a
    # plain heartbeat does not remove those.
    leader_log = make_log(LEADER_TERMS)
    for name in (
        "missing_last_entry",
        "missing_last_six_entries",
        "conflicts_from_index_6",
        "conflicts_from_index_4",
    ):
        follower_log = make_log(FOLLOWER_TERMS[name])
        reconciled, _ = reconcile(leader_log, follower_log)
        assert [e.term for e in reconciled] == LEADER_TERMS, name


def test_follower_missing_the_last_entry_agrees_at_index_9():
    # The first probe (index 10) is past the end of its log and is rejected
    # once; the next one agrees at index 9.
    leader_log = make_log(LEADER_TERMS)
    follower_log = make_log(FOLLOWER_TERMS["missing_last_entry"])
    _, matched_at = reconcile(leader_log, follower_log)
    assert matched_at == 9


def test_follower_conflicting_from_index_6_agrees_at_index_5():
    leader_log = make_log(LEADER_TERMS)
    follower_log = make_log(FOLLOWER_TERMS["conflicts_from_index_6"])
    _, matched_at = reconcile(leader_log, follower_log)
    assert matched_at == 5  # last point of agreement is index 5


def test_follower_conflicting_from_index_4_agrees_at_index_3():
    leader_log = make_log(LEADER_TERMS)
    follower_log = make_log(FOLLOWER_TERMS["conflicts_from_index_4"])
    _, matched_at = reconcile(leader_log, follower_log)
    assert matched_at == 3  # last point of agreement is index 3


def test_one_extra_stale_entry_survives_a_plain_heartbeat():
    # A heartbeat that only covers indices both logs already hold must NOT
    # touch the follower's extra, never-committed entry 11 (term 6). Nothing
    # conflicts with it, and entries are only deleted from the first
    # conflict onward, so it stays exactly where it is. It is overwritten
    # only once the leader produces a genuinely conflicting entry there.
    follower_log = make_log(FOLLOWER_TERMS["one_extra_stale_entry"])

    assert follower_log.matches(prev_log_index=10, prev_log_term=6)
    heartbeat_result = follower_log.after_append_entries(prev_log_index=10, entries=[])
    assert [e.term for e in heartbeat_result] == FOLLOWER_TERMS["one_extra_stale_entry"]

    # Now the leader (still term 8) accepts a new client command as entry 11.
    new_entry = LogEntry(term=8, command="new-write")
    written_result = heartbeat_result.after_append_entries(
        prev_log_index=10, entries=[new_entry]
    )
    assert [e.term for e in written_result] == LEADER_TERMS + [8]


def test_two_extra_stale_entries_survive_a_plain_heartbeat():
    # Same as with one extra entry, but with two (11 and 12, term 7 each) —
    # showing that a genuine conflict deletes the conflicting entry and every
    # entry after it, not just the one entry that directly conflicts.
    follower_log = make_log(FOLLOWER_TERMS["two_extra_stale_entries"])

    assert follower_log.matches(prev_log_index=10, prev_log_term=6)
    heartbeat_result = follower_log.after_append_entries(prev_log_index=10, entries=[])
    assert [e.term for e in heartbeat_result] == FOLLOWER_TERMS["two_extra_stale_entries"]

    new_entry = LogEntry(term=8, command="new-write")
    written_result = heartbeat_result.after_append_entries(
        prev_log_index=10, entries=[new_entry]
    )
    # Both stale entries (11, 12) are gone, not just the conflicting one.
    assert [e.term for e in written_result] == LEADER_TERMS + [8]


def test_last_index_and_term_on_empty_and_nonempty_logs():
    assert Log().last_index == 0
    assert Log().last_term == 0
    log = make_log(LEADER_TERMS)
    assert log.last_index == 10
    assert log.last_term == 6


# --- DD-21: a command is an opaque string ------------------------------------


def test_log_entry_rejects_a_non_string_command():
    with pytest.raises(TypeError):
        LogEntry(term=1, command=5)
    with pytest.raises(TypeError):
        LogEntry(term=1, command=("put", "x", 1))


# --- first_differing_index: the part of a log that actually changed ---------


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
    # A stale entry kept through a heartbeat is not a change, so nothing
    # needs rewriting.
    follower_log = make_log(FOLLOWER_TERMS["one_extra_stale_entry"])
    after_heartbeat = follower_log.after_append_entries(prev_log_index=10, entries=[])
    assert follower_log.first_differing_index(after_heartbeat) is None


def test_first_differing_index_after_repairing_a_conflict_from_index_4_is_4():
    leader_log = make_log(LEADER_TERMS)
    follower_log = make_log(FOLLOWER_TERMS["conflicts_from_index_4"])
    reconciled, _ = reconcile(leader_log, follower_log)
    assert follower_log.first_differing_index(reconciled) == 4
