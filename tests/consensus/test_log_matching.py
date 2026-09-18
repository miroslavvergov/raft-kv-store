"""Tier 1 unit tests for the Log Matching consistency check and repair loop
(REPL-5, REPL-6, REPL-7, REPL-8), against hand-constructed logs — including
the six log-divergence scenarios from Figure 7 of the Raft paper.
"""

from raftkv.consensus import FollowerProgress, Log, LogEntry


def make_log(terms):
    return Log([LogEntry(term=t, command=f"cmd{i + 1}") for i, t in enumerate(terms)])


# Figure 7 of the Raft paper: a leader for term 8, and six possible follower
# logs (a)-(f) it might find on coming to power. Indices below are 1-based,
# matching the paper's own labeling.
LEADER_TERMS = [1, 1, 1, 4, 4, 5, 5, 6, 6, 6]

FOLLOWER_TERMS = {
    "a": [1, 1, 1, 4, 4, 5, 5, 6, 6],  # missing entry 10
    "b": [1, 1, 1, 4],  # missing entries 5-10
    "c": [1, 1, 1, 4, 4, 5, 5, 6, 6, 6, 6],  # extra uncommitted entry 11 (term 6)
    "d": [1, 1, 1, 4, 4, 5, 5, 6, 6, 6, 7, 7],  # extra entries 11-12 (term 7, 7)
    "e": [1, 1, 1, 4, 4, 4, 4],  # diverges at index 6 (term 4 vs leader's term 5)
    "f": [1, 1, 1, 2, 2, 2, 3, 3, 3, 3],  # diverges at index 4 (term 2 vs leader's term 4)
}


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


# --- REPL-6: nextIndex decrement on rejection --------------------------------


def test_follower_progress_record_rejection_decrements_next_index():
    progress = FollowerProgress(next_index=5)
    progress.record_rejection()
    assert progress.next_index == 4


def test_follower_progress_record_rejection_floors_at_one():
    progress = FollowerProgress(next_index=1)
    progress.record_rejection()
    assert progress.next_index == 1


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


# --- Figure 7: all six scenarios reconcile to exactly the leader's log ------


def test_figure_7_scenarios_without_extra_entries_reconcile_to_leaders_log():
    # (a), (b): missing entries only, nothing of the follower's own to
    # conflict with. (e), (f): diverge partway through, but the leader's own
    # log fully covers and overwrites the diverging tail. In all four cases
    # a single reconciliation against the leader's log ends up identical to
    # it. (c) and (d) are deliberately excluded here — see the dedicated
    # tests below for why a plain heartbeat does NOT reconcile them fully.
    leader_log = make_log(LEADER_TERMS)
    for label in ("a", "b", "e", "f"):
        follower_log = make_log(FOLLOWER_TERMS[label])
        reconciled, _ = reconcile(leader_log, follower_log)
        assert [e.term for e in reconciled] == LEADER_TERMS, f"scenario ({label})"


def test_figure_7_scenario_a_agrees_up_to_its_own_last_entry():
    # (a) is only missing the leader's last entry — the retry loop should
    # find agreement on the very first try, at index 9.
    leader_log = make_log(LEADER_TERMS)
    follower_log = make_log(FOLLOWER_TERMS["a"])
    _, matched_at = reconcile(leader_log, follower_log)
    assert matched_at == 9


def test_figure_7_scenario_e_diverges_starting_at_index_six():
    leader_log = make_log(LEADER_TERMS)
    follower_log = make_log(FOLLOWER_TERMS["e"])
    _, matched_at = reconcile(leader_log, follower_log)
    assert matched_at == 5  # last point of agreement is index 5


def test_figure_7_scenario_f_diverges_starting_at_index_four():
    leader_log = make_log(LEADER_TERMS)
    follower_log = make_log(FOLLOWER_TERMS["f"])
    _, matched_at = reconcile(leader_log, follower_log)
    assert matched_at == 3  # last point of agreement is index 3


def test_figure_7_scenario_c_extra_entry_survives_a_plain_heartbeat():
    # A heartbeat/probe that only covers indices already present on both
    # sides must NOT touch follower (c)'s extra, stale, uncommitted entry 11
    # (term 6) — nothing conflicts with it yet, so Figure 2's rules leave it
    # exactly where it is. It is only overwritten once the leader actually
    # produces a genuinely conflicting entry at that index.
    leader_log = make_log(LEADER_TERMS)
    follower_log = make_log(FOLLOWER_TERMS["c"])

    assert follower_log.matches(prev_log_index=10, prev_log_term=6)
    heartbeat_result = follower_log.after_append_entries(prev_log_index=10, entries=[])
    assert [e.term for e in heartbeat_result] == FOLLOWER_TERMS["c"]

    # Now the leader (still term 8) accepts a new client command as entry 11.
    new_entry = LogEntry(term=8, command="new-write")
    written_result = heartbeat_result.after_append_entries(
        prev_log_index=10, entries=[new_entry]
    )
    assert [e.term for e in written_result] == LEADER_TERMS + [8]


def test_figure_7_scenario_d_extra_entries_survive_a_plain_heartbeat():
    # Same story as (c), but with two extra stale entries (11, 12; term 7
    # each) instead of one — demonstrating that a genuine conflict discards
    # "the existing entry and all that follow it" (Figure 2, rule 3), not
    # just the one entry that directly conflicts.
    leader_log = make_log(LEADER_TERMS)
    follower_log = make_log(FOLLOWER_TERMS["d"])

    assert follower_log.matches(prev_log_index=10, prev_log_term=6)
    heartbeat_result = follower_log.after_append_entries(prev_log_index=10, entries=[])
    assert [e.term for e in heartbeat_result] == FOLLOWER_TERMS["d"]

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


def test_first_differing_index_after_a_heartbeat_on_figure_7_c_is_none():
    # The stale tail follower (c) keeps through a heartbeat is not a change,
    # so nothing needs rewriting.
    follower_log = make_log(FOLLOWER_TERMS["c"])
    after_heartbeat = follower_log.after_append_entries(prev_log_index=10, entries=[])
    assert follower_log.first_differing_index(after_heartbeat) is None


def test_first_differing_index_after_repairing_figure_7_f_is_index_four():
    leader_log = make_log(LEADER_TERMS)
    follower_log = make_log(FOLLOWER_TERMS["f"])
    reconciled, _ = reconcile(leader_log, follower_log)
    assert follower_log.first_differing_index(reconciled) == 4
