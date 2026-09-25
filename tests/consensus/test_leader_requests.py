"""Tier 1 tests for the AppendEntries a Leader builds: Leadership.append_entries_request_for.

REPL-2 (replicate to every Follower), REPL-3 (the previous entry travels with the RPC), REPL-4 (so
does the Leader's commit index), REPL-14 (start one past the Leader's last entry).
"""

import pytest

from raftkv.consensus import Leadership
from tests.support.divergent_logs import FOLLOWER_TERMS, LEADER_TERMS, make_log

TERM = 8
LEADER_ID = 1
FOLLOWERS = [2, 3]


def fresh_leadership(log):
    """Return a Leadership just won with `log`, before any Follower has answered."""
    return Leadership(TERM, FOLLOWERS, log.last_index)


def back_off(lead, follower):
    """Record `follower`'s rejection of the probe now outstanding to it."""
    lead.record_rejection(follower, TERM, prev_log_index=lead.next_index(follower) - 1)


# --- What the request carries ----------------------------------------------------------


def test_the_first_request_after_winning_is_a_heartbeat_after_the_last_entry():
    # REPL-14 assumes the Follower has everything, so nothing is sent yet: the previous entry is
    # the Leader's last one, and the Follower's answer shows whether it really has it.
    log = make_log([1, 1, 4])
    request = fresh_leadership(log).append_entries_request_for(2, log, LEADER_ID, 0)
    assert (request.prev_log_index, request.prev_log_term, request.entries) == (3, 4, ())


def test_the_request_carries_the_leaders_term_id_and_commit_index():
    log = make_log([1, 1, 4])
    request = fresh_leadership(log).append_entries_request_for(2, log, LEADER_ID, 2)
    assert (request.term, request.leader_id, request.leader_commit) == (TERM, LEADER_ID, 2)


def test_after_a_rejection_it_carries_everything_from_the_lowered_next_index():
    log = make_log([1, 1, 4, 4])
    lead = fresh_leadership(log)
    back_off(lead, 2)  # next_index 5 -> 4
    back_off(lead, 2)  # next_index 4 -> 3

    request = lead.append_entries_request_for(2, log, LEADER_ID, 0)

    assert (request.prev_log_index, request.prev_log_term) == (2, 1)
    assert request.entries == tuple(log.entries_from(3))


def test_backed_all_the_way_to_the_start_the_request_carries_the_whole_log():
    log = make_log([1, 1])
    lead = fresh_leadership(log)
    back_off(lead, 2)
    back_off(lead, 2)

    request = lead.append_entries_request_for(2, log, LEADER_ID, 0)

    assert (request.prev_log_index, request.prev_log_term) == (0, 0)
    assert request.entries == tuple(log)


def test_after_a_success_it_carries_only_what_the_follower_lacks():
    log = make_log([1, 1, 4])
    lead = fresh_leadership(log)
    lead.record_success(2, TERM, prev_log_index=0, entry_count=3)  # confirmed through 3
    grown = make_log([1, 1, 4, 8, 8])  # the Leader then appended two entries

    request = lead.append_entries_request_for(2, grown, LEADER_ID, 0)

    assert (request.prev_log_index, request.prev_log_term) == (3, 4)
    assert request.entries == tuple(grown.entries_from(4))


def test_a_caught_up_follower_gets_a_heartbeat():
    log = make_log([1, 1, 4])
    lead = fresh_leadership(log)
    lead.record_success(2, TERM, prev_log_index=0, entry_count=3)
    assert lead.append_entries_request_for(2, log, LEADER_ID, 3).entries == ()


def test_each_follower_gets_a_request_from_its_own_next_index():
    log = make_log([1, 1, 4])
    lead = fresh_leadership(log)
    back_off(lead, 3)  # only Follower 3 backs off
    two = lead.append_entries_request_for(2, log, LEADER_ID, 0)
    three = lead.append_entries_request_for(3, log, LEADER_ID, 0)
    assert (two.prev_log_index, three.prev_log_index) == (3, 2)


def test_building_a_request_changes_no_progress():
    log = make_log([1, 1, 4])
    lead = fresh_leadership(log)
    lead.append_entries_request_for(2, log, LEADER_ID, 0)
    assert (lead.next_index(2), lead.match_index(2)) == (4, 0)


def test_a_request_for_a_node_that_is_not_a_follower_raises_key_error():
    log = make_log([1])
    with pytest.raises(KeyError):
        fresh_leadership(log).append_entries_request_for(9, log, LEADER_ID, 0)


# --- The request, the Follower's rule, and the Leader's progress, together -------------


def repair_with_requests(leader_log, follower_log):
    """Build, answer, and record requests until the Follower accepts one.

    Returns:
        The Follower's log after accepting, and the Leadership that repaired it.
    """
    lead = fresh_leadership(leader_log)
    for _ in range(leader_log.last_index + 1):
        request = lead.append_entries_request_for(2, leader_log, LEADER_ID, 0)
        if follower_log.matches(request.prev_log_index, request.prev_log_term):
            follower_log = follower_log.after_append_entries(
                request.prev_log_index, list(request.entries)
            )
            lead.record_success(2, TERM, request.prev_log_index, len(request.entries))
            return follower_log, lead
        back_off(lead, 2)
    raise AssertionError("never reached an index where the logs agree")


@pytest.mark.parametrize(
    "follower",
    [
        "missing_last_entry",
        "missing_last_six_entries",
        "conflicts_from_index_6",
        "conflicts_from_index_4",
    ],
)
def test_requests_repair_any_shorter_or_conflicting_follower_exactly(follower):
    leader_log = make_log(LEADER_TERMS)
    repaired, lead = repair_with_requests(leader_log, make_log(FOLLOWER_TERMS[follower]))
    assert repaired == leader_log
    assert lead.match_index(2) == leader_log.last_index


def test_a_request_built_after_the_repair_is_a_heartbeat():
    leader_log = make_log(LEADER_TERMS)
    _, lead = repair_with_requests(leader_log, make_log(FOLLOWER_TERMS["conflicts_from_index_4"]))
    assert lead.append_entries_request_for(2, leader_log, LEADER_ID, 0).entries == ()
