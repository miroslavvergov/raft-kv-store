"""Tier 1 tests for how a Leadership confirms a read: a majority must answer a later request.

CLIENT-8 (before a read, the Leader confirms it still leads through a majority), DD-34 (only
requests built after the read began count, and each answer is matched to the request it
answers). Each answer changes only its own Follower.
"""

import pytest

from raftkv.consensus import Leadership
from tests.support.divergent_logs import make_log

TERM = 8
LEADER_ID = 1
LOG = make_log([1, 1, 4])


def leadership(followers):
    """Return a Leadership just won with `LOG`, before any request is built."""
    return Leadership(TERM, followers, LOG.last_index, cluster_time=0)


def build(lead, follower):
    """Build the next AppendEntries for `follower`."""
    return lead.append_entries_request_for(follower, LOG, LEADER_ID, 0)


# --- A majority must have answered ----------------------------------------------------------


def test_a_leader_alone_in_its_cluster_confirms_a_read_at_once():
    lead = leadership([])
    assert lead.confirmed_since(lead.confirmation_mark(), majority=1)


def test_three_nodes_confirm_a_read_once_one_follower_answers():
    lead = leadership([2, 3])
    mark = lead.confirmation_mark()
    request_to_two = build(lead, 2)
    build(lead, 3)  # node 3 has a request awaiting its answer too
    assert not lead.confirmed_since(mark, majority=2)

    lead.record_reply(2, request_to_two)

    assert lead.confirmed_since(mark, majority=2)  # the Leader and node 2


def test_five_nodes_need_two_followers_besides_the_leader():
    lead = leadership([2, 3, 4, 5])
    mark = lead.confirmation_mark()
    requests = {follower: build(lead, follower) for follower in (2, 3, 4, 5)}

    lead.record_reply(2, requests[2])
    assert not lead.confirmed_since(mark, majority=3)
    lead.record_reply(4, requests[4])
    assert lead.confirmed_since(mark, majority=3)


def test_an_answer_counts_for_its_own_follower_only():
    lead = leadership([2, 3])
    request_to_two = build(lead, 2)
    build(lead, 3)  # node 3 has a request awaiting its answer too

    lead.record_reply(2, request_to_two)

    assert (lead.answered_request(2), lead.answered_request(3)) == (1, 0)


# --- Only requests built after the read began count -----------------------------------------


def test_an_answer_to_a_request_built_before_the_read_began_does_not_confirm_it():
    # The Follower recognized this Leader before the read began, and the Leader may have been
    # superseded since; only a request built after the read can show it was not.
    lead = leadership([2, 3])
    earlier = build(lead, 2)
    mark = lead.confirmation_mark()

    lead.record_reply(2, earlier)
    assert not lead.confirmed_since(mark, majority=2)

    later = build(lead, 2)
    lead.record_reply(2, later)
    assert lead.confirmed_since(mark, majority=2)


def test_a_later_read_needs_requests_built_after_it_even_when_an_earlier_read_is_confirmed():
    lead = leadership([2, 3])
    first_mark = lead.confirmation_mark()
    lead.record_reply(2, build(lead, 2))
    assert lead.confirmed_since(first_mark, majority=2)

    second_mark = lead.confirmation_mark()
    assert not lead.confirmed_since(second_mark, majority=2)
    lead.record_reply(2, build(lead, 2))
    assert lead.confirmed_since(second_mark, majority=2)


def test_reads_that_begin_with_no_request_built_between_them_share_a_mark():
    lead = leadership([2, 3])
    first = lead.confirmation_mark()
    second = lead.confirmation_mark()
    build(lead, 2)
    third = lead.confirmation_mark()

    assert first == second < third


def test_asking_for_a_mark_changes_no_progress_and_uses_no_request_number():
    lead = leadership([2, 3])
    build(lead, 2)
    before = (lead.next_index(2), lead.match_index(2), lead.answered_request(2))

    lead.confirmation_mark()
    lead.confirmation_mark()

    assert (lead.next_index(2), lead.match_index(2), lead.answered_request(2)) == before
    lead.record_reply(2, build(lead, 2))
    assert lead.answered_request(2) == 2  # asking for a mark used up no request number


def test_requests_are_numbered_in_the_order_they_are_built_across_followers():
    lead = leadership([2, 3])
    build(lead, 2)
    first_to_three = build(lead, 3)
    second_to_two = build(lead, 2)

    lead.record_reply(3, first_to_three)
    lead.record_reply(2, second_to_two)

    assert (lead.answered_request(2), lead.answered_request(3)) == (3, 2)


# --- Each answer is matched to the request it answers ---------------------------------------


def test_only_the_request_built_last_for_a_follower_counts():
    lead = leadership([2, 3])
    older = build(lead, 2)
    newer = build(lead, 2)

    lead.record_reply(2, older)
    assert lead.answered_request(2) == 0
    lead.record_reply(2, newer)
    assert lead.answered_request(2) == 2


def test_a_second_answer_to_the_same_request_changes_nothing():
    lead = leadership([2, 3])
    request = build(lead, 2)
    lead.record_reply(2, request)
    later = build(lead, 2)

    lead.record_reply(2, request)  # a duplicate of the first answer arrives late

    assert lead.answered_request(2) == 1
    lead.record_reply(2, later)
    assert lead.answered_request(2) == 2


def test_two_requests_carrying_the_same_things_are_told_apart():
    # Heartbeats to a caught-up Follower are equal as values; each is still its own request.
    lead = leadership([2, 3])
    older = build(lead, 2)
    newer = build(lead, 2)
    assert older == newer and older is not newer

    lead.record_reply(2, older)

    assert lead.answered_request(2) == 0


def test_a_request_that_carries_entries_confirms_like_a_heartbeat():
    lead = leadership([2, 3])
    lead.record_rejection(2, TERM, prev_log_index=LOG.last_index)  # next_index 4 -> 3
    mark = lead.confirmation_mark()
    request = build(lead, 2)
    assert request.entries

    lead.record_reply(2, request)

    assert lead.confirmed_since(mark, majority=2)


# --- Answers that cannot count --------------------------------------------------------------


def test_an_answer_to_a_request_of_another_term_is_ignored_before_the_follower_is_looked_up():
    lead = leadership([2, 3])
    other = Leadership(TERM - 1, [2, 3, 99], LOG.last_index, cluster_time=0)
    request_to_unknown = build(other, 99)

    lead.record_reply(99, request_to_unknown)  # no KeyError: it is from another term

    assert lead.answered_request(2) == lead.answered_request(3) == 0


def test_an_answer_to_a_request_this_leadership_did_not_build_is_not_counted():
    # Not by its term, which matches, but because this leadership never built it.
    other = leadership([2, 3])
    request = build(other, 2)
    lead = leadership([2, 3])
    mark = lead.confirmation_mark()

    lead.record_reply(2, request)

    assert lead.answered_request(2) == 0
    assert not lead.confirmed_since(mark, majority=2)


def test_an_answer_from_an_unknown_follower_to_a_request_of_this_term_raises():
    lead = leadership([2, 3])
    request = build(lead, 2)
    with pytest.raises(KeyError):
        lead.record_reply(99, request)


def test_looking_up_an_unknown_followers_answer_raises():
    with pytest.raises(KeyError):
        leadership([2, 3]).answered_request(99)
