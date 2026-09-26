"""Tier 1 tests for SessionTable: open sessions, their latest request, and forgetting idle ones.

DD-15 (a session remembers its latest request and result; one idle for longer than the timeout is
forgotten), DD-32 (time is the cluster time passed in, never a clock).
"""

import pytest

from raftkv.kvstore import PutApplied, Session, SessionTable

TIMEOUT = 10


def test_a_new_session_has_no_requests_and_is_active_when_opened():
    table = SessionTable(TIMEOUT)
    table.open(12, now=5)
    assert table.get(12) == Session(last_seq=0, last_result=None, last_active=5)
    assert 12 in table and len(table) == 1


def test_a_session_that_was_never_opened_is_not_there():
    table = SessionTable(TIMEOUT)
    assert table.get(12) is None
    assert 12 not in table


def test_recording_a_request_keeps_its_number_result_and_time():
    table = SessionTable(TIMEOUT)
    table.open(12, now=5)
    table.record(12, seq=1, result=PutApplied("old"), now=7)
    assert table.get(12) == Session(last_seq=1, last_result=PutApplied("old"), last_active=7)


def test_touching_a_session_changes_only_when_it_was_last_active():
    table = SessionTable(TIMEOUT)
    table.open(12, now=5)
    table.record(12, seq=1, result=PutApplied(None), now=7)
    table.touch(12, now=9)
    assert table.get(12) == Session(last_seq=1, last_result=PutApplied(None), last_active=9)


def test_a_session_that_is_not_open_cannot_be_recorded_or_touched():
    table = SessionTable(TIMEOUT)
    with pytest.raises(KeyError):
        table.record(12, seq=1, result=PutApplied(None), now=7)
    with pytest.raises(KeyError):
        table.touch(12, now=7)


def test_only_sessions_idle_for_more_than_the_timeout_are_forgotten():
    table = SessionTable(TIMEOUT)
    table.open(1, now=0)
    table.open(2, now=5)
    assert table.expire_idle(now=10) == []  # 10 idle: exactly the timeout, still kept
    assert table.expire_idle(now=11) == [1]  # 11 idle: more than the timeout
    assert (1 in table, 2 in table) == (False, True)


def test_the_most_idle_sessions_are_forgotten_first():
    table = SessionTable(TIMEOUT)
    for client_id, now in ((1, 0), (2, 1), (3, 2), (4, 20)):
        table.open(client_id, now)
    assert table.expire_idle(now=25) == [1, 2, 3]
    assert list(table.as_dict()) == [4]


def test_activity_keeps_a_session_from_being_forgotten():
    table = SessionTable(TIMEOUT)
    table.open(1, now=0)
    table.open(2, now=1)
    table.touch(1, now=8)
    table.expire_idle(now=15)
    assert (1 in table, 2 in table) == (True, False)
    assert list(table.as_dict()) == [1]


def test_recording_a_request_moves_the_session_behind_idler_ones():
    table = SessionTable(TIMEOUT)
    table.open(1, now=0)
    table.open(2, now=1)
    table.record(1, seq=1, result=PutApplied(None), now=8)
    assert table.expire_idle(now=12) == [2]


def test_as_dict_lists_the_most_idle_first_and_is_a_copy():
    table = SessionTable(TIMEOUT)
    table.open(1, now=0)
    table.open(2, now=1)
    table.touch(1, now=2)
    sessions = table.as_dict()
    assert list(sessions) == [2, 1]
    sessions.clear()
    assert len(table) == 2


@pytest.mark.parametrize("timeout", [0, -1])
def test_a_timeout_below_one_is_refused(timeout):
    with pytest.raises(ValueError):
        SessionTable(timeout)
