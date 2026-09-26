"""Tier 1 tests for the KV Store layer's state machine: the map, and the sessions guarding it.

APPLY-6 (applying is deterministic), FAIL-4, FAIL-6 (a put carries its request identifier and takes
effect once however often it is retried), DD-15 (sessions, and forgetting idle ones), DD-32 (time
is the cluster time on each command's entry).
"""

import pytest

from raftkv.kvstore import (
    KeyValueStore,
    OpenSession,
    Put,
    PutApplied,
    SessionExpired,
    SessionOpened,
    StaleRequest,
)

TIMEOUT = 100


class FedStore:
    """A KeyValueStore fed commands as a log would feed it: at consecutive indexes, with times.

    Attributes:
        store: The store being fed.
        fed: Every (index, cluster_time, command) fed so far, to replay elsewhere.
    """

    def __init__(self, session_timeout=TIMEOUT):
        self.store = KeyValueStore(session_timeout)
        self.fed = []

    def feed(self, command, time=0):
        """Apply `command` at the next index with cluster time `time`; return its result."""
        index = len(self.fed) + 1
        self.fed.append((index, time, command))
        return self.store.apply(index, time, command)

    def open_session(self, time=0):
        """Open a session; return its client ID."""
        return self.feed(OpenSession().encode(), time).client_id


def store_with(*pairs):
    """Return a store where one session put each (key, value) in order."""
    fed = FedStore()
    client = fed.open_session()
    for seq, (key, value) in enumerate(pairs, start=1):
        fed.feed(Put(client, seq, key, value).encode())
    return fed.store


# --- The map ---------------------------------------------------------------------------


def test_a_put_stores_the_value_and_returns_what_the_key_held_before():
    fed = FedStore()
    client = fed.open_session()
    assert fed.feed(Put(client, 1, "x", "5").encode()) == PutApplied(previous_value=None)
    assert fed.feed(Put(client, 2, "x", "6").encode()) == PutApplied(previous_value="5")
    assert fed.store.get("x") == "6"


def test_a_key_that_was_never_put_reads_as_none():
    assert store_with(("x", "5")).get("y") is None


def test_different_keys_hold_their_own_values():
    store = store_with(("x", "5"), ("y", "6"))
    assert (store.get("x"), store.get("y")) == ("5", "6")


def test_keys_come_back_sorted_whatever_order_they_were_put_in():
    assert store_with(("c", "3"), ("a", "1"), ("b", "2")).keys == ["a", "b", "c"]


def test_applying_the_same_commands_in_a_different_order_gives_a_different_map():
    # Why order matters: APPLY-4 requires log order.
    forwards = store_with(("x", "5"), ("x", "6"))
    backwards = store_with(("x", "6"), ("x", "5"))
    assert forwards.as_dict() != backwards.as_dict()


def test_as_dict_and_sessions_are_copies_a_caller_cannot_change_the_store_through():
    fed = FedStore()
    client = fed.open_session()
    fed.feed(Put(client, 1, "x", "5").encode())
    fed.store.as_dict()["x"] = "tampered"
    fed.store.sessions.clear()
    assert fed.store.get("x") == "5"
    assert fed.store.session(client) is not None


# --- Sessions and retries (FAIL-6) ------------------------------------------------------


def test_opening_a_session_gives_it_the_commands_index_as_its_client_id():
    fed = FedStore()
    fed.feed(OpenSession().encode())
    assert fed.feed(OpenSession().encode()) == SessionOpened(client_id=2)
    assert set(fed.store.sessions) == {1, 2}


def test_a_retried_put_does_not_undo_a_later_put_by_another_client():
    # A's put is applied, but A never hears so; B then puts x=2; A's retry arrives last. The
    # retry must not put x back to 1, and must tell A what its put did the first time.
    fed = FedStore()
    a, b = fed.open_session(), fed.open_session()
    first = fed.feed(Put(a, 1, "x", "1").encode())
    fed.feed(Put(b, 1, "x", "2").encode())

    retry = fed.feed(Put(a, 1, "x", "1").encode())

    assert fed.store.get("x") == "2"
    assert retry == first == PutApplied(previous_value=None)


def test_a_request_older_than_the_sessions_latest_is_refused():
    fed = FedStore()
    client = fed.open_session()
    fed.feed(Put(client, 1, "x", "1").encode())
    fed.feed(Put(client, 2, "x", "2").encode())

    assert fed.feed(Put(client, 1, "x", "1").encode()) == StaleRequest()
    assert fed.store.get("x") == "2"


def test_a_put_from_a_session_that_was_never_opened_is_refused():
    fed = FedStore()
    assert fed.feed(Put(7, 1, "x", "1").encode()) == SessionExpired()
    assert fed.store.as_dict() == {}


# --- Expiry (DD-15, DD-32) --------------------------------------------------------------


def test_a_session_idle_for_longer_than_the_timeout_is_refused_and_nothing_is_applied():
    # Another client changes x meanwhile, so a late retry that was applied would show.
    fed = FedStore()
    client = fed.open_session(time=0)
    fed.feed(Put(client, 1, "x", "1").encode(), time=10)
    other = fed.open_session(time=50)
    fed.feed(Put(other, 1, "x", "2").encode(), time=60)

    retry = fed.feed(Put(client, 1, "x", "1").encode(), time=10 + TIMEOUT + 1)

    assert retry == SessionExpired()
    assert fed.store.session(client) is None
    assert fed.store.get("x") == "2"


def test_a_session_idle_for_exactly_the_timeout_is_still_open():
    fed = FedStore()
    client = fed.open_session(time=0)
    assert fed.feed(Put(client, 1, "x", "1").encode(), time=TIMEOUT) == PutApplied(None)


def test_a_retry_counts_as_activity_and_keeps_the_session_open():
    fed = FedStore()
    client = fed.open_session(time=0)
    fed.feed(Put(client, 1, "x", "1").encode(), time=0)
    fed.feed(Put(client, 1, "x", "1").encode(), time=TIMEOUT)  # a retry

    assert fed.feed(Put(client, 2, "x", "2").encode(), time=2 * TIMEOUT) == PutApplied("1")


def test_a_put_counts_as_activity_and_keeps_the_session_open():
    fed = FedStore()
    client = fed.open_session(time=0)
    fed.feed(Put(client, 1, "x", "1").encode(), time=TIMEOUT)

    assert fed.feed(Put(client, 2, "x", "2").encode(), time=2 * TIMEOUT) == PutApplied("1")


def test_a_stale_request_does_not_count_as_activity():
    # A late copy of an answered request says nothing about whether its client is still there.
    fed = FedStore()
    client = fed.open_session(time=0)
    fed.feed(Put(client, 1, "x", "1").encode(), time=0)
    fed.feed(Put(client, 2, "x", "2").encode(), time=0)
    assert fed.feed(Put(client, 1, "x", "1").encode(), time=TIMEOUT) == StaleRequest()

    assert fed.feed(Put(client, 3, "x", "3").encode(), time=TIMEOUT + 1) == SessionExpired()


def test_another_clients_command_is_what_expires_an_idle_session():
    # Expiry is judged whenever any command is applied, at that command's cluster time.
    fed = FedStore()
    idle, busy = fed.open_session(time=0), fed.open_session(time=50)
    fed.feed(Put(busy, 1, "y", "1").encode(), time=TIMEOUT + 1)
    assert fed.store.session(idle) is None
    assert fed.store.session(busy) is not None


def test_a_command_that_is_not_understood_expires_nothing():
    # A refused command changes nothing at all, including which sessions are open.
    fed = FedStore()
    client = fed.open_session(time=0)
    with pytest.raises(ValueError):
        fed.store.apply(2, TIMEOUT + 1, "not a command")
    assert fed.store.session(client) is not None


# --- The same log, the same state (APPLY-6) ---------------------------------------------


def test_replaying_the_same_commands_reproduces_the_map_the_sessions_and_every_result():
    # A restarted node rebuilds its state by applying its log again from index 1 (DD-28); a
    # duplicate skipped the first time must be skipped again, and an expiry must recur.
    fed = FedStore()
    a, b = fed.open_session(time=0), fed.open_session(time=0)
    commands = [
        (Put(a, 1, "x", "1"), 5),
        (Put(b, 1, "x", "2"), 6),
        (Put(a, 1, "x", "1"), 7),  # a retry
        (Put(b, 2, "z", "0"), 60),
        (Put(b, 3, "y", "3"), 8 + TIMEOUT),  # a, idle since 7, expires here
        (Put(a, 2, "x", "4"), 9 + TIMEOUT),  # refused
    ]
    results = [fed.feed(command.encode(), time) for command, time in commands]

    replayed = KeyValueStore(TIMEOUT)
    replayed_results = [replayed.apply(*entry) for entry in fed.fed]

    assert replayed_results[2:] == results
    assert replayed.as_dict() == fed.store.as_dict() == {"x": "2", "y": "3", "z": "0"}
    assert replayed.sessions == fed.store.sessions
    assert results[-1] == SessionExpired()


@pytest.mark.parametrize("timeout", [0, -5])
def test_a_session_timeout_below_one_is_refused(timeout):
    with pytest.raises(ValueError):
        KeyValueStore(session_timeout=timeout)
