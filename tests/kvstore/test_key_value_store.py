"""Tier 1 tests for the KV Store layer's state machine.

APPLY-6 (applying is deterministic), APPLY-7 (the exact command stored is the one applied),
DD-21 (a command is serialized once, here, and carried verbatim by Raft).
"""

import pytest

from raftkv.kvstore import KeyValueStore


def store_with(*pairs):
    """Return a store with each (key, value) applied in order."""
    store = KeyValueStore()
    for key, value in pairs:
        store.apply(KeyValueStore.put_command(key, value))
    return store


# --- Building a command ----------------------------------------------------------------


def test_the_same_put_always_serializes_to_the_same_string():
    # Two nodes must never disagree about what a log entry says, so the serialization
    # cannot depend on dictionary ordering or on when it ran.
    assert KeyValueStore.put_command("x", "5") == KeyValueStore.put_command("x", "5")


def test_a_command_has_one_exact_canonical_form():
    # The fields are sorted, so reordering them in the source cannot change what goes into a
    # log entry, and an entry written by an older build decodes the same (APPLY-7).
    assert KeyValueStore.put_command("x", "5") == '{"key": "x", "op": "put", "value": "5"}'


@pytest.mark.parametrize(("key", "value"), [(5, "v"), ("k", 5), (None, None)])
def test_a_command_can_only_be_built_from_strings(key, value):
    with pytest.raises(TypeError):
        KeyValueStore.put_command(key, value)


# --- Applying --------------------------------------------------------------------------


def test_applying_a_put_stores_the_value():
    assert store_with(("x", "5")).get("x") == "5"


def test_a_key_that_was_never_put_reads_as_none():
    assert store_with(("x", "5")).get("y") is None


def test_a_later_put_replaces_an_earlier_one():
    assert store_with(("x", "5"), ("x", "6")).get("x") == "6"


def test_different_keys_hold_their_own_values():
    store = store_with(("x", "5"), ("y", "6"))
    assert (store.get("x"), store.get("y")) == ("5", "6")


def test_keys_come_back_sorted_whatever_order_they_were_put_in():
    assert store_with(("c", "3"), ("a", "1"), ("b", "2")).keys == ["a", "b", "c"]


def test_applying_the_same_commands_in_the_same_order_gives_the_same_map():
    # APPLY-6: this is what lets every replica reach the same state from the same log.
    commands = [KeyValueStore.put_command(k, v) for k, v in (("x", "5"), ("y", "6"), ("x", "7"))]
    first, second = KeyValueStore(), KeyValueStore()
    for command in commands:
        first.apply(command)
    for command in commands:
        second.apply(command)
    assert first.as_dict() == second.as_dict() == {"x": "7", "y": "6"}


def test_applying_the_same_commands_in_a_different_order_gives_a_different_map():
    # Why order matters: APPLY-4 requires log order.
    forwards = store_with(("x", "5"), ("x", "6"))
    backwards = store_with(("x", "6"), ("x", "5"))
    assert forwards.as_dict() != backwards.as_dict()


@pytest.mark.parametrize(
    "command",
    [
        "",
        "not json",
        "[]",
        '{"op": "delete", "key": "x", "value": "5"}',  # a known shape, but not an op we run
        '{"op": "", "key": "x", "value": "5"}',
        '{"op": "delete", "key": "x"}',
        '{"op": "put", "key": "x"}',
        '{"op": "put", "key": 5, "value": "5"}',
    ],
)
def test_a_command_this_store_does_not_understand_is_refused(command):
    # A replica that quietly ignored a command every other replica applied would diverge,
    # so an unknown command is a bug to be raised, not a request to be declined.
    store = store_with(("x", "5"))
    with pytest.raises(ValueError):
        store.apply(command)
    assert store.as_dict() == {"x": "5"}


def test_as_dict_is_a_copy_a_caller_cannot_change_the_store_through():
    store = store_with(("x", "5"))
    store.as_dict()["x"] = "tampered"
    assert store.get("x") == "5"
