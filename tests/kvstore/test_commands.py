"""Tier 1 tests for the KV Store layer's commands: built once, serialized one way, read back.

DD-21 (a command is serialized once and carried verbatim), APPLY-7 (the exact command stored is the
one applied), FAIL-4 (every put carries its session and request number).
"""

import pytest

from raftkv.kvstore import OpenSession, Put, decode_command


def test_the_same_put_always_serializes_to_the_same_string():
    # Two nodes must never disagree about what a log entry says, so the serialization cannot
    # depend on dictionary ordering or on when it ran.
    assert Put(12, 1, "x", "5").encode() == Put(12, 1, "x", "5").encode()


def test_each_command_has_one_exact_canonical_form():
    # The fields are sorted, so reordering them in the source cannot change what goes into a
    # log entry, and an entry written by an older build decodes the same (APPLY-7).
    assert OpenSession().encode() == '{"op": "open_session"}'
    assert Put(12, 1, "x", "5").encode() == (
        '{"client_id": 12, "key": "x", "op": "put", "seq": 1, "value": "5"}'
    )


@pytest.mark.parametrize("command", [OpenSession(), Put(12, 3, "x", "5"), Put(1, 1, "", "")])
def test_decoding_a_command_gives_back_the_same_command(command):
    assert decode_command(command.encode()) == command


@pytest.mark.parametrize(
    ("fields", "error"),
    [
        ({"client_id": "12"}, TypeError),
        ({"client_id": True}, TypeError),
        ({"client_id": 0}, ValueError),
        ({"seq": 1.0}, TypeError),
        ({"seq": 0}, ValueError),
        ({"key": 5}, TypeError),
        ({"value": None}, TypeError),
    ],
)
def test_a_put_is_built_only_from_a_positive_session_and_number_and_strings(fields, error):
    with pytest.raises(error):
        Put(**{"client_id": 12, "seq": 1, "key": "x", "value": "5", **fields})


@pytest.mark.parametrize(
    "command",
    [
        "",
        "not json",
        "[]",
        '{"op": "delete", "key": "x"}',
        '{"op": "open_session", "client_id": 3}',
        '{"op": "put", "key": "x", "value": "5"}',
        '{"op": "put", "client_id": 12, "seq": 1, "key": "x", "value": "5", "extra": 1}',
        '{"op": "put", "client_id": 12, "seq": 0, "key": "x", "value": "5"}',
        '{"op": "put", "client_id": 12, "seq": true, "key": "x", "value": "5"}',
        '{"op": "put", "client_id": "12", "seq": 1, "key": "x", "value": "5"}',
        '{"op": "put", "client_id": 12, "seq": 1, "key": 5, "value": "5"}',
    ],
)
def test_a_string_that_is_not_a_command_this_layer_builds_is_refused(command):
    with pytest.raises(ValueError):
        decode_command(command)
