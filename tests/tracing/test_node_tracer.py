"""Tier 1 tests for NodeTracer and trace messages: etcd's shapes, each line led by the node's ID.

Nothing is built or emitted unless the loggers are enabled.
"""

import logging

from raftkv.consensus import LogPosition, Role
from raftkv.tracing import (
    LOG_LINES_LOGGER,
    TRACE_EVENTS_LOGGER,
    NodeSnapshot,
    NodeTracer,
    TraceEvent,
    TraceMessage,
)
from tests.vote_messages import refused, vote_request

LEADER = NodeSnapshot(
    node_id=7,
    role=Role.LEADER,
    current_term=3,
    voted_for=7,
    last_log_position=LogPosition(term=2, index=5),
    peers=frozenset({8, 9}),
)


def test_a_request_message_uses_etcds_keys():
    request = vote_request(term=4, candidate=1, last_log_term=3, last_log_index=7)
    assert TraceMessage.from_vote_request(request, receiver=2).as_dict() == {
        "type": "RequestVote",
        "term": 4,
        "from": 1,
        "to": 2,
        "logTerm": 3,
        "index": 7,
    }


def test_a_response_message_carries_reject_and_no_log_fields():
    response = refused(term=4)
    assert TraceMessage.from_vote_response(response, sender=2, receiver=1).as_dict() == {
        "type": "RequestVoteResponse",
        "term": 4,
        "from": 2,
        "to": 1,
        "reject": True,
    }


def test_an_event_has_etcds_shape_and_leaves_out_what_does_not_apply():
    event = TraceEvent(
        name="BecomeCandidate",
        node_id=1,
        role="candidate",
        term=4,
        vote=1,
        last_log_index=7,
        last_log_term=3,
    )
    assert event.as_dict() == {
        "name": "BecomeCandidate",
        "nid": 1,
        "role": "candidate",
        "state": {"term": 4, "vote": 1},
        "log": {"index": 7, "term": 3},
    }


def test_a_line_starts_with_the_node_id(tracing_on):
    NodeTracer(7).emit_line("became leader at term %d", 3)
    [record] = [r for r in tracing_on.records if r.name == LOG_LINES_LOGGER]
    assert record.getMessage() == "7 became leader at term 3"
    assert record.node_id == 7


def test_an_event_is_attached_to_its_record(tracing_on):
    # Under --trace-elections this lone event is the test's whole trace, and must break no rule.
    restarted = LEADER.with_role(Role.FOLLOWER)
    NodeTracer(7).emit_event("InitState", restarted, properties={"peers": [8, 9]})
    [record] = [r for r in tracing_on.records if r.name == TRACE_EVENTS_LOGGER]
    assert record.trace_event.as_dict() == {
        "name": "InitState",
        "nid": 7,
        "role": "follower",
        "state": {"term": 3, "vote": 7},
        "log": {"index": 5, "term": 2},
        "prop": {"peers": [8, 9]},
    }


def test_nothing_is_built_or_emitted_while_the_loggers_are_off(tracing_off, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("a trace event was built while tracing was off")

    monkeypatch.setattr(TraceEvent, "__init__", fail)
    tracer = NodeTracer(7)
    tracer.emit_line("became leader at term %d", 3)
    tracer.emit_event("BecomeLeader", LEADER)
    assert not tracer.enabled
    assert [r for r in tracing_off.records if r.name.startswith("raftkv")] == []


def test_the_library_logger_has_only_a_null_handler():
    handlers = logging.getLogger("raftkv").handlers
    assert [type(h) for h in handlers] == [logging.NullHandler]
