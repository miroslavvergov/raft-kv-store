"""Tier 1 unit tests for raftkv.tracing: trace events in etcd's shape, log
lines prefixed with the node's ID, and nothing built or emitted unless the
loggers are enabled.
"""

import logging

import pytest

from raftkv.consensus import LogPosition, RequestVoteRequest, RequestVoteResponse, Role
from raftkv.tracing import (
    LOG_LINES,
    TRACE_EVENTS,
    NodeSnapshot,
    NodeTracer,
    TraceEvent,
    TraceMessage,
)

LEADER = NodeSnapshot(node_id=7, role=Role.LEADER, term=3, vote=7,
                      last_log=LogPosition(term=2, index=5), peers=frozenset({8, 9}))


@pytest.fixture
def tracing_on(caplog):
    caplog.set_level(logging.DEBUG, logger=LOG_LINES)
    caplog.set_level(logging.DEBUG, logger=TRACE_EVENTS)
    return caplog


@pytest.fixture
def tracing_off(caplog):
    caplog.set_level(logging.WARNING, logger=LOG_LINES)
    caplog.set_level(logging.WARNING, logger=TRACE_EVENTS)
    return caplog


def test_a_request_message_uses_etcds_keys():
    request = RequestVoteRequest(term=4, candidate_id=1, last_log_index=7, last_log_term=3)
    assert TraceMessage.from_request(request, receiver=2).as_dict() == {
        "type": "RequestVote", "term": 4, "from": 1, "to": 2, "logTerm": 3, "index": 7,
    }


def test_a_response_message_carries_reject_and_no_log_fields():
    response = RequestVoteResponse(term=4, vote_granted=False)
    assert TraceMessage.from_response(response, sender=2, receiver=1).as_dict() == {
        "type": "RequestVoteResponse", "term": 4, "from": 2, "to": 1, "reject": True,
    }


def test_an_event_has_etcds_shape_and_leaves_out_what_does_not_apply():
    event = TraceEvent(name="BecomeCandidate", node_id=1, role="candidate", term=4, vote=1,
                       last_log_index=7, last_log_term=3)
    assert event.as_dict() == {
        "name": "BecomeCandidate",
        "nid": 1,
        "role": "candidate",
        "state": {"term": 4, "vote": 1},
        "log": {"index": 7, "term": 3},
    }


def test_a_line_starts_with_the_node_id(tracing_on):
    NodeTracer(7).line("became leader at term %d", 3)
    [record] = [r for r in tracing_on.records if r.name == LOG_LINES]
    assert record.getMessage() == "7 became leader at term 3"
    assert record.node_id == 7


def test_an_event_is_attached_to_its_record(tracing_on):
    NodeTracer(7).event("BecomeLeader", LEADER, properties={"next": {8: 6}})
    [record] = [r for r in tracing_on.records if r.name == TRACE_EVENTS]
    assert record.trace_event.as_dict() == {
        "name": "BecomeLeader", "nid": 7, "role": "leader", "state": {"term": 3, "vote": 7},
        "log": {"index": 5, "term": 2}, "prop": {"next": {8: 6}},
    }


def test_nothing_is_built_or_emitted_while_the_loggers_are_off(tracing_off, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("a trace event was built while tracing was off")

    monkeypatch.setattr("raftkv.tracing.node_tracer.TraceEvent", fail)
    tracer = NodeTracer(7)
    tracer.line("became leader at term %d", 3)
    tracer.event("BecomeLeader", LEADER)
    assert not tracer.enabled
    assert [r for r in tracing_off.records if r.name.startswith("raftkv")] == []


def test_the_library_logger_has_only_a_null_handler():
    handlers = logging.getLogger("raftkv").handlers
    assert [type(h) for h in handlers] == [logging.NullHandler]
