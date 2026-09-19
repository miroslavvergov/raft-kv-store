"""Tier 2 tests for what DurableNodeState reports about itself: etcd's
events in the order etcd emits them, log lines in etcd's format carrying
the facts each decision was made on, nothing reported for a write that
failed, a cancelled caller's installed change still reported — and the
trace of a real three-node election passing the independent checker.
"""

import asyncio
import logging

import pytest

from raftkv.consensus import Cluster, RequestVoteRequest, RequestVoteResponse
from raftkv.persistence import DurableNodeState, SqliteStore
from raftkv.tracing import LOG_LINES, TRACE_EVENTS
from tests.election_traces.checker import check_election_trace
from tests.persistence.store_doubles import FailingStore, GatedStore, seed_log

NODE_ID = 7
THREE_NODES = Cluster([7, 8, 9])


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "node.db")


@pytest.fixture
def trace(caplog):
    caplog.set_level(logging.DEBUG, logger=LOG_LINES)
    caplog.set_level(logging.DEBUG, logger=TRACE_EVENTS)
    return caplog


def lines(caplog, node_id=NODE_ID):
    return [r.getMessage() for r in caplog.records
            if r.name == LOG_LINES and r.node_id == node_id]


def events(caplog, node_id=NODE_ID):
    return [r.trace_event.as_dict() for r in caplog.records
            if r.name == TRACE_EVENTS and r.node_id == node_id]


def granted(term):
    return RequestVoteResponse(term=term, vote_granted=True)


async def test_winning_an_election_is_reported_as_etcd_reports_it(db_path, trace):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        request = await durable.start_election()
        await durable.handle_vote_response(8, request.term, granted(1))

    assert lines(trace) == [
        "7 started [peers: [8, 9], term: 0, vote: 0, lastindex: 0, lastterm: 0]",
        "7 is starting a new election at term 0",
        "7 became candidate at term 1",
        "7 [logterm: 0, index: 0] sent RequestVote request to 8 at term 1",
        "7 [logterm: 0, index: 0] sent RequestVote request to 9 at term 1",
        "7 received RequestVoteResponse from 8 at term 1",
        "7 has received 2 RequestVoteResponse votes and 0 vote rejections",
        "7 became leader at term 1",
    ]
    assert [e["name"] for e in events(trace)] == [
        "InitState",
        "BecomeCandidate",
        "SendRequestVoteRequest",
        "SendRequestVoteRequest",
        "ReceiveRequestVoteResponse",
        "BecomeLeader",
    ]
    assert events(trace)[-1]["prop"] == {"next": {8: 1, 9: 1}, "match": {8: 0, 9: 0}}


async def test_a_vote_is_reported_with_the_facts_it_was_decided_on(db_path, trace):
    # The node voted for 9 in term 3. A term-4 request from 8 first clears
    # that vote (a new term), and is then refused because 8's log is behind:
    # the line shows vote 0 — the vote the decision was actually made against.
    await seed_log(db_path, [1, 3])
    async with SqliteStore(db_path) as store:
        await store.save_term_and_vote(3, 9)
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        stale = RequestVoteRequest(term=4, candidate_id=8, last_log_index=5, last_log_term=2)
        await durable.handle_vote_request(stale)

    assert lines(trace)[1:] == [
        "7 [term: 3] received a RequestVote message with higher term from 8 [term: 4]",
        "7 became follower at term 4",
        "7 [logterm: 3, index: 2, vote: 0] rejected RequestVote from 8 "
        "[logterm: 2, index: 5] at term 4",
    ]


async def test_a_request_from_an_earlier_term_is_reported_as_such(db_path, trace):
    async with SqliteStore(db_path) as store:
        await store.save_term_and_vote(5, None)
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await durable.handle_vote_request(
            RequestVoteRequest(term=3, candidate_id=8, last_log_index=0, last_log_term=0)
        )
    assert lines(trace)[-1] == (
        "7 [term: 5] rejected a RequestVote message with lower term from 8 [term: 3]"
    )


async def test_receiving_shows_the_state_before_the_decision_and_answering_the_state_after(
    db_path, trace
):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await durable.handle_vote_request(
            RequestVoteRequest(term=1, candidate_id=8, last_log_index=0, last_log_term=0)
        )
    received, _, answered = events(trace)[1:]
    assert received["name"] == "ReceiveRequestVoteRequest"
    assert received["state"] == {"term": 0, "vote": None}
    assert answered["name"] == "SendRequestVoteResponse"
    assert answered["state"] == {"term": 1, "vote": 8}
    assert answered["msg"] == {"type": "RequestVoteResponse", "term": 1, "from": 7, "to": 8,
                               "reject": False}


async def test_a_failed_write_reports_no_decision(db_path, trace):
    async with FailingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        with pytest.raises(OSError):
            await durable.start_election()
    assert [e["name"] for e in events(trace)] == ["InitState"]
    assert not any("became candidate" in line for line in lines(trace))


async def test_a_cancelled_callers_installed_change_is_still_reported(db_path, trace):
    async with GatedStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        election = asyncio.create_task(durable.start_election())
        await store.wait_for_write()
        election.cancel()
        store.release.set()
        with pytest.raises(asyncio.CancelledError):
            await election
    # The new term and self-vote were installed, so they are reported; no
    # request was returned, so none is reported as sent.
    assert [e["name"] for e in events(trace)] == ["InitState", "BecomeCandidate"]
    assert events(trace)[-1]["state"] == {"term": 1, "vote": 7}


async def test_with_tracing_off_no_snapshot_is_taken(db_path, caplog, monkeypatch):
    caplog.set_level(logging.WARNING, logger=LOG_LINES)
    caplog.set_level(logging.WARNING, logger=TRACE_EVENTS)

    def fail(node):
        raise AssertionError("a snapshot was taken while tracing was off")

    monkeypatch.setattr("raftkv.tracing.traced_calls.NodeSnapshot.of", fail)
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        request = await durable.start_election()
        assert await durable.handle_vote_response(8, request.term, granted(1)) is True


async def test_a_real_three_node_election_trace_passes_the_checker(tmp_path, trace):
    cluster = Cluster([1, 2, 3])
    async with SqliteStore(str(tmp_path / "1.db")) as s1, \
            SqliteStore(str(tmp_path / "2.db")) as s2, \
            SqliteStore(str(tmp_path / "3.db")) as s3:
        stores = {1: s1, 2: s2, 3: s3}
        nodes = {n: await DurableNodeState.load(n, s, cluster) for n, s in stores.items()}
        request = await nodes[1].start_election()
        for voter in (2, 3):
            response = await nodes[voter].handle_vote_request(request)
            await nodes[1].handle_vote_response(voter, request.term, response)

    entries = [{"source": "node", "event": r.trace_event.as_dict()}
               for r in trace.records if r.name == TRACE_EVENTS]
    verdict = check_election_trace(entries)
    assert verdict.problems == []
    assert verdict.leaders_by_term == {1: [1]}
