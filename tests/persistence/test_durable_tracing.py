"""Tier 2 tests for DurableNodeState's log lines and trace events, in etcd's formats and order.

A change is reported exactly when it is installed.
"""

import asyncio

import pytest

from raftkv.consensus import Cluster, Log, LogEntry, NodeState, Role
from raftkv.persistence import DurableNodeState, SqliteStore
from raftkv.tracing import LOG_LINES_LOGGER, TRACE_EVENTS_LOGGER, NodeSnapshot
from tests.append_entries_messages import append_entries
from tests.election_traces.checker import check_election_trace
from tests.persistence.store_doubles import FailingStore, GatedStore, seed_log, win_election
from tests.vote_messages import granted, refused, vote_request

NODE_ID = 7  # IDs from 7 up never look like the small terms and indexes these tests use.
THREE_NODES = Cluster([7, 8, 9])
SINGLE_NODE_ELECTION_LINES = [
    "7 started [peers: [], term: 0, vote: 0, lastindex: 0, lastterm: 0]",
    "7 is starting a new election at term 0",
    "7 became candidate at term 1",
    "7 became leader at term 1",
]
SINGLE_NODE_ELECTION_EVENTS = [
    ("InitState", "follower"),
    ("BecomeCandidate", "candidate"),
    ("BecomeLeader", "leader"),
]


def log_lines(caplog, node_id=NODE_ID):
    """Return the log lines `node_id` emitted, in order."""
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == LOG_LINES_LOGGER and r.node_id == node_id
    ]


def trace_events(caplog, node_id=NODE_ID):
    """Return the trace events `node_id` emitted, in order, as dicts."""
    return [
        r.trace_event.as_dict()
        for r in caplog.records
        if r.name == TRACE_EVENTS_LOGGER and r.node_id == node_id
    ]


async def test_winning_an_election_is_reported_as_etcd_reports_it(db_path, tracing_on):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        request = await durable.start_election()
        await durable.handle_vote_response(8, request.term, granted(term=request.term))

    assert log_lines(tracing_on) == [
        "7 started [peers: [8, 9], term: 0, vote: 0, lastindex: 0, lastterm: 0]",
        "7 is starting a new election at term 0",
        "7 became candidate at term 1",
        "7 [logterm: 0, index: 0] sent RequestVote request to 8 at term 1",
        "7 [logterm: 0, index: 0] sent RequestVote request to 9 at term 1",
        "7 received RequestVoteResponse from 8 at term 1",
        "7 has received 2 RequestVoteResponse votes and 0 vote rejections",
        "7 became leader at term 1",
    ]
    assert [e["name"] for e in trace_events(tracing_on)] == [
        "InitState",
        "BecomeCandidate",
        "SendRequestVoteRequest",
        "SendRequestVoteRequest",
        "ReceiveRequestVoteResponse",
        "BecomeLeader",
    ]
    assert trace_events(tracing_on)[-1]["prop"] == {
        "next": {8: 1, 9: 1},
        "match": {8: 0, 9: 0},
    }


async def test_a_single_node_election_is_reported_as_candidate_then_leader(db_path, tracing_on):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, Cluster([NODE_ID]))
        await durable.start_election()
    assert log_lines(tracing_on) == SINGLE_NODE_ELECTION_LINES
    assert [(e["name"], e["role"]) for e in trace_events(tracing_on)] == SINGLE_NODE_ELECTION_EVENTS


async def test_a_cancelled_single_node_election_is_still_reported_as_won(db_path, tracing_on):
    async with GatedStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, Cluster([NODE_ID]))
        election = asyncio.create_task(durable.start_election())
        await store.wait_for_write()
        election.cancel()
        store.release.set()
        with pytest.raises(asyncio.CancelledError):
            await election
    assert log_lines(tracing_on) == SINGLE_NODE_ELECTION_LINES
    assert [(e["name"], e["role"]) for e in trace_events(tracing_on)] == SINGLE_NODE_ELECTION_EVENTS


async def test_a_counted_refusal_is_reported_as_a_rejection(db_path, tracing_on):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        request = await durable.start_election()
        await durable.handle_vote_response(8, request.term, refused(term=request.term))
    assert log_lines(tracing_on)[-2:] == [
        "7 received RequestVoteResponse rejection from 8 at term 1",
        "7 has received 1 RequestVoteResponse votes and 1 vote rejections",
    ]


async def test_answers_that_are_not_counted_are_reported_as_ignored(db_path, tracing_on):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        first = await durable.start_election()
        second = await durable.start_election()
        # A grant from the term-1 election, then a grant arriving after the term-2 win.
        await durable.handle_vote_response(8, first.term, granted(term=first.term))
        await durable.handle_vote_response(9, second.term, granted(term=second.term))
        await durable.handle_vote_response(8, second.term, granted(term=second.term))
    assert [line for line in log_lines(tracing_on) if "ignored" in line] == [
        "7 [term: 2, role: candidate] ignored a RequestVoteResponse message from 8 "
        "[sent in term: 1]",
        "7 [term: 2, role: leader] ignored a RequestVoteResponse message from 8 [sent in term: 2]",
    ]


async def test_stepping_down_on_a_higher_term_answer_is_reported(db_path, tracing_on):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        request = await durable.start_election()
        await durable.handle_vote_response(8, request.term, refused(term=5))
    assert log_lines(tracing_on)[-2:] == [
        "7 [term: 1] received a RequestVoteResponse message with higher term from 8 [term: 5]",
        "7 became follower at term 5",
    ]
    assert [e["name"] for e in trace_events(tracing_on)][-2:] == [
        "ReceiveRequestVoteResponse",
        "BecomeFollower",
    ]
    assert trace_events(tracing_on)[-1]["state"] == {"term": 5, "vote": None}


async def test_a_vote_is_reported_with_the_facts_it_was_decided_on(db_path, tracing_on):
    # The node voted for 9 in term 3. A term-4 request from 8 first clears that vote, then is
    # refused because 8's log is behind: the line shows vote 0, the vote actually decided on.
    await seed_log(db_path, [1, 3])
    async with SqliteStore(db_path) as store:
        await store.save_term_and_vote(current_term=3, voted_for=9)
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        stale = vote_request(term=4, candidate=8, last_log_term=2, last_log_index=5)
        await durable.handle_vote_request(stale)

    assert log_lines(tracing_on)[1:] == [
        "7 [term: 3] received a RequestVote message with higher term from 8 [term: 4]",
        "7 became follower at term 4",
        "7 [logterm: 3, index: 2, vote: 0] rejected RequestVote from 8 "
        "[logterm: 2, index: 5] at term 4",
    ]


async def test_a_request_from_an_earlier_term_is_reported_as_such(db_path, tracing_on):
    async with SqliteStore(db_path) as store:
        await store.save_term_and_vote(current_term=5, voted_for=None)
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await durable.handle_vote_request(vote_request(term=3, candidate=8))
    assert log_lines(tracing_on)[-1] == (
        "7 [term: 5] rejected a RequestVote message with lower term from 8 [term: 3]"
    )


async def test_receive_event_has_the_state_before_and_answer_event_the_state_after(
    db_path, tracing_on
):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await durable.handle_vote_request(vote_request(term=1, candidate=8))
    received, _, answered = trace_events(tracing_on)[1:]
    assert received["name"] == "ReceiveRequestVoteRequest"
    assert received["state"] == {"term": 0, "vote": None}
    assert answered["name"] == "SendRequestVoteResponse"
    assert answered["state"] == {"term": 1, "vote": 8}
    assert answered["msg"] == {
        "type": "RequestVoteResponse",
        "term": 1,
        "from": 7,
        "to": 8,
        "reject": False,
    }


async def test_a_failed_write_reports_no_decision(db_path, tracing_on):
    async with FailingStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        with pytest.raises(OSError):
            await durable.start_election()
    assert [e["name"] for e in trace_events(tracing_on)] == ["InitState"]
    assert not any("became candidate" in line for line in log_lines(tracing_on))


async def test_a_cancelled_callers_installed_change_is_still_reported(db_path, tracing_on):
    async with GatedStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        election = asyncio.create_task(durable.start_election())
        await store.wait_for_write()
        election.cancel()
        store.release.set()
        with pytest.raises(asyncio.CancelledError):
            await election
    # The new term and self-vote were installed, so they are reported; no request was
    # returned, so none is reported as sent.
    assert [e["name"] for e in trace_events(tracing_on)] == ["InitState", "BecomeCandidate"]
    assert trace_events(tracing_on)[-1]["state"] == {"term": 1, "vote": 7}


async def test_a_vote_installed_without_an_answer_is_reported_as_persisted(db_path, tracing_on):
    # The caller is cancelled while its same-term vote is written: the vote is installed, but
    # no answer exists, so no SendRequestVoteResponse is reported.
    async with SqliteStore(db_path) as store:
        await store.save_term_and_vote(current_term=1, voted_for=None)
    async with GatedStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        answering = asyncio.create_task(
            durable.handle_vote_request(vote_request(term=1, candidate=8))
        )
        await store.wait_for_write()
        answering.cancel()
        store.release.set()
        with pytest.raises(asyncio.CancelledError):
            await answering
    assert log_lines(tracing_on)[1:] == ["7 [term: 1] voted for 8, but no answer was returned"]
    assert [e["name"] for e in trace_events(tracing_on)] == [
        "InitState",
        "ReceiveRequestVoteRequest",
        "PersistVote",
    ]
    assert trace_events(tracing_on)[-1]["state"] == {"term": 1, "vote": 8}


async def test_with_tracing_off_no_snapshot_is_taken(db_path, tracing_off, monkeypatch):
    def fail(node):
        raise AssertionError("a snapshot was taken while tracing was off")

    monkeypatch.setattr(NodeSnapshot, "of", fail)
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        request = await durable.start_election()
        grant = granted(term=request.term)
        assert await durable.handle_vote_response(8, request.term, grant) is True


async def test_every_method_accepts_its_arguments_by_keyword(db_path, tracing_on_or_off):
    async with SqliteStore(db_path) as store:
        durable = DurableNodeState(
            state=NodeState(NODE_ID), log=Log(), store=store, cluster=THREE_NODES
        )
        entry = LogEntry(term=1, command="x")
        answer = await durable.handle_append_entries(
            request=append_entries(term=1, prev_log_index=0, prev_log_term=0, entries=[entry])
        )
        assert answer.success is True
        assert await durable.handle_observed_term(term=1) is False  # already caught up by the RPC
        answer = await durable.handle_vote_request(
            request=vote_request(term=1, candidate=8, last_log_term=1, last_log_index=1)
        )
        assert answer.vote_granted is True
        request = await durable.start_election()
        won = await durable.handle_vote_response(
            voter=9, sent_in_term=request.term, response=granted(term=request.term)
        )
        assert won is True
        assert (durable.current_term, durable.voted_for, durable.role) == (2, 7, Role.LEADER)


async def test_arguments_passed_by_keyword_reach_the_report(db_path, tracing_on):
    async with SqliteStore(db_path) as store:
        durable = DurableNodeState(
            state=NodeState(NODE_ID), log=Log(), store=store, cluster=THREE_NODES
        )
        await durable.handle_observed_term(term=2)
        await durable.handle_vote_request(request=vote_request(term=2, candidate=8))
        request = await durable.start_election()
        await durable.handle_vote_response(
            voter=9, sent_in_term=request.term, response=granted(term=request.term)
        )

    assert log_lines(tracing_on) == [
        "7 started [peers: [8, 9], term: 0, vote: 0, lastindex: 0, lastterm: 0]",
        "7 [term: 0] observed a higher term 2",
        "7 became follower at term 2",
        "7 [logterm: 0, index: 0, vote: 0] cast RequestVote for 8 [logterm: 0, index: 0] at term 2",
        "7 is starting a new election at term 2",
        "7 became candidate at term 3",
        "7 [logterm: 0, index: 0] sent RequestVote request to 8 at term 3",
        "7 [logterm: 0, index: 0] sent RequestVote request to 9 at term 3",
        "7 received RequestVoteResponse from 9 at term 3",
        "7 has received 2 RequestVoteResponse votes and 0 vote rejections",
        "7 became leader at term 3",
    ]
    received = [e for e in trace_events(tracing_on) if e["name"] == "ReceiveRequestVoteResponse"]
    assert [(e["msg"]["from"], e["prop"]) for e in received] == [(9, {"sentInTerm": 3})]


async def test_a_real_three_node_election_trace_passes_the_checker(start_cluster, tracing_on):
    cluster = await start_cluster([1, 2, 3])
    await cluster.run_election(1)

    entries = [
        {"source": "node", "event": r.trace_event.as_dict()}
        for r in tracing_on.records
        if r.name == TRACE_EVENTS_LOGGER
    ]
    verdict = check_election_trace(entries)
    assert verdict.problems == []
    assert verdict.leaders_by_term == {1: [1]}


# --- AppendEntries, in etcd's MsgApp wording ------------------------------------------


async def test_a_higher_term_append_entries_is_reported_as_a_step_down(db_path, tracing_on):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await durable.start_election()
        await durable.handle_append_entries(append_entries(term=5, leader=8))

    assert log_lines(tracing_on)[-2:] == [
        "7 [term: 1] received a MsgApp message with higher term from 8 [term: 5]",
        "7 became follower at term 5",
    ]
    assert [e["name"] for e in trace_events(tracing_on)][-3:] == [
        "ReceiveAppendEntriesRequest",
        "BecomeFollower",
        "SendAppendEntriesResponse",
    ]


async def test_a_candidate_stepping_down_in_its_own_term_is_reported(db_path, tracing_on):
    # STATE-7: no term rises, so only the role change marks it.
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await durable.start_election()
        await durable.handle_append_entries(append_entries(term=1, leader=8))

    assert log_lines(tracing_on)[-1] == "7 became follower at term 1"
    assert [e["name"] for e in trace_events(tracing_on)][-2:] == [
        "BecomeFollower",
        "SendAppendEntriesResponse",
    ]


async def test_a_failed_consistency_check_is_reported_with_both_positions(db_path, tracing_on):
    await seed_log(db_path, [1, 1])
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await durable.handle_append_entries(
            append_entries(term=4, leader=8, prev_log_index=9, prev_log_term=4)
        )

    # etcd's wording: the node's own last position first, then the one the Leader named.
    assert log_lines(tracing_on)[-1] == (
        "7 [logterm: 1, index: 2] rejected MsgApp [logterm: 4, index: 9] from 8"
    )


async def test_an_outdated_leader_is_reported_as_rejected_for_its_term(db_path, tracing_on):
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await durable.handle_observed_term(9)
        await durable.handle_append_entries(append_entries(term=3, leader=8))

    assert log_lines(tracing_on)[-1] == (
        "7 [term: 9] rejected a MsgApp message with lower term from 8 [term: 3]"
    )
    [answer] = [e for e in trace_events(tracing_on) if e["name"] == "SendAppendEntriesResponse"]
    assert answer["msg"] == {
        "type": "AppendEntriesResponse",
        "term": 9,
        "from": 7,
        "to": 8,
        "reject": True,
    }


async def test_a_leader_refusing_its_own_term_is_reported_as_such_not_as_a_log_mismatch(
    db_path, tracing_on
):
    # The log check never runs here: a Leader refuses any AppendEntries at its own term.
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await win_election(durable)
        await durable.handle_append_entries(append_entries(term=durable.current_term, leader=8))

    assert log_lines(tracing_on)[-1] == (
        "7 [term: 1, role: leader] rejected a MsgApp message from 8 at the same term"
    )


async def test_a_leader_stepping_down_then_failing_the_log_check_is_reported_as_a_log_mismatch(
    db_path, tracing_on
):
    # It was a Leader when the RPC arrived, but it stepped down for the higher term and its log
    # then failed the check, so the rejection is the log's.
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        await win_election(durable)
        await durable.handle_append_entries(
            append_entries(term=5, leader=8, prev_log_index=4, prev_log_term=5)
        )

    assert log_lines(tracing_on)[-3:] == [
        "7 [term: 1] received a MsgApp message with higher term from 8 [term: 5]",
        "7 became follower at term 5",
        "7 [logterm: 0, index: 0] rejected MsgApp [logterm: 5, index: 4] from 8",
    ]


async def test_an_accepted_append_entries_gets_no_line_of_its_own(db_path, tracing_on):
    # etcd logs nothing on success; the event records it.
    async with SqliteStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        lines_before = len(log_lines(tracing_on))
        answer = await durable.handle_append_entries(
            append_entries(term=1, leader=8, entries=[LogEntry(1, "x")], leader_commit=1)
        )

    assert answer.success is True
    # Only the term catch-up's own two lines, and nothing at all about the entries stored.
    assert log_lines(tracing_on)[lines_before:] == [
        "7 [term: 0] received a MsgApp message with higher term from 8 [term: 1]",
        "7 became follower at term 1",
    ]
    [received] = [e for e in trace_events(tracing_on) if e["name"] == "ReceiveAppendEntriesRequest"]
    assert received["msg"] == {
        "type": "AppendEntries",
        "term": 1,
        "from": 8,
        "to": 7,
        "logTerm": 0,
        "index": 0,
        "entries": 1,
        "commit": 1,
    }


async def test_a_cancelled_append_entries_reports_no_answer_as_sent(db_path, tracing_on):
    async with GatedStore(db_path) as store:
        durable = await DurableNodeState.load(NODE_ID, store, THREE_NODES)
        caller = asyncio.create_task(
            durable.handle_append_entries(
                append_entries(term=4, leader=8, entries=[LogEntry(4, "x")])
            )
        )
        await store.wait_for_write()
        caller.cancel()
        store.release.set()
        with pytest.raises(asyncio.CancelledError):
            await caller

    names = [e["name"] for e in trace_events(tracing_on)]
    assert "ReceiveAppendEntriesRequest" in names
    # The change it persisted is reported; the answer it never returned is not.
    assert "BecomeFollower" in names
    assert "SendAppendEntriesResponse" not in names
