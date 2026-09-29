"""Tier 2 tests for RaftNode: one real node, its peers played by a scripted transport.

The test ticks the clock itself, and every election timeout is drawn as the longest there is, so
every deadline is exact. ELECT-1, ELECT-2, ELECT-6, ELECT-13, STATE-3, REPL-2, REPL-7, REPL-9,
APPLY-4, APPLY-5, CLIENT-4, CLIENT-5, CLIENT-6, CLIENT-7, FAIL-2, FAIL-3, DD-9, DD-12, DD-26,
DD-28, DD-29, DD-30, DD-32, DD-33.
"""

import asyncio
import inspect
import random

import pytest

from raftkv.consensus import (
    Cluster,
    CommittedEntryConflictError,
    Log,
    LogEntry,
    LogPosition,
    NotLeaderError,
    Role,
)
from raftkv.kvstore import (
    KeyValueStore,
    OpenSession,
    Put,
    PutApplied,
    SessionExpired,
    SessionOpened,
    StaleRequest,
)
from raftkv.node import DurableNodeState
from raftkv.runtime import (
    LeadershipLostError,
    NodeStoppedError,
    PeerUnreachableError,
    RaftNode,
    Timing,
)
from raftkv.storage import SqliteStore
from tests.support.append_entries_messages import accepted, append_entries, heartbeat, rejected
from tests.support.store_doubles import GatedStore, let_other_tasks_run
from tests.support.vote_messages import granted, refused, vote_request
from tests.support.waiting import eventually, within_bound

NODE_ID = 1
THREE_NODES = Cluster([NODE_ID, 2, 3])
TIMING = Timing(heartbeat_ticks=1, election_ticks=10)
LONGEST = 2 * TIMING.election_ticks - 1  # every timeout these tests draw


class LongestTimeouts(random.Random):
    """Draws every election timeout as the longest in its range, so each deadline is known."""

    def randrange(self, start, stop, step=1):
        return stop - 1


class ScriptedTransport:
    """Records every RPC sent; `answer(kind, peer, request)` replies, or None loses the RPC.

    `answer` may return an awaitable, which holds the RPC in flight until it is done.

    Attributes:
        sent: Every RPC sent, as ("vote" or "append", peer, request).
    """

    def __init__(self, answer=None):
        self.sent = []
        self.answer = answer

    async def request_vote(self, peer, request):
        return await self._reply("vote", peer, request)

    async def append_entries(self, peer, request):
        return await self._reply("append", peer, request)

    def of_kind(self, kind):
        """Return the (peer, request) of every RPC of `kind` sent so far."""
        return [(peer, request) for sent_kind, peer, request in self.sent if sent_kind == kind]

    async def _reply(self, kind, peer, request):
        self.sent.append((kind, peer, request))
        result = None if self.answer is None else self.answer(kind, peer, request)
        if inspect.isawaitable(result):
            result = await result
        if result is None:
            raise PeerUnreachableError(f"no answer from {peer}")
        return result


def everyone_agrees(kind, peer, request):
    """Grant every vote and accept every AppendEntries."""
    return granted(term=request.term) if kind == "vote" else accepted(term=request.term)


class FollowerLogs:
    """Peers that hold real logs and answer AppendEntries as a Follower would; grant votes."""

    def __init__(self, logs):
        self.logs = dict(logs)

    def __call__(self, kind, peer, request):
        if kind == "vote":
            return granted(term=request.term)
        log = self.logs[peer]
        if not log.matches(request.prev_log_index, request.prev_log_term):
            return rejected(term=request.term)
        self.logs[peer] = log.after_append_entries(request.prev_log_index, list(request.entries))
        return accepted(term=request.term)


async def held_forever():
    """An answer that never comes, holding its RPC in flight until the node stops."""
    await asyncio.Event().wait()


async def accepted_once_released(release, request):
    """Accept `request`, once `release` is set."""
    await release.wait()
    return accepted(term=request.term)


def result_of(index, cluster_time, command):
    """A state machine double whose result for a command names it."""
    return f"result of {command}"


class HoldableFollowers:
    """Peers that grant every vote and accept every AppendEntries, held in flight on request."""

    def __init__(self):
        self._held = asyncio.Event()
        self._released = asyncio.Event()

    def hold(self):
        """Hold every AppendEntries in flight from now on, until `release`."""
        self._held.set()

    def release(self):
        """Answer every AppendEntries held, and every one after."""
        self._released.set()

    def __call__(self, kind, peer, request):
        if kind == "append" and self._held.is_set() and not self._released.is_set():
            return accepted_once_released(self._released, request)
        return everyone_agrees(kind, peer, request)


async def node_with(
    store, transport=None, *, timing=TIMING, cluster=THREE_NODES, apply=None, rng=None
):
    """Load node 1 from `store` and return its RaftNode, drawing the longest timeouts."""
    durable = await DurableNodeState.load(NODE_ID, store, cluster, apply=apply)
    return RaftNode(durable, transport or ScriptedTransport(), timing, rng or LongestTimeouts())


async def tick(node, count=1):
    """Tick `node` `count` times, letting every RPC and apply it starts finish after each."""
    for _ in range(count):
        await node.tick()
        await within_bound(node.idle())


async def win(node):
    """Run out node 1's election timeout; with `everyone_agrees`, it becomes Leader."""
    await tick(node, node.election_timeout)
    assert node.durable.role is Role.LEADER


async def start_proposal(node, command="x"):
    """Start a proposal of `command` and return its task once it is appended and waiting."""
    waiting = node.pending_proposals
    proposal = asyncio.create_task(node.propose(command))
    await eventually(lambda: node.pending_proposals > waiting)
    return proposal


# --- The election timeout (ELECT-1, ELECT-2, ELECT-13, DD-9) ----------------------------


async def test_a_follower_starts_an_election_exactly_when_its_timeout_is_reached(db_path):
    transport = ScriptedTransport()
    async with SqliteStore(db_path) as store:
        node = await node_with(store, transport)
        await tick(node, LONGEST - 1)
        assert node.durable.role is Role.FOLLOWER
        assert transport.sent == []

        await tick(node)

        assert (node.durable.role, node.durable.current_term) == (Role.CANDIDATE, 1)
        assert sorted(peer for peer, _ in transport.of_kind("vote")) == [2, 3]  # ELECT-6


async def test_every_restart_draws_a_new_timeout_from_the_configured_range(db_path):
    async with SqliteStore(db_path) as store:
        node = await node_with(store, rng=random.Random(0))
        drawn = set()
        for _ in range(30):
            await node.handle_append_entries(heartbeat(term=1, leader=2))
            drawn.add(node.election_timeout)
        assert drawn <= set(range(10, 20))
        assert len(drawn) > 1


async def test_an_append_entries_from_the_leader_restarts_the_timeout(db_path):
    # The first heartbeat raises the term, which restarts the timeout by itself; the second,
    # in the same term, is the one under test.
    async with SqliteStore(db_path) as store:
        node = await node_with(store)
        await node.handle_append_entries(heartbeat(term=1, leader=2))
        await tick(node, LONGEST - 1)

        await node.handle_append_entries(heartbeat(term=1, leader=2))
        await tick(node, LONGEST - 1)
        assert node.durable.role is Role.FOLLOWER
        await tick(node)
        assert node.durable.role is Role.CANDIDATE


async def test_a_rejected_append_entries_from_the_leader_restarts_it_too(db_path):
    # The node lacks entry 5, so it rejects; the Leader is still alive and in charge.
    async with SqliteStore(db_path) as store:
        node = await node_with(store)
        await node.handle_append_entries(heartbeat(term=1, leader=2))
        await tick(node, LONGEST - 1)

        answer = await node.handle_append_entries(
            heartbeat(term=1, leader=2, prev_log_index=5, prev_log_term=1)
        )
        await tick(node, LONGEST - 1)

        assert answer.success is False
        assert node.durable.role is Role.FOLLOWER


async def test_an_outdated_leaders_append_entries_does_not_restart_it(db_path):
    async with SqliteStore(db_path) as store:
        node = await node_with(store)
        await node.handle_append_entries(heartbeat(term=5, leader=2))
        await tick(node, 10)

        await node.handle_append_entries(heartbeat(term=4, leader=3))
        await tick(node, LONGEST - 10 - 1)
        assert node.durable.role is Role.FOLLOWER
        await tick(node)
        assert node.durable.role is Role.CANDIDATE


async def test_granting_a_vote_restarts_the_timeout(db_path):
    # The node is at term 1 already, so the grant changes no term, which would restart the
    # timeout by itself.
    async with SqliteStore(db_path) as store:
        node = await node_with(store)
        await node.handle_append_entries(heartbeat(term=1, leader=2))
        await tick(node, LONGEST - 1)

        answer = await node.handle_request_vote(vote_request(term=1, candidate=3))
        await tick(node, LONGEST - 1)

        assert answer.vote_granted is True
        assert node.durable.role is Role.FOLLOWER
        await tick(node)
        assert node.durable.role is Role.CANDIDATE


async def test_refusing_an_outdated_vote_request_does_not_restart_it(db_path):
    async with SqliteStore(db_path) as store:
        node = await node_with(store)
        await node.handle_append_entries(heartbeat(term=3, leader=2))
        await tick(node, 10)

        answer = await node.handle_request_vote(vote_request(term=2, candidate=3))
        await tick(node, LONGEST - 10 - 1)

        assert answer.vote_granted is False
        assert node.durable.role is Role.FOLLOWER
        await tick(node)
        assert node.durable.role is Role.CANDIDATE


async def test_a_new_term_restarts_the_timeout_even_when_the_vote_is_refused(db_path):
    # DD-9: the node holds an entry the Candidate lacks, so it refuses, but it takes the new
    # term, and any change of term restarts the timeout.
    async with SqliteStore(db_path) as store:
        node = await node_with(store)
        await node.handle_append_entries(
            append_entries(term=1, leader=2, entries=[LogEntry(1, "a")])
        )
        await tick(node, 10)

        answer = await node.handle_request_vote(vote_request(term=5, candidate=3))
        await tick(node, LONGEST - 1)

        assert answer.vote_granted is False
        assert (node.durable.role, node.durable.current_term) == (Role.FOLLOWER, 5)


async def test_a_candidate_that_times_out_starts_a_new_election_in_a_higher_term(db_path):
    transport = ScriptedTransport()
    async with SqliteStore(db_path) as store:
        node = await node_with(store, transport)
        await tick(node, LONGEST)
        assert node.durable.current_term == 1

        await tick(node, LONGEST)

        assert (node.durable.role, node.durable.current_term) == (Role.CANDIDATE, 2)
        assert {request.term for _, request in transport.of_kind("vote")} == {1, 2}


async def test_an_election_due_while_the_leader_is_being_heard_is_not_started(db_path):
    # The timeout is reached while the Leader's AppendEntries holds the lock, writing. Once it
    # is written the timeout has restarted, so no election follows (ELECT-2).
    transport = ScriptedTransport()
    async with GatedStore(db_path) as store:
        store.release.set()
        node = await node_with(store, transport)
        await node.handle_append_entries(heartbeat(term=1, leader=2))
        await tick(node, LONGEST - 1)

        store.hold_next_write()
        heard = asyncio.create_task(
            node.handle_append_entries(append_entries(term=1, leader=2, entries=[LogEntry(1, "a")]))
        )
        await store.wait_for_write()
        timed_out = asyncio.create_task(node.tick())
        await let_other_tasks_run()
        store.release.set()
        await within_bound(asyncio.gather(heard, timed_out))
        await within_bound(node.idle())

        assert (node.durable.role, node.durable.current_term) == (Role.FOLLOWER, 1)
        assert transport.of_kind("vote") == []


# --- Resending a RequestVote (FAIL-2, DD-30) --------------------------------------------


async def test_a_candidate_resends_its_request_only_to_peers_that_have_not_answered(db_path):
    def two_refuses_three_is_lost(kind, peer, request):
        return None if peer == 3 else refused(term=request.term)

    transport = ScriptedTransport(two_refuses_three_is_lost)
    async with SqliteStore(db_path) as store:
        node = await node_with(store, transport, timing=Timing(heartbeat_ticks=2))
        await tick(node, LONGEST)
        first = dict(transport.of_kind("vote"))
        transport.sent.clear()

        await tick(node, 2)

        assert transport.of_kind("vote") == [(3, first[3])]  # the same request, to 3 only


async def test_a_candidate_that_hears_from_the_leader_stops_resending(db_path):
    transport = ScriptedTransport()
    async with SqliteStore(db_path) as store:
        node = await node_with(store, transport)
        await tick(node, LONGEST)
        transport.sent.clear()

        await node.handle_append_entries(heartbeat(term=1, leader=2))  # STATE-7
        await tick(node, 3)

        assert node.durable.role is Role.FOLLOWER
        assert transport.of_kind("vote") == []


async def test_only_one_request_vote_per_term_is_in_flight_to_a_peer(db_path):
    transport = ScriptedTransport(lambda kind, peer, request: held_forever())
    async with SqliteStore(db_path) as store:
        node = await node_with(store, transport)
        for _ in range(LONGEST + 3):  # the election, then three heartbeat intervals
            await node.tick()
        await let_other_tasks_run()

        assert sorted(peer for peer, _ in transport.of_kind("vote")) == [2, 3]
        await node.stop()


# --- A Leader's AppendEntries (REPL-2, REPL-7, REPL-9, DD-26) ---------------------------


async def test_a_new_leader_sends_its_empty_entry_to_every_follower_at_once(db_path):
    transport = ScriptedTransport(everyone_agrees)
    async with SqliteStore(db_path) as store:
        node = await node_with(store, transport)
        await win(node)

        sent = transport.of_kind("append")

        assert sorted(peer for peer, _ in sent) == [2, 3]
        assert all(request.entries == (LogEntry.empty(1),) for _, request in sent)
        assert node.durable.commit_index == 1


async def test_a_leader_sends_every_follower_an_append_entries_each_heartbeat_interval(db_path):
    transport = ScriptedTransport(everyone_agrees)
    async with SqliteStore(db_path) as store:
        node = await node_with(store, transport, timing=Timing(heartbeat_ticks=3))
        await win(node)
        transport.sent.clear()

        per_tick = []
        for _ in range(6):
            await tick(node)
            per_tick.append(len(transport.of_kind("append")))
            transport.sent.clear()

        assert per_tick == [0, 0, 2, 0, 0, 2]


async def test_rejections_are_followed_by_resends_until_the_follower_is_repaired(db_path):
    # Follower 2 holds only the first entry; follower 3 holds nothing. With no tick in between,
    # each rejection brings the next, one index further back, until the logs agree (REPL-7).
    leader_log = [LogEntry(1, "a"), LogEntry(1, "b"), LogEntry(1, "c")]
    followers = FollowerLogs({2: Log(leader_log[:1]), 3: Log()})
    async with SqliteStore(db_path) as store:
        await store.replace_log_from(1, leader_log)
        await store.save_term_and_vote(1, None)
        node = await node_with(store, ScriptedTransport(followers))

        await win(node)

        assert followers.logs[2] == followers.logs[3] == node.durable.log
        assert node.durable.commit_index == 4


async def test_only_one_append_entries_is_in_flight_to_a_follower(db_path):
    release = asyncio.Event()

    def hold_follower_2(kind, peer, request):
        if kind == "append" and peer == 2 and not release.is_set():
            return accepted_once_released(release, request)
        return everyone_agrees(kind, peer, request)

    transport = ScriptedTransport(hold_follower_2)
    async with SqliteStore(db_path) as store:
        node = await node_with(store, transport)
        # NOTE: ticked without `idle`, which would wait for the held RPC forever.
        for _ in range(LONGEST):
            await node.tick()
        await eventually(lambda: node.durable.commit_index == 1)  # follower 3 answered
        await node.append_command("first")
        await node.append_command("second")
        to_two = [request for peer, request in transport.of_kind("append") if peer == 2]
        assert len(to_two) == 1

        release.set()
        await within_bound(node.idle())

        to_two = [request for peer, request in transport.of_kind("append") if peer == 2]
        assert len(to_two) == 2
        assert [entry.command for entry in to_two[1].entries] == ["first", "second"]


async def test_a_lost_append_entries_is_sent_again_at_the_next_heartbeat_not_before(db_path):
    # FAIL-2, FAIL-3, DD-30: no retry of its own; the next heartbeat interval sends a request
    # that starts where the lost one did.
    def follower_3_unreachable(kind, peer, request):
        return None if peer == 3 else everyone_agrees(kind, peer, request)

    transport = ScriptedTransport(follower_3_unreachable)
    async with SqliteStore(db_path) as store:
        node = await node_with(store, transport, timing=Timing(heartbeat_ticks=4))
        await win(node)
        (lost,) = [request for peer, request in transport.of_kind("append") if peer == 3]
        transport.sent.clear()

        await tick(node, 3)
        assert transport.of_kind("append") == []
        await tick(node)

        assert sorted(peer for peer, _ in transport.of_kind("append")) == [2, 3]
        (resent,) = [request for peer, request in transport.of_kind("append") if peer == 3]
        assert (resent.prev_log_index, resent.entries) == (lost.prev_log_index, lost.entries)


async def test_a_leader_deposed_while_its_sends_wait_for_the_lock_sends_nothing(db_path):
    # A higher-term RequestVote holds the lock, writing, when the heartbeat fires; by the time
    # the sends get the lock the node is a Follower, and each send stops quietly.
    transport = ScriptedTransport(everyone_agrees)
    async with GatedStore(db_path) as store:
        store.release.set()
        node = await node_with(store, transport)
        await win(node)
        transport.sent.clear()

        store.hold_next_write()
        deposing = asyncio.create_task(node.handle_request_vote(vote_request(term=5, candidate=3)))
        await store.wait_for_write()
        await node.tick()
        await let_other_tasks_run()
        store.release.set()
        await within_bound(deposing)
        await within_bound(node.idle())

        assert node.failure is None
        assert (node.durable.role, node.durable.current_term) == (Role.FOLLOWER, 5)
        assert transport.of_kind("append") == []


# --- Cluster time (DD-32) ---------------------------------------------------------------


async def test_only_a_leaders_ticks_count_as_cluster_time_and_each_entry_carries_it(db_path):
    transport = ScriptedTransport(everyone_agrees)
    async with SqliteStore(db_path) as store:
        node = await node_with(store, transport)
        await win(node)  # nineteen ticks begun as Follower, none counted
        assert node.durable.log.entry_at(1).cluster_time == 0

        await tick(node, 5)
        await node.append_command("put x")
        await within_bound(node.idle())

        assert node.durable.log.entry_at(2).cluster_time == 5
        sent = [request for _, request in transport.of_kind("append") if request.entries]
        assert sent[-1].entries[-1].cluster_time == 5  # Followers are sent the same time


async def test_cluster_time_counts_every_leader_tick_not_only_heartbeats(db_path):
    transport = ScriptedTransport(everyone_agrees)
    async with SqliteStore(db_path) as store:
        node = await node_with(store, transport, timing=Timing(heartbeat_ticks=3))
        await win(node)

        await tick(node, 5)
        await node.append_command("put x")

        assert node.durable.log.entry_at(2).cluster_time == 5


# --- Client commands (REPL-1, CLIENT-6) --------------------------------------------------


async def test_a_leader_sends_a_new_command_at_once_not_at_the_next_heartbeat(db_path):
    transport = ScriptedTransport(everyone_agrees)
    async with SqliteStore(db_path) as store:
        node = await node_with(store, transport, timing=Timing(heartbeat_ticks=5))
        await win(node)
        transport.sent.clear()

        position = await node.append_command("put x")
        await within_bound(node.idle())

        carried = [request.entries for _, request in transport.of_kind("append")]
        assert carried == [(LogEntry(1, "put x"),)] * 2
        assert position == LogPosition(term=1, index=2)
        assert node.durable.commit_index == 2


async def test_a_command_given_to_a_follower_is_refused(db_path):
    async with SqliteStore(db_path) as store:
        node = await node_with(store)
        with pytest.raises(NotLeaderError):
            await node.append_command("put x")


# --- Applying (APPLY-4, APPLY-5, DD-28) --------------------------------------------------


async def test_a_follower_applies_what_the_leader_commits_without_being_asked(db_path):
    kv = KeyValueStore()
    entries = [LogEntry(1, OpenSession().encode()), LogEntry(1, Put(1, 1, "x", "1").encode())]
    async with SqliteStore(db_path) as store:
        node = await node_with(store, apply=kv.apply)

        await node.handle_append_entries(
            append_entries(term=1, leader=2, entries=entries, leader_commit=2)
        )
        await within_bound(node.idle())

        assert kv.as_dict() == {"x": "1"}
        assert node.durable.last_applied == 2


async def test_a_node_alone_in_its_cluster_elects_itself_and_applies_its_commands(db_path):
    kv = KeyValueStore()
    async with SqliteStore(db_path) as store:
        node = await node_with(store, cluster=Cluster([NODE_ID]), apply=kv.apply)
        await tick(node, LONGEST)

        session = await node.append_command(OpenSession().encode())
        await node.append_command(Put(session.index, 1, "x", "1").encode())
        await within_bound(node.idle())

        assert node.durable.role is Role.LEADER
        assert kv.as_dict() == {"x": "1"}


async def test_a_node_alone_in_its_cluster_replays_its_log_as_soon_as_it_is_elected(db_path):
    # Its empty entry commits every entry before it, and applying starts with no new command.
    kv = KeyValueStore()
    async with SqliteStore(db_path) as store:
        await store.replace_log_from(
            1, [LogEntry(1, OpenSession().encode()), LogEntry(1, Put(1, 1, "x", "1").encode())]
        )
        await store.save_term_and_vote(1, None)
        node = await node_with(store, cluster=Cluster([NODE_ID]), apply=kv.apply)

        await tick(node, LONGEST)

        assert kv.as_dict() == {"x": "1"}
        assert node.durable.last_applied == 3


async def test_a_long_backlog_is_applied_in_batches(db_path, monkeypatch):
    # Applying runs without yielding, so it is split up to let ticks and RPCs in between.
    applied = []
    async with SqliteStore(db_path) as store:
        await store.replace_log_from(1, [LogEntry(1, "c")] * 2500)
        await store.save_term_and_vote(1, None)
        node = await node_with(store, apply=lambda index, time, command: applied.append(command))
        batches = []
        apply_committed = node.durable.apply_committed

        async def recording(max_entries=None, **options):
            batches.append(max_entries)
            return await apply_committed(max_entries, **options)

        monkeypatch.setattr(node.durable, "apply_committed", recording)
        await node.handle_append_entries(
            heartbeat(term=1, leader=2, prev_log_index=2500, prev_log_term=1, commit=2500)
        )
        await within_bound(node.idle())

        assert len(applied) == 2500
        assert batches == [1000, 1000, 1000]


# --- Proposing a command (CLIENT-4, CLIENT-5, DD-12, DD-33) ------------------------------


async def test_a_node_alone_in_its_cluster_answers_a_proposal_with_the_state_machines_result(
    db_path,
):
    # Its entry commits inside the append itself, so the proposal must already be waiting by the
    # time applying runs, or its answer is missed.
    kv = KeyValueStore()
    async with SqliteStore(db_path) as store:
        node = await node_with(store, cluster=Cluster([NODE_ID]), apply=kv.apply)
        await tick(node, LONGEST)

        opened = await within_bound(node.propose(OpenSession().encode()))
        first = await within_bound(node.propose(Put(opened.client_id, 1, "x", "1").encode()))
        second = await within_bound(node.propose(Put(opened.client_id, 2, "x", "2").encode()))

        assert opened == SessionOpened(client_id=2)  # index 1 is the Leader's empty entry
        assert (first, second) == (PutApplied(None), PutApplied("1"))
        assert node.pending_proposals == 0


async def test_a_proposal_returns_the_state_machines_refusal_of_a_command_that_took_no_effect(
    db_path,
):
    # A refused put is committed and applied and does nothing, so its caller must be told what
    # applying it returned, not that it was applied.
    kv = KeyValueStore()
    async with SqliteStore(db_path) as store:
        node = await node_with(store, cluster=Cluster([NODE_ID]), apply=kv.apply)
        await tick(node, LONGEST)
        opened = await within_bound(node.propose(OpenSession().encode()))
        await within_bound(node.propose(Put(opened.client_id, 2, "x", "2").encode()))

        stale = await within_bound(node.propose(Put(opened.client_id, 1, "x", "1").encode()))
        no_session = await within_bound(node.propose(Put(99, 1, "y", "1").encode()))

        assert (stale, no_session) == (StaleRequest(), SessionExpired())
        assert kv.as_dict() == {"x": "2"}


async def test_a_proposal_waits_until_its_entry_is_committed_and_applied(db_path):
    applied = []
    followers = HoldableFollowers()

    def apply(index, cluster_time, command):
        applied.append(command)
        return f"result of {command}"

    async with SqliteStore(db_path) as store:
        node = await node_with(store, ScriptedTransport(followers), apply=apply)
        await win(node)
        followers.hold()

        proposal = await start_proposal(node)
        await let_other_tasks_run()

        assert not proposal.done()
        assert node.durable.log.last_index == 2  # appended on the Leader, acknowledged by no one
        assert (node.durable.commit_index, applied) == (1, [])

        followers.release()
        assert await within_bound(proposal) == "result of x"
        assert (node.durable.commit_index, node.durable.last_applied, applied) == (2, 2, ["x"])
        assert node.pending_proposals == 0


async def test_proposals_wait_side_by_side_until_their_entries_are_applied(db_path):
    followers = HoldableFollowers()
    async with SqliteStore(db_path) as store:
        node = await node_with(store, ScriptedTransport(followers), apply=result_of)
        await win(node)
        followers.hold()

        first = await start_proposal(node, "a")
        second = await start_proposal(node, "b")
        assert node.pending_proposals == 2

        followers.release()
        assert await within_bound(first) == "result of a"
        assert await within_bound(second) == "result of b"
        assert node.pending_proposals == 0


async def test_a_proposal_has_no_time_limit_of_its_own(db_path):
    # Ticks are this layer's only clock: a Leader whose Followers never answer waits through
    # twenty election timeouts, and the proposal is still waiting for them.
    followers = HoldableFollowers()
    async with SqliteStore(db_path) as store:
        node = await node_with(store, ScriptedTransport(followers), apply=result_of)
        await win(node)
        followers.hold()
        proposal = await start_proposal(node)

        for _ in range(20 * TIMING.election_ticks):
            await node.tick()

        assert not proposal.done() and node.pending_proposals == 1
        followers.release()
        assert await within_bound(proposal) == "result of x"


async def test_concurrent_proposals_each_get_their_own_result(db_path):
    async with SqliteStore(db_path) as store:
        node = await node_with(store, ScriptedTransport(everyone_agrees), apply=result_of)
        await win(node)

        results = await within_bound(asyncio.gather(*(node.propose(c) for c in ("a", "b", "c"))))

        assert results == ["result of a", "result of b", "result of c"]
        assert node.pending_proposals == 0


async def test_a_proposal_to_a_follower_is_refused_with_nothing_left_waiting(db_path):
    async with SqliteStore(db_path) as store:
        node = await node_with(store)

        with pytest.raises(NotLeaderError):
            await node.propose("x")

        assert node.pending_proposals == 0
        assert node.durable.log.last_index == 0


async def test_a_proposal_of_the_empty_command_is_refused_with_nothing_left_waiting(db_path):
    async with SqliteStore(db_path) as store:
        node = await node_with(store, ScriptedTransport(everyone_agrees))
        await win(node)

        with pytest.raises(ValueError, match="empty command"):
            await node.propose("")

        assert node.pending_proposals == 0


DEPOSING_RPCS = {
    "a RequestVote of a later term": lambda node: node.handle_request_vote(
        vote_request(term=5, candidate=3)
    ),
    "an AppendEntries from a later term's Leader": lambda node: node.handle_append_entries(
        heartbeat(term=5, leader=3)
    ),
}


@pytest.mark.parametrize("depose", list(DEPOSING_RPCS.values()), ids=list(DEPOSING_RPCS))
async def test_a_proposal_fails_at_once_when_the_node_stops_being_leader_before_it_is_applied(
    db_path, depose
):
    # At once: no tick runs between the RPC and the failure.
    followers = HoldableFollowers()
    async with SqliteStore(db_path) as store:
        node = await node_with(store, ScriptedTransport(followers), apply=result_of)
        await win(node)
        followers.hold()
        proposal = await start_proposal(node)

        await depose(node)

        with pytest.raises(LeadershipLostError, match="no longer Leader of term 1"):
            await within_bound(proposal)
        followers.release()
        await within_bound(node.idle())
        assert node.pending_proposals == 0
        assert node.failure is None  # only the proposal ended; the node carries on


async def test_a_proposal_fails_when_a_followers_answer_shows_a_later_term(db_path):
    later = asyncio.Event()

    def answer(kind, peer, request):
        if kind == "append" and later.is_set():
            return rejected(term=5)
        return everyone_agrees(kind, peer, request)

    async with SqliteStore(db_path) as store:
        node = await node_with(store, ScriptedTransport(answer), apply=result_of)
        await win(node)
        later.set()

        with pytest.raises(LeadershipLostError):
            await within_bound(node.propose("x"))

        assert (node.durable.role, node.durable.current_term) == (Role.FOLLOWER, 5)
        assert node.pending_proposals == 0


@pytest.mark.parametrize("depose", list(DEPOSING_RPCS.values()), ids=list(DEPOSING_RPCS))
async def test_a_step_down_installed_before_its_caller_was_cancelled_still_fails_the_proposals(
    db_path, depose
):
    # The RPC's write completes and is installed even though its caller is cancelled (DD-22),
    # so the node has stepped down: its proposals must fail now, not at the next tick.
    followers = HoldableFollowers()
    async with GatedStore(db_path) as store:
        store.release.set()
        node = await node_with(store, ScriptedTransport(followers), apply=result_of)
        await win(node)
        followers.hold()
        proposal = await start_proposal(node)

        store.hold_next_write()
        handler = asyncio.create_task(depose(node))
        await store.wait_for_write()
        handler.cancel()
        store.release.set()
        with pytest.raises(asyncio.CancelledError):
            await within_bound(handler)

        assert (node.durable.role, node.durable.current_term) == (Role.FOLLOWER, 5)
        with pytest.raises(LeadershipLostError):
            await within_bound(proposal)
        assert node.pending_proposals == 0


async def test_a_caller_that_gives_up_leaves_nothing_waiting_and_its_entry_still_commits(db_path):
    applied = []
    followers = HoldableFollowers()
    async with SqliteStore(db_path) as store:
        node = await node_with(
            store,
            ScriptedTransport(followers),
            apply=lambda index, cluster_time, command: applied.append(command),
        )
        await win(node)
        followers.hold()
        proposal = await start_proposal(node)

        proposal.cancel()
        with pytest.raises(asyncio.CancelledError):
            await proposal
        assert node.pending_proposals == 0

        followers.release()
        await within_bound(node.idle())
        assert (node.durable.commit_index, applied) == (2, ["x"])  # it went on without its caller


async def test_a_proposal_waiting_when_the_node_stops_fails_with_node_stopped(db_path):
    followers = HoldableFollowers()
    async with SqliteStore(db_path) as store:
        node = await node_with(store, ScriptedTransport(followers))
        await win(node)
        followers.hold()
        proposal = await start_proposal(node)

        await within_bound(node.stop())

        with pytest.raises(NodeStoppedError):
            await within_bound(proposal)
        assert node.pending_proposals == 0


async def test_a_command_appended_while_the_node_stops_fails_instead_of_waiting(db_path):
    # Stopping waits for the write in progress, which then returns a position nothing will ever
    # apply; the proposal must notice the node is closed instead of waiting on it forever.
    async with GatedStore(db_path) as store:
        store.release.set()
        node = await node_with(store, ScriptedTransport(everyone_agrees))
        await win(node)

        store.hold_next_write()
        proposal = asyncio.create_task(node.propose("x"))
        await store.wait_for_write()
        stopping = asyncio.create_task(node.stop())
        await let_other_tasks_run()
        store.release.set()
        await within_bound(stopping)

        with pytest.raises(NodeStoppedError):
            await within_bound(proposal)
        assert node.pending_proposals == 0


async def test_a_state_machine_that_raises_fails_the_proposal_waiting_on_it(db_path):
    def broken(index, cluster_time, command):
        raise ValueError("cannot apply")

    async with SqliteStore(db_path) as store:
        node = await node_with(store, ScriptedTransport(everyone_agrees), apply=broken)
        await win(node)

        with pytest.raises(NodeStoppedError):
            await within_bound(node.propose("boom"))

        assert isinstance(node.failure, ValueError)
        assert node.pending_proposals == 0


async def test_a_proposal_is_answered_when_tasks_start_eagerly(db_path):
    # Eagerly started tasks run at once, so applying happens inside the append itself; the wait
    # must be registered before that, whichever way tasks start.
    loop = asyncio.get_running_loop()
    loop.set_task_factory(asyncio.eager_task_factory)
    try:
        kv = KeyValueStore()
        async with SqliteStore(db_path) as store:
            node = await node_with(store, cluster=Cluster([NODE_ID]), apply=kv.apply)
            await tick(node, LONGEST)

            opened = await within_bound(node.propose(OpenSession().encode()))

            assert opened == SessionOpened(client_id=2)
    finally:
        loop.set_task_factory(None)


def test_the_two_ways_a_proposal_fails_are_unrelated_errors():
    # A caller that handles one must not catch the other by accident.
    assert not issubclass(LeadershipLostError, NodeStoppedError)
    assert not issubclass(NodeStoppedError, LeadershipLostError)


# --- Failing and stopping ------------------------------------------------------------------


async def test_a_state_machine_that_raises_stops_the_node(db_path):
    def broken(index, cluster_time, command):
        raise ValueError("cannot apply")

    async with SqliteStore(db_path) as store:
        node = await node_with(store, apply=broken)
        await node.handle_append_entries(
            append_entries(term=1, leader=2, entries=[LogEntry(1, "x")], leader_commit=1)
        )
        await within_bound(node.idle())

        assert isinstance(node.failure, ValueError)
        with pytest.raises(NodeStoppedError):
            await node.tick()


async def test_a_failure_cancels_the_nodes_other_tasks(db_path):
    # Follower 2's AppendEntries is held in flight when applying the command fails.
    def broken(index, cluster_time, command):
        raise ValueError("cannot apply")

    hold = asyncio.Event()

    def hold_follower_2(kind, peer, request):
        if hold.is_set() and peer == 2:
            return held_forever()
        return everyone_agrees(kind, peer, request)

    async with SqliteStore(db_path) as store:
        node = await node_with(store, ScriptedTransport(hold_follower_2), apply=broken)
        await win(node)
        hold.set()

        await node.append_command("boom")
        await within_bound(node.idle())

        assert isinstance(node.failure, ValueError)
        assert not node.busy


async def test_an_append_entries_that_would_change_a_committed_entry_stops_the_node(db_path):
    # DD-29: answering would only have the Leader send the same entries again, forever.
    async with SqliteStore(db_path) as store:
        node = await node_with(store, apply=lambda index, cluster_time, command: None)
        await node.handle_append_entries(
            append_entries(
                term=1, leader=2, entries=[LogEntry(1, "a"), LogEntry(1, "b")], leader_commit=2
            )
        )
        await within_bound(node.idle())

        with pytest.raises(CommittedEntryConflictError):
            await node.handle_append_entries(
                append_entries(term=2, leader=3, entries=[LogEntry(2, "x")])
            )

        assert isinstance(node.failure, CommittedEntryConflictError)
        with pytest.raises(NodeStoppedError):
            await node.tick()


async def test_a_stopped_node_refuses_every_call(db_path):
    async with SqliteStore(db_path) as store:
        node = await node_with(store)
        await node.stop()

        with pytest.raises(NodeStoppedError):
            node.start()
        with pytest.raises(NodeStoppedError):
            await node.tick()
        with pytest.raises(NodeStoppedError):
            await node.append_command("x")
        with pytest.raises(NodeStoppedError):
            await node.propose("x")
        with pytest.raises(NodeStoppedError):
            await node.handle_request_vote(vote_request(term=1, candidate=2))
        with pytest.raises(NodeStoppedError):
            await node.handle_append_entries(heartbeat(term=1, leader=2))


async def test_stopping_waits_for_an_rpc_still_being_handled(db_path):
    # Its caller may close the store once stop returns, so nothing may still be writing.
    async with GatedStore(db_path) as store:
        node = await node_with(store)
        handling = asyncio.create_task(node.handle_append_entries(heartbeat(term=1, leader=2)))
        await store.wait_for_write()

        stopping = asyncio.create_task(node.stop())
        await let_other_tasks_run()
        assert not stopping.done()

        store.release.set()
        await within_bound(stopping)
        assert handling.done() and handling.result().success is True


async def test_a_command_written_while_the_node_stops_is_not_sent(db_path):
    transport = ScriptedTransport(everyone_agrees)
    async with GatedStore(db_path) as store:
        store.release.set()
        node = await node_with(store, transport)
        await win(node)
        transport.sent.clear()

        store.hold_next_write()
        command = asyncio.create_task(node.append_command("x"))
        await store.wait_for_write()
        stopping = asyncio.create_task(node.stop())
        await let_other_tasks_run()
        store.release.set()
        await within_bound(asyncio.gather(command, stopping))

        assert transport.of_kind("append") == []
        assert not node.busy


# --- Knowing the Leader (CLIENT-7) ------------------------------------------------------


async def test_a_node_knows_the_leader_of_its_current_term_and_forgets_it_on_a_new_term(db_path):
    async with SqliteStore(db_path) as store:
        node = await node_with(store, ScriptedTransport(everyone_agrees))
        assert node.leader_id is None

        await node.handle_append_entries(
            heartbeat(term=1, leader=2, prev_log_index=5, prev_log_term=1)
        )
        assert node.leader_id == 2  # a rejection comes from the Leader all the same

        await node.handle_request_vote(vote_request(term=2, candidate=3))
        assert node.leader_id is None

        await win(node)
        assert node.leader_id == NODE_ID


async def test_a_leader_that_hears_from_a_later_leader_follows_it(db_path):
    async with SqliteStore(db_path) as store:
        node = await node_with(store, ScriptedTransport(everyone_agrees))
        await win(node)

        await node.handle_append_entries(heartbeat(term=2, leader=3))

        assert node.durable.role is Role.FOLLOWER
        assert node.leader_id == 3


# --- Running on its own clock -----------------------------------------------------------


async def test_a_started_node_ticks_by_itself_until_stopped(db_path):
    transport = ScriptedTransport()
    async with SqliteStore(db_path) as store:
        node = await node_with(store, transport, timing=Timing(tick_interval=0.001))
        node.start()
        await eventually(lambda: transport.sent)
        await node.stop()
        term = node.durable.current_term
        await asyncio.sleep(0.05)

        assert term >= 1
        assert node.durable.current_term == term


async def test_a_started_node_refuses_to_be_ticked_by_hand(db_path):
    async with SqliteStore(db_path) as store:
        node = await node_with(store, timing=Timing(tick_interval=10))
        node.start()
        with pytest.raises(RuntimeError):
            await node.tick()
        await node.stop()
