"""InProcessCluster: real nodes, each with its own SQLite file, whose RPCs are direct calls.

The test decides whose election timeout fires, which Leader sends when, and which messages arrive,
in what order, and which are lost or repeated. Each voter's vote is read back from its file before
its answer is handed on (PERSIST-1, PERSIST-2); every committed entry is recorded and checked
against every node that commits it and every later Leader; and every step is reported to the
trace (tests/conftest.py).
"""

import logging
from collections import defaultdict
from dataclasses import dataclass

from raftkv.consensus import (
    AppendEntriesRequest,
    AppendEntriesResponse,
    Cluster,
    RequestVoteRequest,
    RequestVoteResponse,
    Role,
)
from raftkv.kvstore import KeyValueStore
from raftkv.persistence import DurableNodeState, SqliteStore
from tests.divergent_logs import make_log
from tests.election_traces.recorder import HARNESS
from tests.persistence.store_doubles import reload

_harness = logging.getLogger(HARNESS)


def trace_step(source, message, *args, event=None):
    """Report one harness step to the election trace, if one is being recorded.

    Args:
        source: "net", "clock", "cmd", "apply", "crash", "disk", or "state".
        message: A %-style format string, filled from `args`.
        event: The step as a structured event, for the `.jsonl` trace.
    """
    if _harness.isEnabledFor(logging.INFO):
        _harness.info(message, *args, extra={"trace_source": source, "trace_event": event})


@dataclass(frozen=True)
class InFlight:
    """A RequestVote on its way to `voter`, or, once `response` is set, the answer coming back."""

    request: RequestVoteRequest
    voter: int
    response: RequestVoteResponse | None = None

    def describe(self):
        """Return the message as one line of the `.log` trace."""
        if self.response is None:
            return (
                f"RequestVote(term {self.request.term}) "
                f"from {self.request.candidate_id} to {self.voter}"
            )
        answer = "granted" if self.response.vote_granted else "refused"
        return (
            f"RequestVoteResponse(term {self.response.term}, {answer}) "
            f"from {self.voter} to {self.request.candidate_id}"
        )

    def trace_fields(self):
        """Return the message as trace-event fields, in etcd's `msg` shape."""
        if self.response is None:
            return {
                "type": "RequestVote",
                "term": self.request.term,
                "from": self.request.candidate_id,
                "to": self.voter,
            }
        return {
            "type": "RequestVoteResponse",
            "term": self.response.term,
            "from": self.voter,
            "to": self.request.candidate_id,
            "reject": not self.response.vote_granted,
        }


@dataclass(frozen=True)
class AppendEntriesInFlight:
    """An AppendEntries on its way to `follower`, or, once `response` is set, the answer back."""

    request: AppendEntriesRequest
    follower: int
    response: AppendEntriesResponse | None = None

    def describe(self):
        """Return the message as one line of the `.log` trace."""
        request = self.request
        if self.response is None:
            return (
                f"AppendEntries(term {request.term}, after {request.prev_log_index}"
                f"@t{request.prev_log_term}, {len(request.entries)} entries, "
                f"commit {request.leader_commit}) from {request.leader_id} to {self.follower}"
            )
        answer = "accepted" if self.response.success else "rejected"
        return (
            f"AppendEntriesResponse(term {self.response.term}, {answer}) "
            f"from {self.follower} to {request.leader_id}"
        )

    def trace_fields(self):
        """Return the message as trace-event fields, in etcd's `msg` shape."""
        if self.response is None:
            return {
                "type": "AppendEntries",
                "term": self.request.term,
                "from": self.request.leader_id,
                "to": self.follower,
                "index": self.request.prev_log_index,
                "logTerm": self.request.prev_log_term,
                "entries": len(self.request.entries),
                "commit": self.request.leader_commit,
            }
        return {
            "type": "AppendEntriesResponse",
            "term": self.response.term,
            "from": self.follower,
            "to": self.request.leader_id,
            "reject": not self.response.success,
        }


class InProcessCluster:
    """Real nodes whose RPCs are direct calls; nothing happens unless the test does it.

    Messages are delivered at once (`ask_for_vote`, `deliver_vote_response`, `run_election`,
    `replicate`) or put in flight (`send_requests`, `send_append_entries`) to be delivered,
    duplicated, or dropped later, in any order. Every Leader and every granted vote is recorded
    per term, for `assert_election_safety`. Every committed entry is recorded the first time any
    node commits it, and after each step every committing node and every Leader is checked to
    still hold it (`assert_log_safety`).

    Attributes:
        paths: Each node's SQLite file.
        nodes: Each running node's DurableNodeState.
        in_flight: The messages in flight, as InFlight.
        leaders_by_term: Every node ever seen as Leader, per term.
        votes_by_voter_and_term: Every Candidate each node granted its vote, per term,
            its own vote as a Candidate included.
        committed: Every committed entry, by index, with the term of the node first seen
            committing it, as (entry, term).
    """

    def __init__(self, directory, member_ids, check_votes_on_disk=True):
        self.cluster = Cluster(member_ids)
        self.paths = {n: str(directory / f"node-{n}.db") for n in member_ids}
        self.nodes = {}
        self.in_flight = []
        self.leaders_by_term = defaultdict(set)
        self.votes_by_voter_and_term = defaultdict(set)
        self.committed = {}
        self.applied = {}  # index -> the command the first node seen at that index applied
        self.kv = {}
        self._stores = {}
        self._check_votes_on_disk = check_votes_on_disk

    @property
    def member_ids(self):
        """Return every member's ID, in ascending order."""
        return sorted(self.cluster.members)

    async def preload(self, node_id, log_terms=(), current_term=0, voted_for=None):
        """Write a starting log, term, and vote straight to a stopped node's file."""
        async with SqliteStore(self.paths[node_id]) as store:
            await store.replace_log_from(1, list(make_log(list(log_terms))))
            await store.save_term_and_vote(current_term, voted_for)

    async def start(self, node_id):
        """Start a node from its file, with an empty state machine.

        The state machine is not persisted, so a restart gets a fresh one and rebuilds it by
        applying the log again (APPLY-4).
        """
        store = SqliteStore(self.paths[node_id])
        await store.__aenter__()
        self._stores[node_id] = store
        self.kv[node_id] = KeyValueStore()
        self.nodes[node_id] = await DurableNodeState.load(
            node_id, store, self.cluster, apply=self.kv[node_id].apply
        )

    async def stop(self, node_id):
        """Stop a node, losing everything it holds in memory."""
        del self.nodes[node_id]
        await self._stores.pop(node_id).__aexit__(None, None, None)

    async def restart(self, node_id):
        """Crash a node and bring it back from nothing but its file."""
        trace_step(
            "crash",
            "node %d crashes: its memory is lost; it restarts from its file",
            node_id,
            event={"name": "Restart", "nid": node_id},
        )
        await self.stop(node_id)
        await self.start(node_id)
        self._trace_cluster_state()

    async def start_all(self):
        """Start every member."""
        for node_id in self.member_ids:
            await self.start(node_id)

    async def stop_all(self):
        """Stop every running node."""
        for node_id in list(self.nodes):
            await self.stop(node_id)

    async def fire_election_timeout(self, candidate):
        """Fire `candidate`'s election timeout: it starts an election; return its RequestVote."""
        trace_step(
            "clock",
            "election timeout fires on node %d",
            candidate,
            event={"name": "Timeout", "nid": candidate},
        )
        request = await self.nodes[candidate].start_election()
        self.votes_by_voter_and_term[(candidate, request.term)].add(candidate)
        # NOTE: in a one-node cluster the self-vote already wins the term inside start_election,
        # with no answer left to deliver, so the Leader is noted here.
        self._check_after_step()
        self._trace_cluster_state()
        return request

    async def ask_for_vote(self, voter, request):
        """Deliver a RequestVote to `voter` and return its answer, read back from disk first.

        With `check_votes_on_disk`, the test fails unless the voter's file holds the term and
        vote it answered with.
        """
        message = InFlight(request, voter)
        trace_step(
            "net",
            "deliver %s",
            message.describe(),
            event={"name": "Deliver", "msg": message.trace_fields()},
        )
        response = await self.nodes[voter].handle_vote_request(request)
        if response.vote_granted:
            self.votes_by_voter_and_term[(voter, request.term)].add(request.candidate_id)
        # NOTE: with checking off the file is still read whenever tracing is on, so a mismatch
        # reaches the trace.
        if self._check_votes_on_disk or _harness.isEnabledFor(logging.INFO):
            await self._check_vote_on_disk(voter)
        self._check_after_step()
        self._trace_cluster_state()
        return response

    async def deliver_vote_response(self, voter, request, response):
        """Deliver `voter`'s answer to `request` to the Candidate; return whether it won."""
        message = InFlight(request, voter, response)
        trace_step(
            "net",
            "deliver %s",
            message.describe(),
            event={"name": "Deliver", "msg": message.trace_fields()},
        )
        # NOTE: the election an answer belongs to comes from the request's term, never the
        # answer's, since a refusal can carry a later term than the one it was sent in.
        became_leader = await self.nodes[request.candidate_id].handle_vote_response(
            voter, request.term, response
        )
        self._check_after_step()
        self._trace_cluster_state()
        return became_leader

    def send_requests(self, request):
        """Put a Candidate's RequestVote in flight to every one of its peers."""
        for voter in sorted(self.nodes[request.candidate_id].peers):
            self.in_flight.append(InFlight(request, voter))

    async def deliver(self, message, keep_copy=False):
        """Deliver an in-flight message; a delivered request's answer goes in flight.

        Args:
            message: One of `in_flight`.
            keep_copy: Leave a copy in flight, to arrive again later, as a network that
                duplicates messages would.

        Returns:
            The voter's answer if `message` was a request; None if it was an answer.
        """
        if keep_copy:
            trace_step(
                "net",
                "duplicate %s: a copy stays in flight",
                message.describe(),
                event={"name": "Duplicate", "msg": message.trace_fields()},
            )
        else:
            self.in_flight.remove(message)
        if isinstance(message, AppendEntriesInFlight):
            if message.response is None:
                response = await self.deliver_append_entries(message.follower, message.request)
                self.in_flight.append(
                    AppendEntriesInFlight(message.request, message.follower, response)
                )
                return response
            await self.deliver_append_entries_response(
                message.follower, message.request, message.response
            )
            return None
        if message.response is None:
            response = await self.ask_for_vote(message.voter, message.request)
            # NOTE: every delivery is answered afresh, so a duplicated request leaves a second
            # answer in flight, to arrive whenever the test delivers it (FAIL-2).
            self.in_flight.append(InFlight(message.request, message.voter, response))
            return response
        await self.deliver_vote_response(message.voter, message.request, message.response)
        return None

    def drop(self, message):
        """Lose an in-flight message."""
        self.in_flight.remove(message)
        trace_step(
            "net",
            "drop %s",
            message.describe(),
            event={"name": "Drop", "msg": message.trace_fields()},
        )

    async def run_election(self, candidate, reachable=None):
        """Time out `candidate`, then deliver its requests and their answers at once.

        Requests go to every running peer, or only to those in `reachable`.

        Returns:
            The election's RequestVote.
        """
        request = await self.fire_election_timeout(candidate)
        # NOTE: the loop runs on after a win or a step-down, as every request left at once and
        # the remaining answers still arrive at a node that no longer needs them.
        for voter in sorted(self.nodes[candidate].peers):
            # NOTE: a skipped voter's request is simply lost: never answered, never retried.
            if voter not in self.nodes or (reachable is not None and voter not in reachable):
                continue
            response = await self.ask_for_vote(voter, request)
            await self.deliver_vote_response(voter, request, response)
        return request

    # --- Replication -------------------------------------------------------------------

    async def append_command(self, leader, command):
        """Have `leader` append a client command to its log; return the entry's index."""
        trace_step(
            "cmd",
            "a client asks node %d to store %r",
            leader,
            command,
            event={"name": "ClientCommand", "nid": leader, "command": command},
        )
        index = await self.nodes[leader].append_command(command)
        self._check_after_step()
        self._trace_cluster_state()
        return index

    async def send_append_entries(self, leader, follower):
        """Put `leader`'s current AppendEntries for `follower` in flight; return the message."""
        request = await self.nodes[leader].append_entries_request_for(follower)
        message = AppendEntriesInFlight(request, follower)
        self.in_flight.append(message)
        return message

    async def deliver_append_entries(self, follower, request):
        """Deliver an AppendEntries to `follower` and return its answer."""
        message = AppendEntriesInFlight(request, follower)
        trace_step(
            "net",
            "deliver %s",
            message.describe(),
            event={"name": "Deliver", "msg": message.trace_fields()},
        )
        response = await self.nodes[follower].handle_append_entries(request)
        self._check_after_step()
        self._trace_cluster_state()
        return response

    async def deliver_append_entries_response(self, follower, request, response):
        """Deliver `follower`'s answer to the Leader; return whether the Leader should resend."""
        message = AppendEntriesInFlight(request, follower, response)
        trace_step(
            "net",
            "deliver %s",
            message.describe(),
            event={"name": "Deliver", "msg": message.trace_fields()},
        )
        resend = await self.nodes[request.leader_id].handle_append_entries_response(
            follower, request, response
        )
        self._check_after_step()
        self._trace_cluster_state()
        return resend

    async def replicate(self, leader, follower):
        """Send AppendEntries from `leader` to `follower`, and the answer back, until no resend.

        That is an acceptance, a step-down on a higher term, or a rejection with nothing earlier
        left to send. Each rejection lowers `next_index` by one, and a request with
        `prev_log_index` 0 always matches, so more rejections than the first request's
        `prev_log_index` fails the test.

        Returns:
            How many AppendEntries the Follower rejected first.
        """
        rejections, most_rejections = 0, None
        while True:
            request = await self.nodes[leader].append_entries_request_for(follower)
            # NOTE: the bound comes from the first request: each back-off lowers prev_log_index,
            # so a later request's would shrink the bound as the loop runs.
            if most_rejections is None:
                most_rejections = request.prev_log_index
            response = await self.deliver_append_entries(follower, request)
            resend = await self.deliver_append_entries_response(follower, request, response)
            if not resend:
                return rejections
            rejections += 1
            assert rejections <= most_rejections, "never reached a matching entry"

    async def replicate_to_all(self, leader):
        """Replicate from `leader` to every running peer, lowest ID first.

        A peer stays untouched once `leader` has stepped down on a reply.
        """
        for follower in sorted(self.nodes[leader].peers):
            if follower not in self.nodes or self.nodes[leader].role is not Role.LEADER:
                continue
            await self.replicate(leader, follower)

    def assert_log_safety(self):
        """Assert that no committed entry was ever changed or lost.

        Records every node's committed and applied entries the first time they are seen, then
        asserts: no node applies past what it has committed; every node applies the same command
        at each index (APPLY-4, APPLY-6); no node commits past the end of its log; every node that
        has committed an index holds the entry first committed there; and every running Leader
        whose term is at least the term that entry was first seen committed in holds it. A Leader
        of an earlier term, and a lagging Follower, may lack an entry committed after them.
        """
        for node_id, node in self.nodes.items():
            assert node.last_applied <= node.commit_index, (
                f"node {node_id} applied through {node.last_applied} but has only committed "
                f"{node.commit_index}"
            )
            for index in range(1, node.last_applied + 1):
                command = node.log.entry_at(index).command
                first = self.applied.setdefault(index, command)
                assert command == first, (
                    f"node {node_id} applied {command!r} at index {index}, where {first!r} was "
                    "applied before"
                )
            assert node.commit_index <= node.log.last_index, (
                f"node {node_id} has commit index {node.commit_index} but holds only "
                f"{node.log.last_index} entries"
            )
            for index in range(1, node.commit_index + 1):
                entry = node.log.entry_at(index)
                first, _ = self.committed.setdefault(index, (entry, node.current_term))
                assert entry == first, (
                    f"node {node_id} committed {entry} at index {index}, but {first} was "
                    "committed there before"
                )
        for node_id, node in self.nodes.items():
            if node.role is not Role.LEADER:
                continue
            for index, (entry, term) in self.committed.items():
                if node.current_term < term:
                    continue
                held = node.log.entry_at(index) if index <= node.log.last_index else None
                assert held == entry, (
                    f"leader {node_id} of term {node.current_term} holds {held} at index "
                    f"{index}, where {entry} was committed in term {term}"
                )

    async def apply_everywhere(self):
        """Let every running node apply everything it has committed."""
        for node_id in sorted(self.nodes):
            await self.apply_on(node_id)

    async def apply_on(self, node_id):
        """Let one node apply everything it has committed; return how many commands it ran."""
        applied = await self.nodes[node_id].apply_committed()
        if applied:
            trace_step(
                "apply",
                "node %d applied %d command(s), through index %d",
                node_id,
                applied,
                self.nodes[node_id].last_applied,
                event={
                    "name": "ApplyStep",
                    "nid": node_id,
                    "applied": self.nodes[node_id].last_applied,
                },
            )
        self._check_after_step()
        self._trace_cluster_state()
        return applied

    def maps(self):
        """Return each running node's key-value map, by node ID."""
        return {
            node_id: store.as_dict() for node_id, store in self.kv.items() if node_id in self.nodes
        }

    def log_terms(self):
        """Return each running node's log as a list of entry terms, by node ID."""
        return {n: [e.term for e in node.log] for n, node in self.nodes.items()}

    def leaders(self):
        """Return the running nodes that are Leader now, of any term."""
        return {n for n, node in self.nodes.items() if node.role is Role.LEADER}

    def assert_election_safety(self):
        """Assert that no term had two Leaders and no node voted for two Candidates in one term.

        Also asserts that every running node holds exactly its role's record for its term: a
        Candidacy while Candidate, a Leadership while Leader, and neither while Follower.
        """
        for term, leaders in self.leaders_by_term.items():
            assert len(leaders) == 1, f"term {term} had leaders {sorted(leaders)}"
        for (voter, term), candidates in self.votes_by_voter_and_term.items():
            assert len(candidates) == 1, (
                f"node {voter} voted for {sorted(candidates)} in term {term}"
            )
        for node in self.nodes.values():
            _assert_role_record_matches(node)

    async def _check_vote_on_disk(self, voter):
        node = self.nodes[voter]
        # NOTE: a second connection reads the committed file, not the node's own connection,
        # so the check proves durability.
        persisted = await reload(self.paths[voter])
        on_disk = (persisted.current_term, persisted.voted_for)
        in_memory = (node.current_term, node.voted_for)
        trace_step(
            "disk",
            "node %d's file holds term %d, vote %d — %s",
            voter,
            on_disk[0],
            on_disk[1] or 0,
            "same as memory"
            if on_disk == in_memory
            else f"MEMORY HAS term {in_memory[0]}, vote {in_memory[1] or 0}",
            event={
                "name": "DiskCheck",
                "nid": voter,
                "ok": on_disk == in_memory,
                "disk": {"term": on_disk[0], "vote": on_disk[1]},
                "memory": {"term": in_memory[0], "vote": in_memory[1]},
            },
        )
        if self._check_votes_on_disk:
            assert on_disk == in_memory

    def _check_after_step(self):
        """Record every running Leader, per term, and assert that no committed entry was lost."""
        for node_id, node in self.nodes.items():
            if node.role is Role.LEADER:
                self.leaders_by_term[node.current_term].add(node_id)
        self.assert_log_safety()

    def _trace_cluster_state(self):
        """Report each running node's role, term, vote, log length, commit index, and votes."""
        if not _harness.isEnabledFor(logging.INFO):
            return
        parts, nodes = [], {}
        for node_id in self.member_ids:
            node = self.nodes.get(node_id)
            if node is None:
                parts.append(f"{node_id} down")
                continue
            part = (
                f"{node_id} {node.role.value} t{node.current_term} v{node.voted_for or 0}"
                f" log{node.log.last_index} c{node.commit_index}"
            )
            if node.candidacy is not None:
                part += f" votes{sorted(node.candidacy.votes_granted)}"
            parts.append(part)
            nodes[node_id] = {
                "role": node.role.value,
                "term": node.current_term,
                "vote": node.voted_for,
                "lastIndex": node.log.last_index,
                "commit": node.commit_index,
            }
        trace_step("state", "%s", " | ".join(parts), event={"name": "ClusterState", "nodes": nodes})


def _assert_role_record_matches(node):
    if node.role is Role.CANDIDATE:
        assert node.candidacy.term == node.current_term and node.leadership is None
    elif node.role is Role.LEADER:
        assert node.leadership.term == node.current_term and node.candidacy is None
    else:
        assert node.candidacy is None and node.leadership is None
