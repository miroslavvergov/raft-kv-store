"""The driver that runs a node by itself: its clock, its RPCs, applying, proposals, and reads."""

import asyncio
import random

from raftkv.consensus import (
    AppendEntriesRequest,
    AppendEntriesResponse,
    CommittedEntryConflictError,
    LogPosition,
    NotLeaderError,
    RequestVoteRequest,
    RequestVoteResponse,
    Role,
)
from raftkv.node import DurableNodeState
from raftkv.runtime.errors import LeadershipLostError, NodeStoppedError, PeerUnreachableError
from raftkv.runtime.pending_proposals import PendingProposals
from raftkv.runtime.pending_reads import PendingReads
from raftkv.runtime.task_supervisor import TaskSupervisor
from raftkv.runtime.timing import Timing
from raftkv.runtime.transport import Transport

# NOTE: applying runs without yielding to the event loop, so a long backlog is applied in
# batches of this many entries, with a yield between them, to keep ticks and RPCs on time.
_APPLY_BATCH_SIZE = 1000


class RaftNode:
    """Runs one `DurableNodeState`: advances its clock, sends its RPCs, applies what commits.

    Time is counted in ticks (DD-9). Either the caller calls `tick`, as tests do,
    or `start` ticks every `Timing.tick_interval` seconds until `stop`. On each tick:

    - A Follower or Candidate counts toward its election timeout; when it is
      reached, the node starts an election and sends RequestVote to every peer
      (ELECT-2, ELECT-6). A Candidate that times out starts a new one (STATE-3).
    - A Candidate resends its RequestVote, every heartbeat interval, to each peer
      that has not answered yet (FAIL-2, DD-30).
    - A Leader sends each Follower its AppendEntries every heartbeat interval
      (REPL-9), with whatever entries that Follower lacks.
    - A Leader counts each tick as one unit of cluster time (DD-32), which it
      stamps on the entries it appends.

    The election timeout restarts with a newly drawn length (ELECT-1) when an
    AppendEntries from the current term's Leader arrives, accepted or not
    (ELECT-2); when this node grants a vote (ELECT-13); and when its role or term
    changes (DD-9). A timeout reached while another call holds the node's lock
    starts no election if that call restarted it.

    A Leader also sends at once when it appends an entry, and sends a Follower
    again at once after a rejection that lowered its `next_index` (REPL-7), after
    a success that leaves it short of the Leader's log, and after an answer
    carrying the request's term while a read still awaits that Follower's answer
    to a request built after the read began (DD-34). At most one AppendEntries
    and one RequestVote per term are in flight to each peer; a send wanted
    meanwhile is skipped, since the answer to the one in flight brings the next
    send and the next heartbeat interval sends in any case. An RPC that gets no
    answer is not retried on its own (DD-30).

    Committed entries are applied by a task started whenever the commit index may
    have moved (APPLY-4, APPLY-5, DD-28). An unexpected error in any task, a state
    machine that raises, or an AppendEntries that would change a committed entry
    (DD-29) stops the node and is kept in `failure`.

    A Leader's caller proposes a command with `propose`, which returns what the state
    machine returned once the command's entry is committed and applied (CLIENT-3,
    CLIENT-4, CLIENT-5, DD-12). It fails if the node stops leading, or stops, before
    that; the failure says the outcome is open, since the command may still take
    effect (DD-33). It sets no time limit of its own: the caller sets one.

    A Leader's caller reads with `read_barrier`, which returns the read index, the
    commit index when the read began, once the state machine has applied through
    it; the state machine may then be read. The read waits until the Leader has
    committed an entry of its own term, then until a majority has answered a request
    built after the read began. Such requests go out at once to each Follower with
    none in flight, and to any other as soon as its request is answered (CLIENT-8,
    CLIENT-9, CLIENT-10, DD-12, DD-14, DD-34). It fails with `NotLeaderError` if the
    node is not Leader or stops leading first, and with `NodeStoppedError` if the
    node stops. It sets no time limit of its own: the caller sets one.

    Attributes:
        node_id: This node's ID.
        durable: The node state this driver runs. Read it freely; change it only
            through this driver, which otherwise notices a change only at its next
            event.
        leader_id: The Leader of the current term, as far as this node knows; None
            if it knows none yet. For redirecting clients (CLIENT-7).
        election_timeout: The current election timeout, in ticks.
        failure: The error that stopped the node, or None.
        busy: Whether any send or apply task is still running.
        pending_proposals: How many proposals are waiting to be applied.
        pending_reads: How many reads are waiting to be served.
    """

    def __init__(
        self,
        durable: DurableNodeState,
        transport: Transport,
        timing: Timing | None = None,
        rng: random.Random | None = None,
    ) -> None:
        """Wrap a loaded node; nothing runs until `tick` or `start` is called.

        Args:
            durable: The node's state, as loaded from its store.
            transport: How this node reaches its peers.
            timing: The tick length and timeouts; `Timing()` if omitted.
            rng: The source of random election timeouts; seed it for a repeatable run.
        """
        self._durable = durable
        self._transport = transport
        self._timing = timing or Timing()
        self._rng = rng or random.Random()
        self._proposals = PendingProposals()
        self._reads = PendingReads()
        # NOTE: told through the supervisor, not `stop`: a failing task closes it too.
        self._supervisor = TaskSupervisor(on_close=self._fail_waiting_callers)
        self._ticker: asyncio.Task | None = None
        self._election_elapsed = 0
        self._heartbeat_elapsed = 0
        self._election_timeout = self._timing.random_election_timeout(self._rng)
        self._seen_role = durable.role
        self._seen_term = durable.current_term
        self._leader_id: int | None = None
        self._vote_request: RequestVoteRequest | None = None
        self._vote_requests_in_flight: set[tuple[int, int]] = set()
        self._append_entries_in_flight: set[tuple[int, int]] = set()
        self._applying = False

    @property
    def node_id(self) -> int:
        return self._durable.node_id

    @property
    def durable(self) -> DurableNodeState:
        return self._durable

    @property
    def leader_id(self) -> int | None:
        return self._leader_id

    @property
    def election_timeout(self) -> int:
        return self._election_timeout

    @property
    def failure(self) -> BaseException | None:
        return self._supervisor.failure

    @property
    def busy(self) -> bool:
        return self._supervisor.busy

    @property
    def pending_proposals(self) -> int:
        return len(self._proposals)

    @property
    def pending_reads(self) -> int:
        return len(self._reads)

    # --- Running and stopping ---------------------------------------------------------

    def start(self) -> None:
        """Tick every `Timing.tick_interval` seconds until `stop`; calling it again does nothing.

        Ticks keep to fixed deadlines, so a slow tick does not delay the ones after
        it; ticks missed while the event loop was busy are skipped, not run back to
        back, so a stall never fires several timeouts at once.

        Raises:
            NodeStoppedError: If the node was stopped.
        """
        self._supervisor.check_open()
        if self._ticker is None:
            self._ticker = self._supervisor.spawn(self._run_clock(), counted=False)

    async def stop(self) -> None:
        """Stop the clock and every task, then wait for the calls still in progress.

        A task cancelled during a write still finishes that write first (DD-22).
        Once this returns, nothing uses the node's store, and every later call
        raises `NodeStoppedError`. Every proposal and read still waiting fails with it
        too (DD-33, DD-34). Must not be awaited from within this node's own RPC handling.
        """
        await self._supervisor.close()

    async def idle(self) -> None:
        """Wait until no send or apply task is left, including any they start meanwhile."""
        await self._supervisor.idle()

    async def tick(self) -> None:
        """Advance this node's clock by one tick, acting on any timeout it reaches.

        Raises:
            RuntimeError: If `start` is ticking the clock already.
            NodeStoppedError: If the node was stopped.
        """
        if self._ticker is not None:
            raise RuntimeError(f"node {self.node_id}'s clock is ticking on its own")
        with self._supervisor.call():
            await self._tick()

    # --- Client commands --------------------------------------------------------------

    async def append_command(self, command: str) -> LogPosition:
        """Append a client command to this Leader's log and send it to every Follower at once.

        Returns once the entry is on this node's disk, not once it is committed.

        Args:
            command: The command, serialized once by the KV Store layer (DD-21).

        Returns:
            The entry's term and 1-based index.

        Raises:
            NotLeaderError: If this node is not Leader (CLIENT-6).
            TypeError: If `command` is not a str.
            ValueError: If `command` is empty, which marks a new Leader's empty entry.
            sqlite3.Error: If writing the entry fails. Nothing was appended.
            NodeStoppedError: If the node was stopped.
        """
        with self._supervisor.call():
            try:
                return await self._durable.append_command(command)
            finally:
                # NOTE: in a finally, so an entry a cancelled caller installed is still sent and,
                # in a one-node cluster, applied.
                self._send_and_apply()

    async def propose(self, command: str) -> object:
        """Append a client command and wait until it is committed and applied; return its result.

        The result is what the state machine returned for the command on this node (CLIENT-3,
        CLIENT-4, CLIENT-5, DD-12). The wait has no time limit of its own: a caller that cannot
        wait as long as it takes wraps the call in `asyncio.timeout`. A caller that gives up, by
        timeout or cancellation, leaves the entry in the log, where it is committed and applied,
        or overwritten, like any other; its result is dropped (DD-33).

        Args:
            command: The command, serialized once by the KV Store layer (DD-21).

        Returns:
            What the state machine returned for `command`.

        Raises:
            NotLeaderError: If this node is not Leader (CLIENT-6). Nothing was appended.
            TypeError: If `command` is not a str.
            ValueError: If `command` is empty, which marks a new Leader's empty entry.
            sqlite3.Error: If writing the entry fails. Nothing was appended.
            LeadershipLostError: If this node stopped being Leader before the entry was
                applied. A later Leader may still commit the entry, so whether the command
                takes effect is unknown.
            NodeStoppedError: If the node was stopped before the call; nothing was appended.
                Also if it stopped while the command was being written or waited: the entry
                may be in the log, and whether the command takes effect is unknown.
        """
        with self._supervisor.call():
            try:
                position = await self._durable.append_command(command)
                # NOTE: a stop during the write closed the node, and no apply will come.
                self._supervisor.check_open()
                # NOTE: registered before the `finally` sends or applies anything, so the entry
                # cannot be applied, and its result dropped, before its caller is waiting.
                future = self._proposals.register(position)
            finally:
                self._send_and_apply()
        try:
            return await future
        finally:
            self._proposals.discard(position)

    def _send_and_apply(self) -> None:
        """Send the entries just appended to every Follower at once and start applying.

        Does nothing unless this node leads.
        """
        if self._durable.role is Role.LEADER:
            self._start_replicating_to_all()
            self._schedule_apply()

    async def read_barrier(self) -> int:
        """Wait until a linearizable read may be served from this node; return its read index.

        The read waits, in this order:

        1. Until this Leader has committed an entry of its own term; it then takes its
           commit index as the read index (CLIENT-10).
        2. Until a majority has answered an AppendEntries built after the read began,
           which shows no other Leader had been elected by then (CLIENT-8, DD-14).
        3. Until the state machine has applied through the read index (CLIENT-9).

        The state machine may then be read at once, and reflects every write committed
        before the read began (DD-12, DD-34). Reads that begin together share the round
        of requests. The wait has no time limit of its own: a caller that cannot wait as
        long as it takes wraps the call in `asyncio.timeout`.

        Returns:
            The read index.

        Raises:
            NotLeaderError: If this node is not Leader, or stopped being Leader while the
                read waited (CLIENT-6). No read was served.
            NodeStoppedError: If the node was stopped, before the call or while the read
                waited.
        """
        # NOTE: not inside `call()`: a read touches no store, and a stop fails it through
        # `on_close`.
        self._supervisor.check_open()
        if self._durable.leadership is None:
            raise NotLeaderError(f"node {self.node_id} is {self._durable.role.value}, not leader")
        future = self._reads.register()
        try:
            self._settle_reads()
            return await future
        finally:
            self._reads.discard(future)

    def _settle_reads(self) -> None:
        """Start, confirm and end the waiting reads as far as the node's state now allows.

        Runs whenever something a read waits for may have changed: a read arrives, an
        answer is handled, a batch is applied. Once this Leader has committed an entry
        of its own term, every read that has not started starts, together, at the
        commit index of that moment. Only requests built after a read began confirm it,
        so a round goes out at once to each Follower with no request in flight; the
        others are asked when they answer, or at the next heartbeat.
        """
        durable = self._durable
        if not self._reads or durable.leadership is None:
            return
        if self._reads.has_unstarted() and durable.has_committed_in_current_term:
            mark, read_index = durable.mark_and_read_index()
            self._reads.start(mark, read_index)
            self._start_replicating_to_all()
        self._reads.settle(durable.read_confirmed, durable.last_applied)

    def _fail_waiting_callers(self) -> None:
        """Fail every proposal and read still waiting with `NodeStoppedError`.

        The node applies and serves nothing more.
        """
        self._proposals.fail_all(
            NodeStoppedError,
            f"node {self.node_id} stopped while its command waited to be applied; "
            "whether the command takes effect is unknown",
        )
        self._reads.fail_all(NodeStoppedError, f"node {self.node_id} stopped while its read waited")

    # --- RPCs from peers --------------------------------------------------------------

    async def handle_request_vote(self, request: RequestVoteRequest) -> RequestVoteResponse:
        """Answer a Candidate's RequestVote; granting it restarts the election timeout (ELECT-13).

        Args:
            request: The Candidate's RequestVote.

        Returns:
            The answer to send back.

        Raises:
            NodeStoppedError: If the node was stopped.
        """
        with self._supervisor.call():
            try:
                response = await self._durable.handle_vote_request(request)
            finally:
                # NOTE: in a finally, so a change installed before its caller was cancelled is
                # noticed now, not at the next tick.
                self._notice_role_or_term_change()
            if response.vote_granted:
                self._restart_election_timeout()
            return response

    async def handle_append_entries(self, request: AppendEntriesRequest) -> AppendEntriesResponse:
        """Answer a Leader's AppendEntries; one from this term's Leader holds off an election.

        An AppendEntries from the current term's Leader, accepted or rejected,
        restarts the election timeout (ELECT-2), makes its sender `leader_id`
        (CLIENT-7), and starts applying what it committed.

        Args:
            request: The Leader's AppendEntries.

        Returns:
            The answer to send back.

        Raises:
            CommittedEntryConflictError: If accepting it would change a committed
                entry (DD-29). The node stops.
            NodeStoppedError: If the node was stopped.
        """
        with self._supervisor.call():
            try:
                response = await self._durable.handle_append_entries(request)
            except CommittedEntryConflictError as error:
                self._supervisor.fail(error)
                raise
            finally:
                # NOTE: in a finally, so a change installed before its caller was cancelled is
                # noticed now, not at the next tick.
                self._notice_role_or_term_change()
            if request.term == self._durable.current_term and self._durable.role is Role.FOLLOWER:
                self._leader_id = request.leader_id
                self._restart_election_timeout()
                self._schedule_apply()
            return response

    # --- The clock --------------------------------------------------------------------

    async def _run_clock(self) -> None:
        loop = asyncio.get_running_loop()
        interval = self._timing.tick_interval
        deadline = loop.time()
        while True:
            deadline += interval
            # NOTE: a deadline already past is dropped, not caught up with, so ticks missed
            # while the loop was busy never run back to back.
            deadline = max(deadline, loop.time())
            await asyncio.sleep(deadline - loop.time())
            await self._tick()

    async def _tick(self) -> None:
        self._notice_role_or_term_change()
        self._heartbeat_elapsed += 1
        heartbeat_due = self._heartbeat_elapsed >= self._timing.heartbeat_ticks
        if heartbeat_due:
            self._heartbeat_elapsed = 0
        if self._durable.role is Role.LEADER:
            self._durable.advance_cluster_time()
            if heartbeat_due:
                self._start_replicating_to_all()
            return
        self._election_elapsed += 1
        if self._election_timed_out():
            await self._start_election()
        elif heartbeat_due:
            self._resend_vote_requests()

    def _election_timed_out(self) -> bool:
        """Whether this node is not Leader and its election timeout has been reached."""
        return (
            self._durable.role is not Role.LEADER
            and self._election_elapsed >= self._election_timeout
        )

    def _restart_election_timeout(self) -> None:
        """Start counting toward a newly drawn election timeout (ELECT-1)."""
        self._election_elapsed = 0
        self._election_timeout = self._timing.random_election_timeout(self._rng)

    def _notice_role_or_term_change(self) -> None:
        """React to a new role or term, once, however many calls observe it.

        A Leader that stops leading fails every proposal and read still waiting (DD-33,
        DD-34). A new term forgets its predecessor's Leader and RequestVote. Any change
        restarts both timers (DD-9). A new Leader records itself as `leader_id`,
        sends its empty entry at once rather than a heartbeat later, and starts
        applying.
        """
        role, term = self._durable.role, self._durable.current_term
        if (role, term) == (self._seen_role, self._seen_term):
            return
        if self._seen_role is Role.LEADER:
            self._proposals.fail_all(
                LeadershipLostError,
                f"node {self.node_id} is no longer Leader of term {self._seen_term}; "
                "whether the command takes effect is unknown",
            )
            self._reads.fail_all(
                NotLeaderError, f"node {self.node_id} is no longer Leader of term {self._seen_term}"
            )
        if term != self._seen_term:
            self._leader_id = None
            self._vote_request = None
        self._seen_role, self._seen_term = role, term
        self._restart_election_timeout()
        self._heartbeat_elapsed = 0
        if role is Role.LEADER:
            self._leader_id = self.node_id
            self._start_replicating_to_all()
            self._schedule_apply()

    # --- Elections --------------------------------------------------------------------

    async def _start_election(self) -> None:
        """Start an election, if still due once the lock is held, and ask every peer (ELECT-6)."""
        request = await self._durable.start_election(still_due=self._election_timed_out)
        if request is None:
            return
        # NOTE: noticing the new term clears `_vote_request`, so it must run before the new
        # request is kept.
        self._notice_role_or_term_change()
        self._vote_request = request
        for peer in sorted(self._durable.peers):
            self._start_vote_request(peer, request)

    def _resend_vote_requests(self) -> None:
        """Resend this Candidate's RequestVote to every peer that has not answered (FAIL-2)."""
        request, candidacy = self._vote_request, self._durable.candidacy
        if request is None or candidacy is None:
            return
        answered = candidacy.votes_granted | candidacy.votes_refused
        for peer in sorted(self._durable.peers - answered):
            self._start_vote_request(peer, request)

    def _start_vote_request(self, peer: int, request: RequestVoteRequest) -> None:
        """Start a task sending `request` to `peer`, unless one of this term is in flight."""
        in_flight = (peer, request.term)
        if in_flight in self._vote_requests_in_flight:
            return
        self._vote_requests_in_flight.add(in_flight)
        self._supervisor.spawn(self._request_vote(peer, request))

    async def _request_vote(self, peer: int, request: RequestVoteRequest) -> None:
        """Send one RequestVote and count its answer."""
        try:
            response = await self._transport.request_vote(peer, request)
            await self._durable.handle_vote_response(peer, request.term, response)
            self._notice_role_or_term_change()
        except PeerUnreachableError:
            pass  # the next heartbeat interval asks again (DD-30)
        finally:
            self._vote_requests_in_flight.discard((peer, request.term))

    # --- Replication ------------------------------------------------------------------

    def _start_replicating_to_all(self) -> None:
        for peer in sorted(self._durable.peers):
            self._start_replicating(peer)

    def _start_replicating(self, peer: int) -> None:
        """Start a task sending AppendEntries to `peer`, unless one of this term is in flight."""
        term = self._durable.current_term
        if (peer, term) in self._append_entries_in_flight:
            return
        self._append_entries_in_flight.add((peer, term))
        self._supervisor.spawn(self._replicate(peer, term))

    async def _replicate(self, peer: int, term: int) -> None:
        """Send AppendEntries to `peer` while this term's Leader has more to send it at once.

        Each round sends the request `next_index` calls for (REPL-2), records the
        answer, and goes again after a rejection that lowered `next_index` (REPL-7),
        after a success that leaves `peer` short of the log, or after an answer
        carrying the request's term while a read still awaits `peer`'s answer to a
        request built after the read began (DD-34). Any other rejection is not
        resent: the same request would be rejected the same way.
        """
        try:
            while self._durable.role is Role.LEADER and self._durable.current_term == term:
                try:
                    request = await self._durable.append_entries_request_for(peer)
                    response = await self._transport.append_entries(peer, request)
                except (NotLeaderError, PeerUnreachableError):
                    return
                backed_off = await self._durable.handle_append_entries_response(
                    peer, request, response
                )
                self._notice_role_or_term_change()
                self._schedule_apply()
                self._settle_reads()
                # NOTE: asked for again only after an answer `handle_append_entries_response`
                # credits, one carrying the request's term; nothing credits any other, so it
                # would be asked for again without end.
                answered_in_term = response.term == request.term
                if not (
                    backed_off
                    or (response.success and self._has_entries_for(peer))
                    or (answered_in_term and self._read_wants_round_from(peer))
                ):
                    return
        finally:
            self._append_entries_in_flight.discard((peer, term))

    def _has_entries_for(self, peer: int) -> bool:
        """Whether the next AppendEntries to `peer` would carry entries."""
        leadership = self._durable.leadership
        return (
            leadership is not None and leadership.next_index(peer) <= self._durable.log.last_index
        )

    def _read_wants_round_from(self, peer: int) -> bool:
        """Whether a read still needs `peer` to answer a request built after the read began."""
        leadership = self._durable.leadership
        if not self._reads or leadership is None:
            return False
        return self._reads.wants_round(
            leadership.answered_request(peer), self._durable.read_confirmed
        )

    # --- Applying ---------------------------------------------------------------------

    def _schedule_apply(self) -> None:
        """Start the apply task unless it runs already or nothing new is committed."""
        if self._applying or self._durable.last_applied >= self._durable.commit_index:
            return
        self._applying = True
        self._supervisor.spawn(self._apply())

    async def _apply(self) -> None:
        """Apply committed entries in batches until the state machine has caught up.

        Each applied entry answers the proposal waiting on it, if any (DD-33), and each
        batch may end a read waiting for the state machine to catch up (DD-34).
        """
        try:
            while self._durable.last_applied < self._durable.commit_index:
                await self._durable.apply_committed(
                    max_entries=_APPLY_BATCH_SIZE, on_applied=self._proposals.answer
                )
                self._settle_reads()
                await asyncio.sleep(0)
        finally:
            self._applying = False
