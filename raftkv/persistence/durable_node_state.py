"""A node's role, term, vote, and log, made durable under one lock (DD-8, DD-19, DD-22)."""

import asyncio
import copy
import functools
from typing import Any, Coroutine, Optional

from raftkv.consensus import (
    Candidacy,
    Cluster,
    Leadership,
    Log,
    LogEntry,
    NodeState,
    RequestVoteRequest,
    RequestVoteResponse,
    Role,
)
from raftkv.persistence.sqlite_store import SqliteStore
from raftkv.tracing import NodeTracer, traced


def _holding_the_lock(method):
    """Run `method` entirely inside one hold of the node's lock (DD-8, DD-19).

    The lock is taken before the method reads any state and released only
    after it returns or raises — so it is held across every `await` inside
    the method, including the wait for a write to commit. A second call on
    the same node waits until the first has decided, persisted, and
    installed, and then decides against the state the first one left.
    """

    @functools.wraps(method)
    async def locked(self, *args: Any) -> Any:
        async with self._lock:
            return await method(self, *args)

    return locked


class DurableNodeState:
    """The single place a node's role, term, vote, and log are changed.

    Owns a pure `NodeState`, the node's `Log`, its `SqliteStore`, the
    `Cluster` it belongs to, and the one per-node `asyncio.Lock` DD-8
    requires. Every method that changes any of that state does all of the
    following while holding the lock, in this order (DD-19, DD-22):

    1. Compute the next state on a copy, using the pure consensus
       classes.
    2. Persist whatever part of it must be durable, and wait for the
       write to commit.
    3. Only then install it as the node's current in-memory state.

    Three properties follow. First, a caller that awaits one of these
    methods and only then responds to an RPC has persisted before
    responding (PERSIST-1 through PERSIST-3), and a Candidate gets the
    RequestVote to send only from `start_election`, after its new term
    and self-vote are on disk (ELECT-5). Second, the in-memory state is
    never ahead of what is on disk: if a write fails, the method raises
    and the node's state is exactly what it was before the call. Third,
    what is on disk is never ahead of the in-memory state either: once a
    write has been handed to the store it runs to completion even if the
    calling task is cancelled meanwhile, the change is installed, and
    only then does the cancellation propagate. A write can finish on the
    store's background thread after the task that issued it has been
    cancelled, so a cancelled caller that simply stopped before step 3
    would leave a change on disk that the node does not know it made.

    Each such method is marked `_holding_the_lock`: the lock is held for
    the whole call, across the `await` on the write, which is where
    `aiosqlite` suspends the calling coroutine while its background
    thread does the disk I/O. Any other coroutine calling one of these
    methods in the meantime waits for the lock rather than computing its
    own next state from the one about to be replaced.

    Alongside the role, the node keeps the in-memory records that only
    one role needs: a `Candidacy` counting votes while it is a
    Candidate, and a `Leadership` tracking every Follower while it is
    Leader. Every change of role or term brings them in line in the same
    step: a Candidate always has a Candidacy for its current term, a
    Leader always has a Leadership for its current term, and a Follower
    has neither. Entering a role, or entering a new term in the same
    role, always starts a fresh record, so nothing counted in an earlier
    term — votes or Follower progress — can carry over into a later
    one. Neither record is persisted; after a restart the node is a
    Follower (STATE-2) and has neither.

    The only way to become Leader is to reach a majority of granted
    votes (ELECT-11): in `handle_vote_response`, or in `start_election`
    when the node's own vote is already a majority.

    Methods marked `traced` are reported, from outside, to the node's
    `NodeTracer`: the decorator records the node's state just before and
    just after the call — inside the lock, so no other call can change it
    in between — and the tracer reports what changed, in etcd's format
    (see `raftkv.tracing`).

    Attributes:
        node_id: This node's permanent positive-integer identity.
        role: The node's current Role. Held in memory only; STATE-2 has
            every node restart as a Follower.
        current_term: The node's current term, as last persisted.
        voted_for: The node ID voted for in `current_term`, as last
            persisted, or None.
        log: The node's log, as last persisted.
        peers: The IDs of every other member of the cluster.
        candidacy: The votes collected in the current term while
            Candidate; None in any other role.
        leadership: The progress of every Follower in the current term
            while Leader; None in any other role.
    """

    @traced(NodeTracer.report_started)
    def __init__(
        self, state: NodeState, log: Log, store: SqliteStore, cluster: Cluster
    ) -> None:
        """Wrap state and a log that already match what `store` holds.

        `load` is the way to build one from a store; this constructor
        trusts that `state` and `log` are exactly what `store` would
        return from `SqliteStore.load`.

        Args:
            state: The node's current role, term, and vote.
            log: The node's current log.
            store: The open store holding the persisted copy of both.
            cluster: The cluster this node is a member of.

        Raises:
            ValueError: If this node is not a member of `cluster`.
        """
        self._state = state
        self._log = log
        self._store = store
        self._cluster = cluster
        self._peers = cluster.peers_of(state.node_id)
        self._candidacy: Optional[Candidacy] = None
        self._leadership: Optional[Leadership] = None
        self._lock = asyncio.Lock()
        self._align_role_records()

    @classmethod
    async def load(
        cls, node_id: int, store: SqliteStore, cluster: Cluster
    ) -> "DurableNodeState":
        """Rebuild a node from its persisted state, in the Follower role.

        Implements start-up for PERSIST-4, PERSIST-5, PERSIST-6, and
        STATE-2: the term, vote, and log are reloaded from `store`, and
        the node begins as a Follower carrying them, whatever role it had
        before it stopped. A store that has never been written to yields
        a brand-new node at term 0 with no vote and an empty log.

        Args:
            node_id: This node's permanent positive-integer identity.
            store: The node's open store.
            cluster: The cluster this node is a member of.

        Returns:
            The node's durable state, ready to accept or issue RPCs.

        Raises:
            ValueError: If `node_id` is not a member of `cluster`.
        """
        persisted = await store.load()
        state = NodeState.reloaded(node_id, persisted.current_term, persisted.voted_for)
        return cls(state, persisted.log, store, cluster)

    @property
    def node_id(self) -> int:
        return self._state.node_id

    @property
    def role(self) -> Role:
        return self._state.role

    @property
    def current_term(self) -> int:
        return self._state.current_term

    @property
    def voted_for(self) -> Optional[int]:
        return self._state.voted_for

    @property
    def log(self) -> Log:
        return self._log

    @property
    def peers(self) -> frozenset[int]:
        return self._peers

    @property
    def candidacy(self) -> Optional[Candidacy]:
        return self._candidacy

    @property
    def leadership(self) -> Optional[Leadership]:
        return self._leadership

    @_holding_the_lock
    @traced(NodeTracer.report_election)
    async def start_election(self) -> RequestVoteRequest:
        """Become Candidate in a new term, persist it, and return the RequestVote to send.

        Applies `NodeState.become_candidate` — STATE-3's Follower-to-
        Candidate or Candidate-to-Candidate edge, with ELECT-3's term
        increment and ELECT-4's vote for self — persists the new term and
        vote, and starts a fresh `Candidacy` for the new term holding
        that one vote. Any Candidacy from an earlier term is discarded
        with it, so no vote granted in an earlier election counts in
        this one.

        Returns the RequestVote to send to every peer (ELECT-6), built
        only after the write has committed: there is no way to obtain the
        request before the term and vote it announces are on disk
        (ELECT-5). It carries the new term, this node's ID, and the index
        and term of the node's last log entry (ELECT-7), all read under
        the same lock hold as the write, so they describe exactly the
        state that was persisted.

        In a single-node cluster the node's own vote is already a
        majority (ELECT-11), so it becomes Leader before returning, and
        there is no peer to send the request to.

        Returns:
            The RequestVote for this election.

        Raises:
            IllegalTransition: If the node is a Leader. Nothing is
                changed or written.
            sqlite3.Error: If the write fails. The node's in-memory
                state is left as it was before the call.
            asyncio.CancelledError: If the calling task was cancelled
                while the write was in flight — raised only after the
                write has committed and the new state is installed. No
                request is returned, so none is sent; the node is a
                Candidate in the new term until its next election
                timeout starts another election.
        """
        next_state = copy.copy(self._state)
        next_state.become_candidate()
        await self._persist_then_install(
            self._store.save_term_and_vote(next_state.current_term, next_state.voted_for),
            next_state=next_state,
        )
        if self._candidacy.has_majority:
            self._become_leader()
        return RequestVoteRequest(
            term=self._state.current_term,
            candidate_id=self._state.node_id,
            last_log_index=self._log.last_index,
            last_log_term=self._log.last_term,
        )

    @_holding_the_lock
    @traced(NodeTracer.report_observed_term)
    async def handle_observed_term(self, term: int) -> bool:
        """Catch up to a higher term seen in an RPC, persisting it before returning.

        Applies `NodeState.handle_observed_term` — STATE-5's term update,
        STATE-6's vote reset, and STATE-4's step-down for a Candidate or
        Leader — and, only when that actually changed something, persists
        the new term and cleared vote. Returns only once that write has
        committed, so a caller that responds to the RPC after awaiting
        this has persisted before responding (PERSIST-1, PERSIST-2). A
        node that steps down discards its Candidacy or Leadership with
        its old role.

        Args:
            term: The term observed in an incoming RPC or RPC response.

        Returns:
            True if `term` was higher than `current_term` and the node
            caught up to it, False if nothing changed and nothing was
            written.

        Raises:
            sqlite3.Error: If the write fails. The node's in-memory
                state is left as it was before the call.
            asyncio.CancelledError: If the calling task was cancelled
                while the write was in flight — raised only after the
                write has committed and the new state is installed.
        """
        next_state = copy.copy(self._state)
        if not next_state.handle_observed_term(term):
            return False
        await self._persist_then_install(
            self._store.save_term_and_vote(next_state.current_term, next_state.voted_for),
            next_state=next_state,
        )
        return True

    @_holding_the_lock
    @traced(NodeTracer.report_vote_request)
    async def handle_vote_request(self, request: RequestVoteRequest) -> RequestVoteResponse:
        """Answer a Candidate's RequestVote, persisting any change before returning the answer.

        The receiving side of RequestVote (DD-18), in one lock hold
        (DD-8, DD-19): decide with `NodeState.handle_vote_request` —
        catch up to a higher term (STATE-4, STATE-5, STATE-6), refuse a
        request from an earlier term or a second Candidate in the same
        term (ELECT-8), refuse a Candidate whose log is behind this
        node's (ELECT-9, ELECT-10), and otherwise record the vote — then
        persist `current_term` and `voted_for` if they changed, and only
        then return the answer. The answer cannot be sent before the vote
        it grants is on disk (PERSIST-1, PERSIST-2), so a node that
        crashes right after answering still remembers its vote when it
        restarts and cannot give it to a different Candidate in the same
        term.

        Holding the lock across the write is what makes ELECT-8 hold
        when two Candidates' requests arrive at once: the second request
        is decided only after the first one's vote has been persisted and
        installed, so it sees that vote and is refused.

        A request that changes nothing writes nothing: a refusal in the
        current term, or a repeat of a request already granted, which is
        granted again (FAIL-1).

        A caller that receives a granted answer must reset its election
        timeout before sending it (ELECT-13).

        Args:
            request: The Candidate's RequestVote.

        Returns:
            The answer to send back to the Candidate.

        Raises:
            sqlite3.Error: If the write fails. The node's in-memory state
                is left as it was before the call, and no answer is
                returned, so none is sent.
            asyncio.CancelledError: If the calling task was cancelled
                while the write was in flight — raised only after the
                write has committed and the new state is installed.
        """
        next_state = copy.copy(self._state)
        response = next_state.handle_vote_request(request, self._log.last_position)
        if (next_state.current_term, next_state.voted_for) != (
            self._state.current_term,
            self._state.voted_for,
        ):
            await self._persist_then_install(
                self._store.save_term_and_vote(next_state.current_term, next_state.voted_for),
                next_state=next_state,
            )
        return response

    @_holding_the_lock
    @traced(NodeTracer.report_vote_response)
    async def handle_vote_response(
        self, voter: int, sent_in_term: int, response: RequestVoteResponse
    ) -> bool:
        """Count a voter's answer toward this node's election, becoming Leader on a majority.

        The Candidate's side of RequestVote (DD-18), in one lock hold:

        1. If the answer carries a term higher than `current_term`, some
           other election has moved past this one. The node catches up
           and steps down (STATE-4, STATE-5, STATE-6), persisting the new
           term and cleared vote, and its Candidacy is discarded.
        2. Otherwise, if the node is a Candidate, the answer is recorded
           on its `Candidacy`, which ignores answers to requests sent in
           any other term and repeated answers from the same voter.
        3. If the votes granted now make up a strict majority of the
           cluster, including this node's own (ELECT-11, ELECT-12), the
           node becomes Leader for this term, with a fresh `Leadership`:
           every Follower starts at `next_index` one past this node's
           last log index (REPL-14) and `match_index` 0 (REPL-15).

        A node that is no longer a Candidate — it already won, or has
        stepped down — ignores the answer, apart from step 1.

        Args:
            voter: The node ID of the peer that answered.
            sent_in_term: The term the answered RequestVote was sent in.
            response: The peer's answer.

        Returns:
            True if this answer completed a majority and the node became
            Leader, False otherwise.

        Raises:
            KeyError: If the request was sent in the current term but
                `voter` is not a member of the cluster.
            sqlite3.Error: If catching up to a higher term fails to
                write. The node's in-memory state is left as it was.
            asyncio.CancelledError: If the calling task was cancelled
                while that write was in flight — raised only after the
                write has committed and the new state is installed.
        """
        next_state = copy.copy(self._state)
        if next_state.handle_observed_term(response.term):
            await self._persist_then_install(
                self._store.save_term_and_vote(next_state.current_term, next_state.voted_for),
                next_state=next_state,
            )
            return False
        if self._candidacy is None:
            return False
        if not self._candidacy.record_vote(voter, sent_in_term, response.vote_granted):
            return False
        if not self._candidacy.has_majority:
            return False
        self._become_leader()
        return True

    @_holding_the_lock
    async def receive_entries(
        self, prev_log_index: int, prev_log_term: int, entries: list[LogEntry]
    ) -> bool:
        """Accept a Leader's entries into this node's log, persisting them before returning.

        The log half of what a Follower does with an AppendEntries RPC.
        First REPL-5's check: if this node's log has no entry at
        `prev_log_index` with term `prev_log_term`, the entries are
        rejected and nothing is written. Otherwise REPL-8's rule computes
        the new log (`Log.after_append_entries`), and only the part of it
        that actually differs from the current log is persisted
        (PERSIST-3): everything from the first differing index onward is
        rewritten, and everything before it is left as it is on disk.

        The persisted log therefore always equals the in-memory one. A
        heartbeat that changes nothing writes nothing, and a stale,
        unconflicted tail of extra entries that REPL-8 leaves in place
        is left in place on disk as well.

        Only the log is checked here. The RPC's term is not compared
        against `current_term`.

        Args:
            prev_log_index: The 1-based index of the entry immediately
                preceding `entries`, as carried by the RPC.
            prev_log_term: The term the entry at `prev_log_index` is
                expected to have.
            entries: The Leader's entries, in order, starting immediately
                after `prev_log_index`.

        Returns:
            True if the entries were accepted and the resulting log is
            persisted, False if REPL-5's check rejected them.

        Raises:
            sqlite3.Error: If the write fails. The node's in-memory log
                is left as it was before the call.
            asyncio.CancelledError: If the calling task was cancelled
                while the write was in flight — raised only after the
                write has committed and the new log is installed.
        """
        if not self._log.matches(prev_log_index, prev_log_term):
            return False
        next_log = self._log.after_append_entries(prev_log_index, entries)
        changed_from = self._log.first_differing_index(next_log)
        if changed_from is None:
            self._log = next_log
            return True
        await self._persist_then_install(
            self._store.save_log_from(changed_from, next_log[changed_from - 1 :]),
            next_log=next_log,
        )
        return True

    async def _persist_then_install(
        self,
        write: Coroutine[Any, Any, None],
        next_state: Optional[NodeState] = None,
        next_log: Optional[Log] = None,
    ) -> None:
        """Run `write` to completion, then install what it persisted.

        The write runs as its own task, which cancelling the caller does
        not cancel: `asyncio.wait` stops waiting when the caller is
        cancelled but leaves the awaited task running. The caller keeps
        waiting until the write has finished either way. If the write
        committed, `next_state` and/or `next_log` are installed, and the
        per-role records are brought in line with the installed state; if
        it failed, nothing is installed. A cancellation received
        meanwhile is raised only after that.

        Args:
            write: The store write that makes the change durable.
            next_state: The NodeState to install once `write` commits.
            next_log: The Log to install once `write` commits.

        Raises:
            asyncio.CancelledError: If the caller was cancelled while the
                write was in flight.
            Exception: Whatever `write` raised, if it failed and the
                caller was not cancelled.
        """
        pending = asyncio.ensure_future(write)
        cancelled = False
        while not pending.done():
            try:
                await asyncio.wait({pending})
            except asyncio.CancelledError:
                cancelled = True
        failure = pending.exception()
        if failure is None:
            if next_state is not None:
                self._state = next_state
                self._align_role_records()
            if next_log is not None:
                self._log = next_log
        if cancelled:
            raise asyncio.CancelledError() from failure
        if failure is not None:
            raise failure

    def _become_leader(self) -> None:
        """Become Leader of the current term, with a fresh Leadership.

        Applies `NodeState.become_leader` — STATE-3's Candidate-to-Leader
        edge — and replaces the Candidacy with a new `Leadership`.
        Called only with the lock held, once a Candidacy has reached a
        majority (ELECT-11). Nothing is written: role is not persisted.
        """
        self._state.become_leader()
        self._align_role_records()

    def _align_role_records(self) -> None:
        """Bring the Candidacy and Leadership in line with the current role and term.

        A Candidate gets a Candidacy for its current term, holding its
        own vote; a Leader gets a Leadership for its current term, with
        every Follower reset (REPL-14, REPL-15); a Follower gets neither.
        A record that already belongs to the current role and term is
        kept. Any other is replaced by a fresh one, which is what
        discards votes and Follower progress from an earlier term the
        moment the node moves to a new one.
        """
        role, term = self._state.role, self._state.current_term
        if role is not Role.CANDIDATE:
            self._candidacy = None
        elif self._candidacy is None or self._candidacy.term != term:
            self._candidacy = Candidacy(term, self._state.node_id, self._cluster)
        if role is not Role.LEADER:
            self._leadership = None
        elif self._leadership is None or self._leadership.term != term:
            self._leadership = Leadership(term, self._peers, self._log.last_index)
