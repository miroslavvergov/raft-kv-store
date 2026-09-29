"""A node's Raft state, persisted before installed, under one lock (DD-8, DD-19, DD-22)."""

import asyncio
import copy
import functools
import inspect
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, Protocol

from raftkv.consensus import (
    AppendEntriesRequest,
    AppendEntriesResponse,
    Candidacy,
    Cluster,
    CommittedEntryConflictError,
    Leadership,
    Log,
    LogEntry,
    LogPosition,
    NodeState,
    NotLeaderError,
    RequestVoteRequest,
    RequestVoteResponse,
    Role,
)
from raftkv.storage import SqliteStore
from raftkv.tracing import NodeTracer, traced


class ApplyCallback(Protocol):
    """The KV Store layer's callback for one committed command, in log order (DD-12, DD-28)."""

    def __call__(self, index: int, cluster_time: int, command: str) -> object: ...


class AppliedCallback(Protocol):
    """The caller's callback for one applied entry: its index, term, and the state machine's result.

    The result is what `ApplyCallback` returned for the entry's command, and None for an empty
    entry, which reaches no command (DD-26, DD-28, DD-33).
    """

    def __call__(self, index: int, term: int, result: object) -> None: ...


def _holding_the_lock[T](
    method: Callable[..., Awaitable[T]],
) -> Callable[..., Coroutine[Any, Any, T]]:
    """Run the decorated method entirely under the node's lock (DD-8, DD-19).

    The lock is held across every `await` in the method, including the commit
    of its write, so a second call decides only against the state the first
    one installed.
    """

    @functools.wraps(method)
    async def locked(self: "DurableNodeState", *args: Any, **kwargs: Any) -> T:
        async with self._lock:
            return await method(self, *args, **kwargs)

    return locked


class DurableNodeState:
    """The single place a node's role, term, vote, and log change.

    Owns the node's `NodeState`, `Log`, `SqliteStore`, `Cluster`, and one
    `asyncio.Lock` (DD-8). Each changing method holds the lock for the whole
    call, across the `await` on its write, and decides the next state on a
    copy, persists it and waits for the commit, and only then installs it
    (DD-19, DD-22). Hence:

    - A caller that awaits a method before answering an RPC has persisted
      before responding (PERSIST-1, PERSIST-2, PERSIST-3), and a Candidate
      gets its RequestVote only once its term and self-vote are on disk
      (ELECT-5).
    - Memory is never ahead of disk: a failed write raises and changes
      nothing.
    - Disk is never ahead of memory: a write handed to the store completes and
      is installed even if the caller is cancelled, which is raised afterwards.

    A `Candidacy` exists exactly while Candidate and a `Leadership` exactly
    while Leader, each for the current term; a new role or term starts fresh
    ones, so no votes or Follower progress carry across terms. Neither is
    persisted. The only way to become Leader is a majority of granted votes
    (ELECT-11), in `handle_vote_response` or, when the node's own vote is a
    majority, in `start_election`; a new Leader then appends an empty entry in
    its term, so the entries before it can commit (APPLY-3).

    A Leader appends client commands (`append_command`), builds each Follower's
    AppendEntries (`append_entries_request_for`), and records the answers
    (`handle_append_entries_response`), committing what a majority holds from
    its own term. A Follower answers AppendEntries (`handle_append_entries`).
    Every node hands its committed commands, in order, to the `apply` callback
    (`apply_committed`), which tells its caller each entry's result through `on_applied`.

    A Leader stamps every entry it appends with its leadership's cluster time,
    which `advance_cluster_time` moves one tick at a time (DD-32); a Follower
    stores the times it is sent. Every node hands each command's index and
    cluster time to `apply`.

    Attributes:
        node_id: This node's permanent positive-integer identity.
        role: The current Role; in memory only, since every node restarts as a
            Follower (STATE-2).
        current_term: The current term, as last persisted.
        voted_for: The node voted for in `current_term`, as last persisted, or
            None.
        log: The log, as last persisted.
        commit_index: The highest log index known to be committed; 0 after a
            restart, since it is never persisted (DD-28): a Follower relearns it
            from the Leader's next AppendEntries (REPL-13), and a new Leader by
            committing its empty entry.
        last_applied: The highest log index applied; 0 after a restart, since the
            state machine is rebuilt by applying the log again. An empty entry
            advances it without reaching the state machine (DD-26).
        has_committed_in_current_term: Whether an entry of the current term is
            committed, which a Leader must have before answering a read (CLIENT-10).
        peers: The IDs of every other cluster member.
        candidacy: The current term's vote tally while Candidate; else None.
        leadership: The current term's Follower progress and cluster clock while
            Leader; else None.
    """

    @traced(NodeTracer.report_started)
    def __init__(
        self,
        state: NodeState,
        log: Log,
        store: SqliteStore,
        cluster: Cluster,
        apply: ApplyCallback | None = None,
    ) -> None:
        """Wrap a state and log that already equal what `store` holds.

        Not checked; `load` builds one from a store and guarantees it.

        Args:
            state: The node's role, term, and vote.
            log: The node's log.
            store: The open store holding the persisted copy of both.
            cluster: The cluster this node is a member of.
            apply: The KV Store layer's callback for one committed command, called
                as `apply(index, cluster_time, command)` (DD-12, DD-28); without it
                `apply_committed` refuses to pass a command on. It runs under the
                node's lock, so it must not call back into this node, and must
                apply the command or raise, never partly apply it and raise.

        Raises:
            ValueError: If this node is not a member of `cluster`.
            TypeError: If `apply` is a coroutine function: its coroutine would never
                be awaited, and `last_applied` would pass entries the state machine
                never saw (DD-28).
        """
        if inspect.iscoroutinefunction(apply):
            raise TypeError("the apply callback must be synchronous, not a coroutine function")
        self._state = state
        self._log = log
        self._store = store
        self._cluster = cluster
        self._peers = cluster.peers_of(state.node_id)
        # NOTE: commitment is not persisted, so a restarted node starts at 0 and relearns it
        # from the Leader; nothing is lost, since an entry's durability is the log's job.
        self._commit_index = 0
        # NOTE: like the commit index, how far the log has been applied is not persisted: a
        # restarted node rebuilds its state machine by applying the log again from the start.
        self._last_applied = 0
        self._apply = apply
        self._candidacy: Candidacy | None = None
        self._leadership: Leadership | None = None
        self._lock = asyncio.Lock()
        self._align_role_records()

    @classmethod
    async def load(
        cls,
        node_id: int,
        store: SqliteStore,
        cluster: Cluster,
        apply: ApplyCallback | None = None,
    ) -> "DurableNodeState":
        """Rebuild a node from `store`, as a Follower (STATE-2).

        Reloads term, vote, and log (PERSIST-4, PERSIST-5, PERSIST-6); the node
        starts as a Follower whatever role it held before stopping. A store never
        written to yields a new node at term 0 with no vote and an empty log.

        Args:
            node_id: This node's permanent positive-integer identity.
            store: The node's open store.
            cluster: The cluster this node is a member of.
            apply: The KV Store layer's callback for one committed command, called
                as `apply(index, cluster_time, command)` (DD-12, DD-28). It must
                belong to a state machine holding nothing yet: nothing is applied
                here, so the caller applies the whole log from index 1.

        Returns:
            The node, ready to accept or issue RPCs.

        Raises:
            ValueError: If `node_id` is not a member of `cluster`.
        """
        persisted = await store.load()
        state = NodeState.reloaded(node_id, persisted.current_term, persisted.voted_for)
        return cls(state, persisted.log, store, cluster, apply)

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
    def voted_for(self) -> int | None:
        return self._state.voted_for

    @property
    def log(self) -> Log:
        return self._log

    @property
    def commit_index(self) -> int:
        return self._commit_index

    @property
    def last_applied(self) -> int:
        return self._last_applied

    @property
    def has_committed_in_current_term(self) -> bool:
        """Whether the entry at `commit_index` is from the current term (CLIENT-10, DD-26).

        Terms never fall along a log, so if any current-term entry is committed then
        the entry at `commit_index` is one too, and checking that one index is enough.
        Only meaningful on a Leader, which answers no read until this holds.
        """
        return self._commit_index > 0 and self._log.term_at(self._commit_index) == self.current_term

    @property
    def peers(self) -> frozenset[int]:
        return self._peers

    @property
    def candidacy(self) -> Candidacy | None:
        return self._candidacy

    @property
    def leadership(self) -> Leadership | None:
        return self._leadership

    @_holding_the_lock
    @traced(NodeTracer.report_start_election)
    async def start_election(
        self, still_due: Callable[[], bool] | None = None
    ) -> RequestVoteRequest | None:
        """Become Candidate in a new term, persist it, and return the RequestVote to send.

        Applies `NodeState.become_candidate` (STATE-3, ELECT-3, ELECT-4), persists
        the new term and self-vote, and starts a fresh `Candidacy` holding that
        vote, discarding any earlier one. The request is built only after the
        commit (ELECT-5), under the same lock hold, from the new term, this node's
        ID, and its last log index and term (ELECT-7). In a single-node cluster the
        own vote is a majority (ELECT-11), so the node becomes Leader in the same
        install, even if the caller is cancelled, then appends its empty entry,
        which commits at once; the request has no one to go to.

        Args:
            still_due: Asked once the lock is held, before anything changes; if it
                returns False, no election starts. It lets a caller whose timeout
                fired while another call held the lock skip an election that call
                made unnecessary, such as hearing from the Leader (ELECT-2).

        Returns:
            The RequestVote to send to every peer (ELECT-6), or None if `still_due`
            returned False.

        Raises:
            IllegalTransitionError: If the node is Leader. Nothing changes.
            sqlite3.Error: If the term-and-vote write fails, nothing changes. If a
                single-node cluster's empty-entry write fails, the node is Leader and
                keeps that log for the term, with no entry of its own term.
            asyncio.CancelledError: If the caller was cancelled during a write, raised
                once that write is persisted and installed. No request is returned.
                Cancelled during the term-and-vote write, a node in a multi-node cluster
                stays a Candidate until its next election timeout, and a single-node
                Leader is left without its empty entry; cancelled during the empty
                entry's write, the entry is installed but not committed.
        """
        if still_due is not None and not still_due():
            return None
        next_state = copy.copy(self._state)
        next_state.become_candidate()
        # NOTE: in a one-node cluster the self-vote is already a majority and no answer will
        # come. Winning on the copy puts the win in the same install, even if cancelled.
        if self._cluster.is_majority({next_state.node_id}):
            next_state.become_leader()
        await self._persist_then_install_state(next_state)
        if self._state.role is Role.LEADER:
            await self._append_empty_entry()
        # NOTE: built from the installed state, so a request exists only for a term on disk
        # (ELECT-5).
        return RequestVoteRequest(
            term=self._state.current_term,
            candidate_id=self._state.node_id,
            last_log_index=self._log.last_index,
            last_log_term=self._log.last_term,
        )

    @_holding_the_lock
    @traced(NodeTracer.report_observed_term)
    async def handle_observed_term(self, term: int) -> bool:
        """Catch up to a higher `term` seen in an RPC, persisting it before returning.

        Applies `NodeState.handle_observed_term` (STATE-4, STATE-5, STATE-6) and,
        only if it fired, persists the new term and cleared vote, so a reply sent
        after awaiting this has persisted first (PERSIST-1, PERSIST-2). A node that
        steps down drops its Candidacy or Leadership.

        Args:
            term: The term seen in an incoming RPC or RPC response.

        Returns:
            True if `term` was higher and the node caught up; False if nothing
            changed and nothing was written.

        Raises:
            sqlite3.Error: If the write fails. Nothing changes.
            asyncio.CancelledError: If the caller was cancelled during the write;
                raised after the change is persisted and installed.
        """
        next_state = copy.copy(self._state)
        if not next_state.handle_observed_term(term):
            return False
        await self._persist_then_install_state(next_state)
        return True

    @_holding_the_lock
    @traced(NodeTracer.report_vote_request)
    async def handle_vote_request(self, request: RequestVoteRequest) -> RequestVoteResponse:
        """Answer a Candidate's RequestVote, persisting any change before returning.

        The voter's side of RequestVote (DD-18). Decides with
        `NodeState.handle_vote_request` (STATE-4, STATE-5, STATE-6, ELECT-8,
        ELECT-9, ELECT-10), then persists `current_term` and `voted_for` if either
        changed. No answer exists before the vote it grants is on disk, so a crash
        cannot free that vote for another Candidate in the same term (PERSIST-1,
        PERSIST-2). The lock is held across the write (DD-8, DD-19), so a second
        Candidate's concurrent request is decided against the first one's vote and
        refused (ELECT-8). A refusal in the current term or a repeated grant
        (FAIL-1) writes nothing. The caller resets its election timeout before
        sending a grant (ELECT-13).

        Args:
            request: The Candidate's RequestVote.

        Returns:
            The answer to send back.

        Raises:
            sqlite3.Error: If the write fails. Nothing changes and no answer is
                returned.
            asyncio.CancelledError: If the caller was cancelled during the write;
                raised after the change is persisted and installed.
        """
        next_state = copy.copy(self._state)
        response = next_state.handle_vote_request(request, self._log.last_position)
        # NOTE: a refusal can still raise the term, which must persist, so the write follows a
        # changed term or vote, not the grant.
        if (next_state.current_term, next_state.voted_for) != (
            self._state.current_term,
            self._state.voted_for,
        ):
            await self._persist_then_install_state(next_state)
        return response

    @_holding_the_lock
    @traced(NodeTracer.report_vote_response)
    async def handle_vote_response(
        self, voter: int, sent_in_term: int, response: RequestVoteResponse
    ) -> bool:
        """Count a voter's answer toward this node's election; become Leader on a majority.

        The Candidate's side of RequestVote (DD-18):

        1. An answer with a higher term: catch up and step down (STATE-4, STATE-5,
           STATE-6), persisting the new term and cleared vote; nothing else.
        2. Otherwise only a Candidate records it, on its `Candidacy`, which ignores
           answers to requests from other terms and repeated answers.
        3. Once granted votes, its own included, form a strict majority (ELECT-11,
           ELECT-12), the node becomes Leader with a fresh `Leadership` (REPL-14,
           REPL-15) and appends its empty entry, persisted before returning.

        Args:
            voter: The ID of the peer that answered.
            sent_in_term: The term the answered RequestVote was sent in.
            response: The peer's answer.

        Returns:
            True if this answer made the node Leader, False otherwise.

        Raises:
            KeyError: If the node is a Candidate, `sent_in_term` is the current
                term, and `voter` is not a member.
            sqlite3.Error: If persisting a higher term fails, nothing changes. If
                persisting the empty entry fails, the node is Leader without it.
            asyncio.CancelledError: If the caller was cancelled during a write;
                raised after the change is persisted and installed.
        """
        next_state = copy.copy(self._state)
        # NOTE: checked before the Candidacy, so a late answer with a higher term also updates
        # a Follower or steps down a Leader (STATE-4, STATE-5).
        if next_state.handle_observed_term(response.term):
            await self._persist_then_install_state(next_state)
            return False
        if self._candidacy is None:
            return False
        if not self._candidacy.record_vote(voter, sent_in_term, response.vote_granted):
            return False
        if not self._candidacy.has_majority:
            return False
        await self._become_leader()
        return True

    @_holding_the_lock
    @traced(NodeTracer.report_append_entries)
    async def handle_append_entries(self, request: AppendEntriesRequest) -> AppendEntriesResponse:
        """Answer a Leader's AppendEntries, persisting every change before returning.

        The Follower's side of AppendEntries, in one lock hold (DD-8):

        1. Decide on the term (`NodeState.recognize_leader`: STATE-4, STATE-5,
           STATE-6, STATE-7). A Leader of an older term is refused and nothing
           changes.
        2. Reject unless the log agrees at `prev_log_index` (REPL-5); a rejection
           still keeps any term this RPC brought.
        3. Otherwise take the entries in (REPL-8) and advance `commit_index`
           (REPL-13).

        Whatever changed among term, vote, and log is persisted before the answer
        exists (PERSIST-1, PERSIST-2, PERSIST-3), both together in one transaction.
        The term check and the log change share one lock hold, so no newer term
        can arrive between them and have a deposed Leader's entries written under
        it. An RPC that would change an entry at or below `commit_index` is
        refused before anything is written (DD-29).

        Args:
            request: The Leader's AppendEntries.

        Returns:
            The answer to send back, carrying `current_term` after step 1 so an
            outdated Leader learns the newer term and steps down.

        Raises:
            CommittedEntryConflictError: If accepting the entries would change one
                this node has committed. Nothing changes and no answer is returned.
            sqlite3.Error: If the write fails. Nothing changes and no answer is
                returned.
            asyncio.CancelledError: If the caller was cancelled during the write;
                raised after the change is persisted and installed, with
                `commit_index` not advanced.
        """
        next_state = copy.copy(self._state)
        if not next_state.recognize_leader(request.term):
            # NOTE: recognize_leader changes nothing when it refuses, so there is nothing to
            # install or persist and the node's own term goes back untouched.
            return AppendEntriesResponse(term=self.current_term, success=False)
        accepted = self._log.matches(request.prev_log_index, request.prev_log_term)
        next_log = (
            self._log.after_append_entries(request.prev_log_index, list(request.entries))
            if accepted
            else None
        )
        # NOTE: accepting entries never changes one before prev_log_index + 1, so the comparison
        # starts there: a heartbeat costs nothing however long the log is.
        changed_from = (
            None
            if next_log is None
            else self._log.first_differing_index(next_log, start=request.prev_log_index + 1)
        )
        # NOTE: the election and commit rules already rule this out; checking it here turns a
        # broken rule into a named failure before anything is written, not a lost entry later.
        if changed_from is not None and changed_from <= self._commit_index:
            raise CommittedEntryConflictError(
                f"node {self.node_id} committed through {self._commit_index}, but an "
                f"AppendEntries from {request.leader_id} would change entry {changed_from}"
            )
        await self._persist_then_install_append_entries(next_state, next_log, changed_from)
        if accepted:
            # NOTE: after the install, so a failed write commits nothing; a cancelled caller
            # installs the entries but commits nothing until the Leader's retry. In memory
            # only: commitment is recomputed from the Leader after a restart.
            self._commit_index = request.commit_index_after(self._commit_index)
        return AppendEntriesResponse(term=self.current_term, success=accepted)

    @_holding_the_lock
    @traced(NodeTracer.report_append_command)
    async def append_command(self, command: str) -> LogPosition:
        """Append a client command to this Leader's log, persisted before returning (REPL-1).

        The entry takes the current term. Returns once it is on disk, not once it
        is committed; waiting for that is the caller's, and the position returned
        tells it apart from any other entry a later Leader puts at the same index.
        In a single-node cluster the Leader alone is a majority, so the entry
        commits at once (APPLY-1).

        Args:
            command: The command, serialized once by the KV Store layer (DD-21).

        Returns:
            The entry's term and 1-based index.

        Raises:
            NotLeaderError: If the node is not Leader (CLIENT-6). Nothing changes.
            TypeError: If `command` is not a str.
            ValueError: If `command` is empty, which marks a new Leader's empty entry.
            sqlite3.Error: If the write fails. Nothing changes.
            asyncio.CancelledError: If the caller was cancelled during the write;
                raised after the entry is persisted and installed and the commit
                index advanced.
        """
        # NOTE: checked before the entry is built: a node that never led may be at term 0,
        # which no entry can carry.
        if self._leadership is None:
            raise NotLeaderError(f"node {self.node_id} is {self.role.value}, not leader")
        entry = LogEntry(
            term=self.current_term, command=command, cluster_time=self._leadership.cluster_time
        )
        if entry.is_empty:
            raise ValueError("an empty command is reserved for a new Leader's empty entry")
        return await self._append_to_own_log(entry)

    @_holding_the_lock
    @traced(NodeTracer.report_append_entries_request)
    async def append_entries_request_for(self, follower: int) -> AppendEntriesRequest:
        """Return the AppendEntries this Leader sends a Follower now (REPL-2).

        Built by `Leadership.append_entries_request_for` from the Follower's
        `next_index`, this node's log, and its commit index (REPL-4). Changes
        nothing; taken under the lock, so it never reflects a change still being
        written.

        Args:
            follower: A peer of this node.

        Raises:
            NotLeaderError: If the node is not Leader.
            KeyError: If `follower` is not a peer.
        """
        if self._leadership is None:
            raise NotLeaderError(f"node {self.node_id} is {self.role.value}, not leader")
        return self._leadership.append_entries_request_for(
            follower, self._log, self.node_id, self._commit_index
        )

    @_holding_the_lock
    @traced(NodeTracer.report_append_entries_response)
    async def handle_append_entries_response(
        self, follower: int, request: AppendEntriesRequest, response: AppendEntriesResponse
    ) -> bool:
        """Record a Follower's answer to this node's AppendEntries; advance the commit index.

        The Leader's side of AppendEntries:

        1. An answer with a higher term: catch up and step down (STATE-4, STATE-5,
           STATE-6), persisting the new term and cleared vote; nothing else.
        2. Otherwise only a Leader records it, and only an answer to an RPC sent in
           this term counts (REPL-16, DD-25); `request` is the RPC answered, so its
           term is the term it was sent in.
        3. A success raises the Follower's `match_index` to the last entry `request`
           carried (REPL-16, REPL-17), then advances `commit_index` (APPLY-1,
           APPLY-2, APPLY-3). A rejection of the probe now outstanding lowers its
           `next_index` (REPL-6, DD-27); a duplicate or a late rejection of an
           earlier probe changes nothing (FAIL-1).

        Args:
            follower: The Follower that answered.
            request: The AppendEntries it answered.
            response: Its answer.

        Returns:
            True if a rejection lowered the Follower's `next_index`, so a resend
            reaches further back (REPL-7); False otherwise, including at the floor,
            `match_index + 1`, where a resend would repeat the rejected request.

        Raises:
            KeyError: If the node is Leader, `request` was sent in its term, and
                `follower` is not one of its Followers.
            sqlite3.Error: If persisting a higher term fails. Nothing changes.
            asyncio.CancelledError: If the caller was cancelled during that write;
                raised after the change is persisted and installed.
        """
        next_state = copy.copy(self._state)
        if next_state.handle_observed_term(response.term):
            await self._persist_then_install_state(next_state)
            return False
        if self._leadership is None:
            return False
        if not response.success:
            return self._leadership.record_rejection(follower, request.term, request.prev_log_index)
        counted = self._leadership.record_success(
            follower, request.term, request.prev_log_index, len(request.entries)
        )
        if counted:
            self._advance_commit_index()
        return False

    def advance_cluster_time(self) -> None:
        """Count one tick of cluster time; does nothing unless Leader (DD-32).

        In memory only: the time reaches disk inside the entries this Leader
        stamps. Synchronous, so it takes no lock.
        """
        if self._leadership is not None:
            self._leadership.advance_cluster_time()

    @_holding_the_lock
    @traced(NodeTracer.report_apply_committed)
    async def apply_committed(
        self, max_entries: int | None = None, *, on_applied: AppliedCallback | None = None
    ) -> int:
        """Apply committed entries not yet applied, in order; return how many carried a command.

        Walks from `last_applied + 1` to `commit_index`, or `max_entries` entries if
        fewer, and applies each entry exactly once, in log order (APPLY-4), never
        reaching past what is committed (APPLY-5, DD-28). A new Leader's empty entry
        carries no command, so it advances `last_applied` without reaching the state
        machine (DD-26). Applying changes nothing on disk: the state machine is
        rebuilt from the log after a restart.

        Args:
            max_entries: The most entries to apply in this call, empty ones included;
                all that are committed if None. Applying runs without yielding to
                the event loop, so a caller with a long backlog applies it in batches.
            on_applied: Told of each entry right after it is applied, empty ones
                included, with its index, its term, and what the state machine
                returned for it (DD-28, DD-33). Called under the node's lock, so it
                must not wait or call back into this node. It is how a Leader gets a
                proposed command's result.

        Returns:
            How many commands were handed to the state machine; empty entries are
            not counted.

        Raises:
            RuntimeError: If an entry carries a command and there is no `apply` callback.
                No command is applied; `last_applied` stops at the entry before it.
            Exception: Whatever the `apply` callback raised. `last_applied` stops at the
                entry before the one that failed, so no entry is skipped. If
                `on_applied` raised instead, `last_applied` is already past that entry.
        """
        applied = 0
        last = self._commit_index
        if max_entries is not None:
            last = min(last, self._last_applied + max_entries)
        while self._last_applied < last:
            index = self._last_applied + 1
            entry = self._log.entry_at(index)
            result = None
            if not entry.is_empty:
                if self._apply is None:
                    raise RuntimeError(
                        f"node {self.node_id} has no state machine to apply index {index} to"
                    )
                result = self._apply(index, entry.cluster_time, entry.command)
                applied += 1
            self._last_applied = index
            if on_applied is not None:
                on_applied(index, entry.term, result)
        return applied

    async def _persist_then_install_append_entries(
        self, next_state: NodeState, next_log: Log | None, changed_from: int | None
    ) -> None:
        """Persist whatever `handle_append_entries` changed, then install it.

        Term and vote change when the RPC carried a higher term (STATE-5, STATE-6);
        the log changes when accepted entries differ from what is held. Both
        changing is one transaction (DD-7). Only the suffix from the first
        differing index is written (PERSIST-3), so disk always equals memory: a
        heartbeat that changes nothing writes nothing, and a stale, unconflicted
        tail stays on disk too. A Candidate stepping down for a Leader of its own
        term changes neither, and is installed without a write, role not being
        persisted (STATE-2).

        Args:
            next_state: The state decided on a copy, to install.
            next_log: The log after accepting the entries, or None if rejected.
            changed_from: The first index where `next_log` differs from the current
                log, or None if it does not.
        """
        state_changed = (next_state.current_term, next_state.voted_for) != (
            self._state.current_term,
            self._state.voted_for,
        )
        # NOTE: next_log's suffix is written, not the RPC's entries: they differ when the
        # conflict starts past the first incoming entry, and the store deletes from there on.
        entries = [] if changed_from is None else next_log.entries_from(changed_from)
        if state_changed and changed_from is not None:
            write = self._store.save_term_vote_and_log_from(
                next_state.current_term, next_state.voted_for, changed_from, entries
            )
        elif state_changed:
            write = self._store.save_term_and_vote(next_state.current_term, next_state.voted_for)
        elif changed_from is not None:
            write = self._store.replace_log_from(changed_from, entries)
        else:
            self._install(next_state, next_log)
            return
        await self._persist_then_install(write, next_state=next_state, next_log=next_log)

    async def _persist_then_install_state(self, next_state: NodeState) -> None:
        """Persist `next_state`'s term and vote, then install it via `_persist_then_install`."""
        await self._persist_then_install(
            self._store.save_term_and_vote(next_state.current_term, next_state.voted_for),
            next_state=next_state,
        )

    async def _persist_then_install(
        self,
        write: Coroutine[Any, Any, None],
        next_state: NodeState,
        next_log: Log | None = None,
    ) -> None:
        """Run `write` to completion, then install what it persisted.

        `write` runs as its own task, so cancelling the caller does not cancel it,
        and the caller waits for it either way. If it commits, `next_state` (with
        the role records realigned) and `next_log` are installed; if it fails,
        nothing is. A cancellation is raised only after that, because aiosqlite can
        finish the write on its thread after the caller is cancelled, and stopping
        early would leave a change on disk the node does not know it made. If the
        write task itself is cancelled, as at shutdown, whether it committed is
        unknown: the store is read back and the change installed only if it holds it.

        Args:
            write: The store write that makes the change durable.
            next_state: The NodeState to install once `write` commits.
            next_log: The Log to install once `write` commits; None leaves the log.

        Raises:
            asyncio.CancelledError: If the caller was cancelled during the write.
            Exception: Whatever `write` raised, if it failed and the caller was not
                cancelled.
        """
        pending = asyncio.ensure_future(write)
        cancelled = False
        # NOTE: `await pending` would cancel the write along with the caller; `asyncio.wait`
        # does not, and the loop keeps waiting through each cancellation until it settles.
        while not pending.done():
            try:
                await asyncio.wait({pending})
            except asyncio.CancelledError:
                cancelled = True
        # NOTE: a write task cancelled from outside may still have committed, so the store
        # decides what to install.
        if pending.cancelled():
            cancelled = True
            committed = await self._store_holds(next_state, next_log)
            failure = None if committed else asyncio.CancelledError("write cancelled uncommitted")
        else:
            failure = pending.exception()
        if failure is None:
            self._install(next_state, next_log)
        # NOTE: cancellation outranks a write failure, which is chained as the cause.
        if cancelled:
            raise asyncio.CancelledError() from failure
        if failure is not None:
            raise failure

    def _install(self, next_state: NodeState, next_log: Log | None) -> None:
        """Install a persisted state, and log if given, realigning the role records."""
        # NOTE: the log is installed first: realigning builds a Leadership from the log's
        # last index (REPL-14), which must be the log this write persisted.
        if next_log is not None:
            self._log = next_log
        self._state = next_state
        self._align_role_records()

    async def _store_holds(self, next_state: NodeState, next_log: Log | None) -> bool:
        """Return whether the store holds `next_state`'s term and vote, and `next_log` if given."""
        persisted = await self._store.load()
        state_held = (persisted.current_term, persisted.voted_for) == (
            next_state.current_term,
            next_state.voted_for,
        )
        return state_held and (next_log is None or persisted.log == next_log)

    async def _become_leader(self) -> None:
        """Become Leader of the current term with a fresh Leadership, then append its empty entry.

        Applies `NodeState.become_leader` (STATE-3's Candidate-to-Leader edge)
        once the Candidacy has a majority (ELECT-11). Role is not persisted; the
        empty entry is (DD-26).
        """
        # NOTE: role is not persisted, so this change needs no write and is made in place.
        self._state.become_leader()
        self._align_role_records()
        await self._append_empty_entry()

    async def _append_empty_entry(self) -> None:
        """Append this Leader's empty entry, so earlier entries can commit (APPLY-3, DD-26)."""
        # NOTE: appended after the Leadership is built, so each Follower's next_index is this
        # entry's index and the first AppendEntries carries it.
        await self._append_to_own_log(
            LogEntry.empty(self._state.current_term, self._leadership.cluster_time)
        )

    async def _append_to_own_log(self, entry: LogEntry) -> LogPosition:
        """Persist `entry` at the end of this Leader's log, install it, advance commitment.

        Returns:
            The entry's term and 1-based index.
        """
        next_log = self._log.after_append_entries(self._log.last_index, [entry])
        try:
            await self._persist_then_install(
                self._store.replace_log_from(next_log.last_index, [entry]),
                next_state=self._state,
                next_log=next_log,
            )
        finally:
            # NOTE: a cancelled caller still installed the entry; committing it here keeps a
            # one-node cluster from leaving it uncommitted until its next command.
            if self._log is next_log:
                self._advance_commit_index()
        return next_log.last_position

    def _advance_commit_index(self) -> None:
        """Raise this Leader's commit index as `Leadership.commit_index_after` allows."""
        self._commit_index = self._leadership.commit_index_after(
            self._commit_index, self._log, self._cluster.majority
        )

    def _align_role_records(self) -> None:
        """Make the Candidacy and Leadership match the current role and term.

        A Candidate gets a Candidacy and a Leader a Leadership (REPL-14, REPL-15)
        for the current term; a Follower gets neither. A record already for the
        current role and term is kept; any other is replaced, discarding an
        earlier term's votes and Follower progress.
        """
        role, term = self._state.role, self._state.current_term
        # NOTE: a new election leaves the role Candidate, so only the term check discards the
        # previous term's votes.
        if role is not Role.CANDIDATE:
            self._candidacy = None
        elif self._candidacy is None or self._candidacy.term != term:
            self._candidacy = Candidacy(term, self._state.node_id, self._cluster)
        if role is not Role.LEADER:
            self._leadership = None
        elif self._leadership is None or self._leadership.term != term:
            # NOTE: the clock resumes from the log's last entry, so cluster time never goes back
            # across a change of Leader and does not count the time without one (DD-32).
            self._leadership = Leadership(
                term, self._peers, self._log.last_index, self._log.last_cluster_time
            )
