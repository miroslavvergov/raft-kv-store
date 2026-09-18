"""A node's role, term, vote, and log, made durable under one lock (DD-8, DD-19, DD-22)."""

import asyncio
import copy
from typing import Any, Coroutine, Optional

from raftkv.consensus import Log, LogEntry, NodeState, Role
from raftkv.persistence.sqlite_store import SqliteStore


class DurableNodeState:
    """The single place a node's role, term, vote, and log are changed.

    Owns a pure `NodeState`, the node's `Log`, its `SqliteStore`, and the
    one per-node `asyncio.Lock` DD-8 requires. Every method that changes
    any of that state does all of the following while holding the lock,
    in this order (DD-19, DD-22):

    1. Compute the next state on a copy, using the pure consensus
       classes.
    2. Persist whatever part of it must be durable, and wait for the
       write to commit.
    3. Only then install it as the node's current in-memory state.

    Three properties follow. First, a caller that awaits one of these
    methods and only then responds to an RPC has persisted before
    responding (PERSIST-1 through PERSIST-3), and a Candidate that awaits
    `become_candidate` before sending RequestVote has persisted before
    sending (ELECT-5). Second, the in-memory state is never ahead of what
    is on disk: if a write fails, the method raises and the node's state
    is exactly what it was before the call. Third, what is on disk is
    never ahead of the in-memory state either: once a write has been
    handed to the store it runs to completion even if the calling task is
    cancelled meanwhile, the change is installed, and only then does the
    cancellation propagate. A write can finish on the store's background
    thread after the task that issued it has been cancelled, so a
    cancelled caller that simply stopped before step 3 would leave a
    change on disk that the node does not know it made.

    The lock is held across the `await` on the write, which is where
    `aiosqlite` suspends the calling coroutine while its background
    thread does the disk I/O. Any other coroutine calling one of these
    methods in the meantime waits for the lock rather than computing its
    own next state from the one about to be replaced.

    Attributes:
        node_id: This node's permanent positive-integer identity.
        role: The node's current Role. Held in memory only; STATE-2 has
            every node restart as a Follower.
        current_term: The node's current term, as last persisted.
        voted_for: The node ID voted for in `current_term`, as last
            persisted, or None.
        log: The node's log, as last persisted.
    """

    def __init__(self, state: NodeState, log: Log, store: SqliteStore) -> None:
        """Wrap state and a log that already match what `store` holds.

        `load` is the way to build one from a store; this constructor
        trusts that `state` and `log` are exactly what `store` would
        return from `SqliteStore.load`.

        Args:
            state: The node's current role, term, and vote.
            log: The node's current log.
            store: The open store holding the persisted copy of both.
        """
        self._state = state
        self._log = log
        self._store = store
        self._lock = asyncio.Lock()

    @classmethod
    async def load(cls, node_id: int, store: SqliteStore) -> "DurableNodeState":
        """Rebuild a node from its persisted state, in the Follower role.

        Implements start-up for PERSIST-4, PERSIST-5, PERSIST-6, and
        STATE-2: the term, vote, and log are reloaded from `store`, and
        the node begins as a Follower carrying them, whatever role it had
        before it stopped. A store that has never been written to yields
        a brand-new node at term 0 with no vote and an empty log.

        Args:
            node_id: This node's permanent positive-integer identity.
            store: The node's open store.

        Returns:
            The node's durable state, ready to accept or issue RPCs.
        """
        persisted = await store.load()
        state = NodeState.reloaded(node_id, persisted.current_term, persisted.voted_for)
        return cls(state, persisted.log, store)

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

    async def become_candidate(self) -> None:
        """Become Candidate, and persist the new term and self-vote before returning.

        Applies `NodeState.become_candidate` — STATE-3's Follower-to-
        Candidate or Candidate-to-Candidate edge, with ELECT-3's term
        increment and ELECT-4's vote for self — then persists the new
        term and vote. Returns only once that write has committed, so a
        caller that sends RequestVote after awaiting this satisfies
        ELECT-5.

        Raises:
            IllegalTransition: If the node is a Leader. Nothing is
                changed or written.
            sqlite3.Error: If the write fails. The node's in-memory
                state is left as it was before the call.
            asyncio.CancelledError: If the calling task was cancelled
                while the write was in flight — raised only after the
                write has committed and the new state is installed.
        """
        async with self._lock:
            next_state = copy.copy(self._state)
            next_state.become_candidate()
            await self._persist_then_install(
                self._store.save_term_and_vote(
                    next_state.current_term, next_state.voted_for
                ),
                next_state=next_state,
            )

    async def become_leader(self) -> None:
        """Become Leader.

        Applies `NodeState.become_leader` — STATE-3's Candidate-to-Leader
        edge, taken once ELECT-11's majority has been reached. Nothing is
        written: this transition changes only the role, and role is not
        persisted. The lock is still taken, because role is part of the
        state DD-8 guards.

        Raises:
            IllegalTransition: If the node is not a Candidate. Nothing is
                changed.
        """
        async with self._lock:
            self._state.become_leader()

    async def handle_observed_term(self, term: int) -> bool:
        """Catch up to a higher term seen in an RPC, persisting it before returning.

        Applies `NodeState.handle_observed_term` — STATE-5's term update,
        STATE-6's vote reset, and STATE-4's step-down for a Candidate or
        Leader — and, only when that actually changed something, persists
        the new term and cleared vote. Returns only once that write has
        committed, so a caller that responds to the RPC after awaiting
        this has persisted before responding (PERSIST-1, PERSIST-2).

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
        async with self._lock:
            next_state = copy.copy(self._state)
            if not next_state.handle_observed_term(term):
                return False
            await self._persist_then_install(
                self._store.save_term_and_vote(
                    next_state.current_term, next_state.voted_for
                ),
                next_state=next_state,
            )
            return True

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
        async with self._lock:
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
        committed, `next_state` and/or `next_log` are installed; if it
        failed, nothing is installed. A cancellation received meanwhile
        is raised only after that.

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
            if next_log is not None:
                self._log = next_log
        if cancelled:
            raise asyncio.CancelledError() from failure
        if failure is not None:
            raise failure
