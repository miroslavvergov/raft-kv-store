"""How a node reports itself: etcd-format log lines and trace events derived from its state."""

import logging
from collections.abc import Callable
from typing import Any

from raftkv.consensus import (
    AppendEntriesRequest,
    AppendEntriesResponse,
    LogPosition,
    RequestVoteRequest,
    RequestVoteResponse,
    Role,
)
from raftkv.tracing.node_snapshot import NodeSnapshot
from raftkv.tracing.trace_event import TraceEvent, TraceMessage

LOG_LINES_LOGGER = "raftkv.node"
TRACE_EVENTS_LOGGER = "raftkv.trace"

_line_logger = logging.getLogger(LOG_LINES_LOGGER)
_event_logger = logging.getLogger(TRACE_EVENTS_LOGGER)


class NodeTracer:
    """Reports one node's decisions as etcd-format log lines and trace events.

    Log lines go to the `raftkv.node` logger at INFO, prefixed with the node's
    ID and worded as etcd's ("2 became follower at term 1"; `vote: 0` means no
    vote). Trace events go to `raftkv.trace` at DEBUG, as
    `record.trace_event`. `traced` calls each `report_*` method with the
    call's before and after snapshots, its arguments, and its result or error;
    a change is reported only if "after" shows it installed. Unless a logger is
    enabled for its level (by default both inherit WARNING), `enabled` is False
    and `traced` skips reporting entirely.

    Handlers run on the event loop, inside the node's coroutines; a blocking
    handler would violate NODE-7, so route records through a
    `logging.handlers.QueueHandler`.

    Attributes:
        node_id: The node this tracer reports for.
    """

    def __init__(self, node_id: int) -> None:
        """Create a tracer for one node.

        Args:
            node_id: The node's ID, written at the start of every log line and in
                every event.
        """
        self.node_id = node_id

    @property
    def enabled(self) -> bool:
        """Whether either log lines or trace events would be emitted."""
        return _line_logger.isEnabledFor(logging.INFO) or _event_logger.isEnabledFor(logging.DEBUG)

    # --- Reports: one per traced DurableNodeState method ------------------------------

    def report_started(self, after: NodeSnapshot) -> None:
        """Report a node just built from its persisted state: etcd's InitState."""
        self.emit_line(
            "started [peers: %s, term: %d, vote: %d, lastindex: %d, lastterm: %d]",
            sorted(after.peers),
            after.current_term,
            after.voted_for or 0,
            after.last_log_position.index,
            after.last_log_position.term,
        )
        self.emit_event("InitState", after, properties={"peers": sorted(after.peers)})

    def report_start_election(
        self,
        before: NodeSnapshot,
        after: NodeSnapshot,
        still_due: Callable[[], bool] | None,
        *,
        result: RequestVoteRequest | None,
        error: BaseException | None,
    ) -> None:
        """Report `start_election`: the new candidacy, a win, and each request sent.

        Nothing is reported unless the term went up; it does not after a failed
        write or a refused Leader. Requests are reported as sent only if the call
        returned them.
        """
        if after.current_term == before.current_term:
            return
        # NOTE: this line names the term the election starts from, so it reads `before`.
        self.emit_line("is starting a new election at term %d", before.current_term)
        self.emit_line("became candidate at term %d", after.current_term)
        # NOTE: a one-node cluster wins within this call, so `after` is already Leader; the
        # event carries the Candidate role the node passed through.
        self.emit_event("BecomeCandidate", after.with_role(Role.CANDIDATE))
        if after.role is Role.LEADER:
            self._emit_became_leader(after)
            self._emit_replicate(before, after)
            self._emit_commit(before, after)
        if result is None:
            return
        for peer in sorted(after.peers):
            self.emit_line(
                "[logterm: %d, index: %d] sent RequestVote request to %d at term %d",
                result.last_log_term,
                result.last_log_index,
                peer,
                result.term,
            )
            self.emit_event(
                "SendRequestVoteRequest", after, TraceMessage.from_vote_request(result, peer)
            )

    def report_observed_term(
        self,
        before: NodeSnapshot,
        after: NodeSnapshot,
        term: int,
        *,
        result: bool | None,
        error: BaseException | None,
    ) -> None:
        """Report `handle_observed_term`: a catch-up to a higher term, if one was installed."""
        if after.current_term > before.current_term:
            self.emit_line("[term: %d] observed a higher term %d", before.current_term, term)
            self._emit_became_follower(after)

    def report_vote_request(
        self,
        before: NodeSnapshot,
        after: NodeSnapshot,
        request: RequestVoteRequest,
        *,
        result: RequestVoteResponse | None,
        error: BaseException | None,
    ) -> None:
        """Report `handle_vote_request`: the request, any catch-up, the decision, the answer.

        As in etcd, the decision line shows the vote held in the request's term
        before it arrived (none if the node was in an earlier term). If no answer
        was returned (failed write or cancelled caller), none is reported as sent,
        but a vote the cancelled call installed is reported.
        """
        # NOTE: a Receive event carries the state the message arrived to, so it reads `before`.
        self.emit_event(
            "ReceiveRequestVoteRequest",
            before,
            TraceMessage.from_vote_request(request, self.node_id),
        )
        if request.term > before.current_term:
            self.emit_line(
                "[term: %d] received a RequestVote message with higher term from %d [term: %d]",
                before.current_term,
                request.candidate_id,
                request.term,
            )
        if after.current_term > before.current_term:
            self._emit_became_follower(after)
        if result is None:
            # NOTE: the whole (term, vote) pair is compared, since a vote for the same node in
            # a higher term is a new vote.
            installed = (after.current_term, after.voted_for)
            if after.voted_for is not None and installed != (before.current_term, before.voted_for):
                self.emit_line(
                    "[term: %d] voted for %d, but no answer was returned",
                    after.current_term,
                    after.voted_for,
                )
                self.emit_event("PersistVote", after)
            return
        if request.term < after.current_term:
            self.emit_line(
                "[term: %d] rejected a RequestVote message with lower term from %d [term: %d]",
                after.current_term,
                request.candidate_id,
                request.term,
            )
        else:
            # NOTE: `after.voted_for` may be the vote just cast; the line shows the vote held
            # in the request's term before it arrived.
            vote_in_request_term = before.voted_for if before.current_term == request.term else None
            self.emit_line(
                "[logterm: %d, index: %d, vote: %d] %s RequestVote %s %d "
                "[logterm: %d, index: %d] at term %d",
                before.last_log_position.term,
                before.last_log_position.index,
                vote_in_request_term or 0,
                "cast" if result.vote_granted else "rejected",
                "for" if result.vote_granted else "from",
                request.candidate_id,
                request.last_log_term,
                request.last_log_index,
                after.current_term,
            )
        self.emit_event(
            "SendRequestVoteResponse",
            after,
            TraceMessage.from_vote_response(result, self.node_id, request.candidate_id),
        )

    def report_vote_response(
        self,
        before: NodeSnapshot,
        after: NodeSnapshot,
        voter: int,
        sent_in_term: int,
        response: RequestVoteResponse,
        *,
        result: bool | None,
        error: BaseException | None,
    ) -> None:
        """Report `handle_vote_response`: a step-down, or the answer counted or ignored, and a win.

        Whether the answer was counted is read from the live Candidacy: its tally
        grew during the call, or it did not.
        """
        self.emit_event(
            "ReceiveRequestVoteResponse",
            before,
            TraceMessage.from_vote_response(response, voter, self.node_id),
            {"sentInTerm": sent_in_term},
        )
        if response.term > before.current_term:
            self.emit_line(
                "[term: %d] received a RequestVoteResponse message with higher term "
                "from %d [term: %d]",
                before.current_term,
                voter,
                response.term,
            )
            if after.current_term > before.current_term:
                self._emit_became_follower(after)
            return
        # NOTE: a win discards the Candidacy, so the live object from `before` holds the
        # final tally.
        tally = before.live_candidacy
        counted = tally is not None and (tally.votes_granted, tally.votes_refused) != (
            before.votes_granted,
            before.votes_refused,
        )
        if not counted:
            # NOTE: a call that raised, as on a non-member voter, did not ignore the answer, so
            # only a completed call reports it as ignored.
            if error is None:
                self.emit_line(
                    "[term: %d, role: %s] ignored a RequestVoteResponse message from %d "
                    "[sent in term: %d]",
                    before.current_term,
                    before.role.value,
                    voter,
                    sent_in_term,
                )
            return
        self.emit_line(
            "received RequestVoteResponse %sfrom %d at term %d",
            "" if response.vote_granted else "rejection ",
            voter,
            before.current_term,
        )
        self.emit_line(
            "has received %d RequestVoteResponse votes and %d vote rejections",
            len(tally.votes_granted),
            len(tally.votes_refused),
        )
        if after.role is Role.LEADER and before.role is not Role.LEADER:
            self._emit_became_leader(after)
            self._emit_replicate(before, after)
            self._emit_commit(before, after)

    def report_append_entries(
        self,
        before: NodeSnapshot,
        after: NodeSnapshot,
        request: AppendEntriesRequest,
        *,
        result: AppendEntriesResponse | None,
        error: BaseException | None,
    ) -> None:
        """Report `handle_append_entries`: the RPC, any step-down, the rejection, the answer.

        A step-down is reported whether a higher term caused it (STATE-4, STATE-5)
        or a Leader of the node's own term did (STATE-7). As in etcd, an accepted
        RPC gets no line of its own; only the event records it. If no answer was
        returned (failed write or cancelled caller), none is reported as sent.
        """
        # NOTE: a Receive event carries the state the message arrived to, so it reads `before`.
        self.emit_event(
            "ReceiveAppendEntriesRequest",
            before,
            TraceMessage.from_append_entries_request(request, self.node_id),
        )
        if request.term > before.current_term:
            self.emit_line(
                "[term: %d] received a MsgApp message with higher term from %d [term: %d]",
                before.current_term,
                request.leader_id,
                request.term,
            )
        # NOTE: a term rise reports a step-down even from Follower; the role check adds the case
        # with no term rise, a Candidate stepping down for a Leader of its own term (STATE-7).
        stepped_down = after.role is Role.FOLLOWER and before.role is not Role.FOLLOWER
        if after.current_term > before.current_term or stepped_down:
            self._emit_became_follower(after)
        if result is None:
            return
        if not result.success:
            if request.term < after.current_term:
                self.emit_line(
                    "[term: %d] rejected a MsgApp message with lower term from %d [term: %d]",
                    after.current_term,
                    request.leader_id,
                    request.term,
                )
            # NOTE: `after`, not `before`: a Leader that steps down for a higher term and then
            # fails the log check was a Leader before the call, but its rejection is the log's.
            elif after.role is Role.LEADER:
                self.emit_line(
                    "[term: %d, role: leader] rejected a MsgApp message from %d at the same term",
                    after.current_term,
                    request.leader_id,
                )
            else:
                self.emit_line(
                    "[logterm: %d, index: %d] rejected MsgApp [logterm: %d, index: %d] from %d",
                    after.last_log_position.term,
                    after.last_log_position.index,
                    request.prev_log_term,
                    request.prev_log_index,
                    request.leader_id,
                )
        self._emit_commit(before, after)
        self.emit_event(
            "SendAppendEntriesResponse",
            after,
            TraceMessage.from_append_entries_response(result, self.node_id, request.leader_id),
        )

    def report_append_command(
        self,
        before: NodeSnapshot,
        after: NodeSnapshot,
        command: str,
        *,
        result: LogPosition | None,
        error: BaseException | None,
    ) -> None:
        """Report `append_command`: the entry the Leader appended, and a commit it made.

        Nothing is reported if no entry was installed: a refused command, a
        non-Leader, or a failed write.
        """
        self._emit_replicate(before, after)
        self._emit_commit(before, after)

    def report_append_entries_request(
        self,
        before: NodeSnapshot,
        after: NodeSnapshot,
        follower: int,
        *,
        result: AppendEntriesRequest | None,
        error: BaseException | None,
    ) -> None:
        """Report `append_entries_request_for`: the AppendEntries built for a Follower."""
        if result is not None:
            self.emit_event(
                "SendAppendEntriesRequest",
                after,
                TraceMessage.from_append_entries_request(result, follower),
            )

    def report_append_entries_response(
        self,
        before: NodeSnapshot,
        after: NodeSnapshot,
        follower: int,
        request: AppendEntriesRequest,
        response: AppendEntriesResponse,
        *,
        result: bool | None,
        error: BaseException | None,
    ) -> None:
        """Report `handle_append_entries_response`: a step-down, a rejection, or a commit.

        Always reports the answer received. Unless the call raised, it adds a line for
        a step-down on a higher term, for an answer ignored as sent in another term or
        reaching a non-Leader, for a rejection that lowered the Follower's
        `next_index`, and for one that lowered nothing. Any rise in the commit index
        is reported too.
        """
        self.emit_event(
            "ReceiveAppendEntriesResponse",
            before,
            TraceMessage.from_append_entries_response(response, follower, self.node_id),
            {"sentInTerm": request.term},
        )
        if response.term > before.current_term:
            self.emit_line(
                "[term: %d] received a MsgAppResp message with higher term from %d [term: %d]",
                before.current_term,
                follower,
                response.term,
            )
            if after.current_term > before.current_term:
                self._emit_became_follower(after)
            return
        if before.role is not Role.LEADER or request.term != before.current_term:
            if error is None:
                self.emit_line(
                    "[term: %d, role: %s] ignored a MsgAppResp message from %d [sent in term: %d]",
                    before.current_term,
                    before.role.value,
                    follower,
                    request.term,
                )
            return
        if not response.success:
            if result:
                self.emit_line(
                    "received MsgAppResp(rejected) from %d for index %d",
                    follower,
                    request.prev_log_index,
                )
                self.emit_line(
                    "decreased progress of %d to [next = %d, match = %d]",
                    follower,
                    after.next_index[follower],
                    after.match_index[follower],
                )
            elif error is None:
                # NOTE: a duplicate, a rejection of a probe already backed off from, or one at
                # the floor: nothing moved, so the trace says why the answer changed nothing.
                self.emit_line(
                    "ignored MsgAppResp(rejected) from %d for index %d [next = %d]",
                    follower,
                    request.prev_log_index,
                    after.next_index[follower],
                )
        self._emit_commit(before, after)

    def report_apply_committed(
        self,
        before: NodeSnapshot,
        after: NodeSnapshot,
        max_entries: int | None,
        *,
        result: int | None,
        error: BaseException | None,
    ) -> None:
        """Report `apply_committed`: every index the node applied, empty entries included.

        An empty entry takes an index without reaching the state machine, so reporting
        it keeps a trace's indexes aligned with the log's.
        """
        if after.last_applied > before.last_applied:
            self.emit_event(
                "Apply",
                after,
                properties={
                    "applied": after.last_applied,
                    "entries": _entries(after, before.last_applied + 1, after.last_applied),
                },
            )

    # --- Output -----------------------------------------------------------------------

    def emit_line(self, message: str, *args: Any) -> None:
        """Emit one etcd-format log line, prefixed with this node's ID.

        Args:
            message: A %-style format string for the rest of the line.
            *args: Its arguments, formatted only if the line is emitted.
        """
        if _line_logger.isEnabledFor(logging.INFO):
            _line_logger.info("%d " + message, self.node_id, *args, extra={"node_id": self.node_id})

    def emit_event(
        self,
        name: str,
        state: NodeSnapshot,
        message: TraceMessage | None = None,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Emit one trace event carrying `state`, if the trace logger is enabled.

        Args:
            name: The etcd event name, such as "BecomeLeader".
            state: The node's state for this event.
            message: The RPC sent or received, for message events; else None.
            properties: Extra facts for this event, if any.
        """
        if not _event_logger.isEnabledFor(logging.DEBUG):
            return
        event = TraceEvent(
            name=name,
            node_id=self.node_id,
            role=state.role.value,
            term=state.current_term,
            vote=state.voted_for,
            last_log_index=state.last_log_position.index,
            last_log_term=state.last_log_position.term,
            message=message,
            properties=properties or {},
        )
        _event_logger.debug(name, extra={"node_id": self.node_id, "trace_event": event})

    def _emit_became_follower(self, after: NodeSnapshot) -> None:
        self.emit_line("became follower at term %d", after.current_term)
        self.emit_event("BecomeFollower", after)

    def _emit_became_leader(self, after: NodeSnapshot) -> None:
        self.emit_line("became leader at term %d", after.current_term)
        # NOTE: the whole log goes with it, so a trace alone shows whether a new Leader holds
        # every entry committed before its term.
        self.emit_event(
            "BecomeLeader",
            after,
            properties={
                "next": after.next_index,
                "match": after.match_index,
                "log": [[entry.term, entry.cluster_time, entry.command] for entry in after.log],
            },
        )

    def _emit_replicate(self, before: NodeSnapshot, after: NodeSnapshot) -> None:
        """Report the entries a Leader appended to its own log during the call."""
        if after.log.last_index > before.log.last_index:
            self.emit_event(
                "Replicate",
                after,
                properties={"entries": _entries(after, before.log.last_index + 1, None)},
            )

    def _emit_commit(self, before: NodeSnapshot, after: NodeSnapshot) -> None:
        """Report a rise in the commit index, with every entry it newly commits."""
        if after.commit_index > before.commit_index:
            self.emit_event(
                "Commit",
                after,
                properties={
                    "commit": after.commit_index,
                    "entries": _entries(after, before.commit_index + 1, after.commit_index),
                },
            )


def _entries(state: NodeSnapshot, first: int, last: int | None) -> list[list[Any]]:
    """Return `state`'s log entries from `first` through `last` (the end if None).

    Each is `[index, term, cluster time, command]`, so a trace shows exactly which entry each
    index holds.
    """
    last = state.log.last_index if last is None else last
    entries = []
    for index in range(first, last + 1):
        entry = state.log.entry_at(index)
        entries.append([index, entry.term, entry.cluster_time, entry.command])
    return entries
