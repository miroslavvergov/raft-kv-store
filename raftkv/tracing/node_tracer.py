"""How a node reports itself: etcd-format log lines and trace events derived from its state."""

import logging
from typing import Any

from raftkv.consensus import RequestVoteRequest, RequestVoteResponse, Role
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

    # --- Reports: one per traced DurableNodeState method -------------------------

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
        self.emit_line("is starting a new election at term %d", before.current_term)
        self.emit_line("became candidate at term %d", after.current_term)
        # In a one-node cluster the candidacy is won within the same call;
        # the Candidate state is the one that was persisted, before the win.
        self.emit_event("BecomeCandidate", after.with_role(Role.CANDIDATE))
        if after.role is Role.LEADER:
            self._emit_became_leader(after)
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
        tally = before.live_candidacy
        counted = tally is not None and (tally.votes_granted, tally.votes_refused) != (
            before.votes_granted,
            before.votes_refused,
        )
        if not counted:
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
        self.emit_event(
            "BecomeLeader", after, properties={"next": after.next_index, "match": after.match_index}
        )
