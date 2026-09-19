"""How a node reports itself: etcd's log lines and trace events, worked out from its state."""

import logging
from typing import Any, Optional

from raftkv.consensus import RequestVoteRequest, RequestVoteResponse, Role
from raftkv.tracing.node_snapshot import NodeSnapshot
from raftkv.tracing.trace_event import TraceEvent, TraceMessage

LOG_LINES = "raftkv.node"
TRACE_EVENTS = "raftkv.trace"

_lines = logging.getLogger(LOG_LINES)
_events = logging.getLogger(TRACE_EVENTS)


class NodeTracer:
    """Reports one node's decisions in the two forms etcd's raft uses.

    - **Log lines** go to the `raftkv.node` logger at INFO, in etcd's raft
      log format: each starts with the node's ID, followed by the fact in
      etcd's wording — "2 became follower at term 1", "2 [logterm: 0,
      index: 0, vote: 0] cast RequestVote for 1 [logterm: 0, index: 0] at
      term 1". As in etcd, `vote: 0` means no vote has been cast.
    - **Trace events** go to the `raftkv.trace` logger at DEBUG, each one a
      `TraceEvent` attached to its log record as `record.trace_event`, the
      counterpart of etcd's `TracingEvent`.

    The `report_*` methods are called by the `traced` decorator after a
    node's method has run, with the node's state just before and just
    after the call. Everything a report says is read from those two
    snapshots and the call's own arguments and result: a change is
    reported only if the "after" state shows it was installed.

    Both kinds of output are off unless something enables those loggers:
    by default the `raftkv` logger has only a `NullHandler` and inherits
    the root logger's WARNING level, so `enabled` is False and `traced`
    skips the reports entirely. This is how etcd keeps tracing free when
    it is not compiled in.

    Handlers attached to these loggers run on the event loop, inside the
    node's coroutines. A handler that writes to a file or socket there
    would block the loop, which NODE-7 forbids; a production node sends
    records through `logging.handlers.QueueHandler` so that a background
    thread does the writing.

    Attributes:
        node_id: The node this tracer reports for.
    """

    def __init__(self, node_id: int) -> None:
        """Create a tracer for one node.

        Args:
            node_id: The node's positive-integer ID, written at the start
                of every log line and in every event.
        """
        self.node_id = node_id

    @property
    def enabled(self) -> bool:
        """Whether either log lines or trace events would be emitted."""
        return _lines.isEnabledFor(logging.INFO) or _events.isEnabledFor(logging.DEBUG)

    # --- Reports: one per traced DurableNodeState method -------------------------

    def report_started(
        self, before: None, after: NodeSnapshot, *constructor_args: Any,
        result: None, error: None,
    ) -> None:
        """Report a node just built from its persisted state: etcd's InitState.

        `constructor_args` are the constructor's own arguments; the
        report reads the node's state from `after` instead.
        """
        self.line(
            "started [peers: %s, term: %d, vote: %d, lastindex: %d, lastterm: %d]",
            sorted(after.peers), after.term, after.vote or 0,
            after.last_log.index, after.last_log.term,
        )
        self.event("InitState", after, properties={"peers": sorted(after.peers)})

    def report_election(
        self, before: NodeSnapshot, after: NodeSnapshot, *,
        result: Optional[RequestVoteRequest], error: Optional[BaseException],
    ) -> None:
        """Report `start_election`: the new candidacy, a win, and each request sent.

        Nothing is reported as changed unless the term went up — a failed
        write, or a Leader refused the transition. The requests are
        reported as sent only if the call returned them.
        """
        if after.term == before.term:
            return
        self.line("is starting a new election at term %d", before.term)
        self.line("became candidate at term %d", after.term)
        # In a one-node cluster the candidacy is won within the same call;
        # the Candidate state is the one that was persisted, before the win.
        self.event("BecomeCandidate", after.as_role(Role.CANDIDATE))
        if after.role is Role.LEADER:
            self._became_leader(after)
        if result is None:
            return
        for peer in sorted(after.peers):
            self.line(
                "[logterm: %d, index: %d] sent RequestVote request to %d at term %d",
                result.last_log_term, result.last_log_index, peer, result.term,
            )
            self.event("SendRequestVoteRequest", after, TraceMessage.from_request(result, peer))

    def report_observed_term(
        self, before: NodeSnapshot, after: NodeSnapshot, term: int, *,
        result: Optional[bool], error: Optional[BaseException],
    ) -> None:
        """Report `handle_observed_term`: a catch-up to a higher term, if one was installed."""
        if after.term > before.term:
            self.line("[term: %d] observed a higher term %d", before.term, term)
            self._became_follower(after)

    def report_vote_request(
        self, before: NodeSnapshot, after: NodeSnapshot, request: RequestVoteRequest, *,
        result: Optional[RequestVoteResponse], error: Optional[BaseException],
    ) -> None:
        """Report `handle_vote_request`: the request, any catch-up, the decision, the answer.

        The decision line shows, as etcd's does, the vote the node held
        in the request's term before the request arrived: its vote if it
        was already in that term, and none if it was in an earlier one.
        If no answer was returned — the write failed, or the caller was
        cancelled — no answer is reported as sent; a vote the cancelled
        call installed anyway is reported as such.
        """
        self.event(
            "ReceiveRequestVoteRequest", before, TraceMessage.from_request(request, self.node_id)
        )
        if request.term > before.term:
            self.line(
                "[term: %d] received a RequestVote message with higher term from %d [term: %d]",
                before.term, request.candidate_id, request.term,
            )
        if after.term > before.term:
            self._became_follower(after)
        if result is None:
            if after.vote is not None and (after.term, after.vote) != (before.term, before.vote):
                self.line(
                    "[term: %d] voted for %d, but no answer was returned", after.term, after.vote
                )
            return
        if request.term < after.term:
            self.line(
                "[term: %d] rejected a RequestVote message with lower term from %d [term: %d]",
                after.term, request.candidate_id, request.term,
            )
        else:
            vote_in_request_term = before.vote if before.term == request.term else None
            self.line(
                "[logterm: %d, index: %d, vote: %d] %s RequestVote %s %d "
                "[logterm: %d, index: %d] at term %d",
                before.last_log.term, before.last_log.index, vote_in_request_term or 0,
                "cast" if result.vote_granted else "rejected",
                "for" if result.vote_granted else "from",
                request.candidate_id, request.last_log_term, request.last_log_index, after.term,
            )
        self.event(
            "SendRequestVoteResponse",
            after,
            TraceMessage.from_response(result, self.node_id, request.candidate_id),
        )

    def report_vote_response(
        self, before: NodeSnapshot, after: NodeSnapshot,
        voter: int, sent_in_term: int, response: RequestVoteResponse, *,
        result: Optional[bool], error: Optional[BaseException],
    ) -> None:
        """Report `handle_vote_response`: a step-down, or the answer counted or ignored, and a win.

        Whether the answer was counted is read from the Candidacy itself:
        its tally grew, or it did not.
        """
        self.event(
            "ReceiveRequestVoteResponse",
            before,
            TraceMessage.from_response(response, voter, self.node_id),
            {"sentInTerm": sent_in_term},
        )
        if response.term > before.term:
            self.line(
                "[term: %d] received a RequestVoteResponse message with higher term "
                "from %d [term: %d]",
                before.term, voter, response.term,
            )
            if after.term > before.term:
                self._became_follower(after)
            return
        tally = before.candidacy
        counted = tally is not None and (tally.votes_granted, tally.votes_refused) != (
            before.votes_granted, before.votes_refused,
        )
        if not counted:
            if error is None:
                self.line(
                    "[term: %d, role: %s] ignored a RequestVoteResponse message from %d "
                    "[sent in term: %d]",
                    before.term, before.role.value, voter, sent_in_term,
                )
            return
        self.line(
            "received RequestVoteResponse %sfrom %d at term %d",
            "" if response.vote_granted else "rejection ", voter, before.term,
        )
        self.line(
            "has received %d RequestVoteResponse votes and %d vote rejections",
            len(tally.votes_granted), len(tally.votes_refused),
        )
        if after.role is Role.LEADER and before.role is not Role.LEADER:
            self._became_leader(after)

    # --- Output -----------------------------------------------------------------------

    def line(self, message: str, *args: Any) -> None:
        """Emit one etcd-format log line, prefixed with this node's ID.

        Args:
            message: A %-style format string for the rest of the line.
            *args: Its arguments, formatted only if the line is emitted.
        """
        if _lines.isEnabledFor(logging.INFO):
            _lines.info("%d " + message, self.node_id, *args, extra={"node_id": self.node_id})

    def event(
        self,
        name: str,
        state: NodeSnapshot,
        message: Optional[TraceMessage] = None,
        properties: Optional[dict[str, Any]] = None,
    ) -> None:
        """Emit one trace event carrying `state`.

        Args:
            name: The etcd event name, such as "BecomeLeader".
            state: The node's state for this event.
            message: The RPC sent or received, for message events.
            properties: Extra facts for this event.
        """
        if not _events.isEnabledFor(logging.DEBUG):
            return
        event = TraceEvent(
            name=name,
            node_id=self.node_id,
            role=state.role.value,
            term=state.term,
            vote=state.vote,
            last_log_index=state.last_log.index,
            last_log_term=state.last_log.term,
            message=message,
            properties=properties or {},
        )
        _events.debug(name, extra={"node_id": self.node_id, "trace_event": event})

    def _became_follower(self, after: NodeSnapshot) -> None:
        self.line("became follower at term %d", after.term)
        self.event("BecomeFollower", after)

    def _became_leader(self, after: NodeSnapshot) -> None:
        self.line("became leader at term %d", after.term)
        self.event(
            "BecomeLeader", after, properties={"next": after.next_index, "match": after.match_index}
        )
