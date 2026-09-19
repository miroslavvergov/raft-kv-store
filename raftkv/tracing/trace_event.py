"""One structured trace event: a node's state at a step, and the message involved."""

from dataclasses import dataclass, field
from typing import Any, Optional

from raftkv.consensus import RequestVoteRequest, RequestVoteResponse


@dataclass(frozen=True)
class TraceMessage:
    """An RPC as it appears in a trace event: who sent what to whom, in which term.

    Keys in `as_dict` follow etcd's trace format (`type`, `term`, `from`,
    `to`, `logTerm`, `index`, `reject`), so a trace reads the same way as
    one recorded from etcd.

    Attributes:
        type: "RequestVote" or "RequestVoteResponse".
        sender: The node ID that sent it.
        receiver: The node ID it was sent to.
        term: The term the message carries.
        log_term: For a request, the term of the Candidate's last log
            entry; None for a response.
        index: For a request, the index of the Candidate's last log
            entry; None for a response.
        reject: For a response, True if the vote was refused; None for a
            request.
    """

    type: str
    sender: int
    receiver: int
    term: int
    log_term: Optional[int] = None
    index: Optional[int] = None
    reject: Optional[bool] = None

    @classmethod
    def from_request(cls, request: RequestVoteRequest, receiver: int) -> "TraceMessage":
        """Describe a RequestVote from its Candidate to `receiver`."""
        return cls(
            type="RequestVote",
            sender=request.candidate_id,
            receiver=receiver,
            term=request.term,
            log_term=request.last_log_term,
            index=request.last_log_index,
        )

    @classmethod
    def from_response(
        cls, response: RequestVoteResponse, sender: int, receiver: int
    ) -> "TraceMessage":
        """Describe a voter's answer, from `sender` (the voter) to `receiver` (the Candidate)."""
        return cls(
            type="RequestVoteResponse",
            sender=sender,
            receiver=receiver,
            term=response.term,
            reject=not response.vote_granted,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the message in etcd's trace-format keys, leaving out fields that don't apply."""
        fields = {
            "type": self.type,
            "term": self.term,
            "from": self.sender,
            "to": self.receiver,
            "logTerm": self.log_term,
            "index": self.index,
            "reject": self.reject,
        }
        return {key: value for key, value in fields.items() if value is not None}


@dataclass(frozen=True)
class TraceEvent:
    """A named step in one node's life, with the node's state right after it.

    Mirrors etcd's `TracingEvent`: every event carries the node's ID, role,
    term, and vote, how far its log goes, and — for sending or receiving an
    RPC — the message. Event names are etcd's: `InitState`,
    `BecomeCandidate`, `BecomeFollower`, `BecomeLeader`,
    `SendRequestVoteRequest`, `ReceiveRequestVoteRequest`,
    `SendRequestVoteResponse`, and `ReceiveRequestVoteResponse`.

    The state is always state the node has actually installed: an event is
    emitted only after the change it describes has been persisted and made
    the node's current state, so a trace never shows a term or vote the
    node did not really hold.

    Attributes:
        name: The event name.
        node_id: The node the event happened on.
        role: The node's role, as `Role.value`.
        term: The node's `current_term`.
        vote: The node's `voted_for`, or None.
        last_log_index: The index of the node's last log entry.
        last_log_term: The term of the node's last log entry.
        message: The RPC sent or received, for message events.
        properties: Extra facts for this event, such as a new Leader's
            per-Follower progress.
    """

    name: str
    node_id: int
    role: str
    term: int
    vote: Optional[int]
    last_log_index: int
    last_log_term: int
    message: Optional[TraceMessage] = None
    properties: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return the event in etcd's trace-format shape, ready to write as one JSON line."""
        event: dict[str, Any] = {
            "name": self.name,
            "nid": self.node_id,
            "role": self.role,
            "state": {"term": self.term, "vote": self.vote},
            "log": {"index": self.last_log_index, "term": self.last_log_term},
        }
        if self.message is not None:
            event["msg"] = self.message.as_dict()
        if self.properties:
            event["prop"] = self.properties
        return event
