"""One structured trace event: a node's state at a step, and the message involved."""

from dataclasses import dataclass, field
from typing import Any

from raftkv.consensus import RequestVoteRequest, RequestVoteResponse


@dataclass(frozen=True)
class TraceMessage:
    """An RPC as a trace event shows it: who sent what to whom, in which term.

    `as_dict` uses etcd's trace-format keys, so a trace reads like etcd's.

    Attributes:
        type: "RequestVote" or "RequestVoteResponse".
        sender: The sender's node ID.
        receiver: The receiver's node ID.
        term: The term the message carries.
        log_term: The Candidate's last log term, for a request; else None.
        index: The Candidate's last log index, for a request; else None.
        reject: Whether the vote was refused, for a response; else None.
    """

    type: str
    sender: int
    receiver: int
    term: int
    log_term: int | None = None
    index: int | None = None
    reject: bool | None = None

    @classmethod
    def from_vote_request(cls, request: RequestVoteRequest, receiver: int) -> "TraceMessage":
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
    def from_vote_response(
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
        # NOTE: `is not None`, not truthiness, so `reject: false` and a 0 index or log term stay.
        return {key: value for key, value in fields.items() if value is not None}


@dataclass(frozen=True)
class TraceEvent:
    """One step of a node in the shape of etcd's `TracingEvent`, with its state.

    Names are etcd's: `InitState`, `BecomeCandidate`, `BecomeFollower`,
    `BecomeLeader`, `SendRequestVoteRequest`, `ReceiveRequestVoteRequest`,
    `SendRequestVoteResponse`, `ReceiveRequestVoteResponse`; plus `PersistVote`,
    for a vote a cancelled call installed without answering. Receive events
    carry the state the message arrived to; all others, the state after the
    step. A change appears only once persisted and installed, so a trace never
    shows a term or vote the node did not hold.

    Attributes:
        name: The event name.
        node_id: The node the event happened on.
        role: The node's role, as `Role.value`.
        term: The node's `current_term`.
        vote: The node's `voted_for`, or None.
        last_log_index: The index of the node's last log entry.
        last_log_term: The term of the node's last log entry.
        message: The RPC sent or received, for message events; else None.
        properties: Extra facts, such as a new Leader's per-Follower progress.
    """

    name: str
    node_id: int
    role: str
    term: int
    vote: int | None
    last_log_index: int
    last_log_term: int
    message: TraceMessage | None = None
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
