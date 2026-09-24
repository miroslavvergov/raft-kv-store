"""Exceptions raised by the pure Raft consensus core."""


class IllegalTransitionError(Exception):
    """Raised on a role change outside STATE-3's five legal edges.

    The legal edges are Follower to Candidate, Candidate to Candidate, Candidate
    to Leader, Candidate to Follower, and Leader to Follower. It signals a bug
    in the caller, not a recoverable condition, and `NodeState` is unchanged.
    """


class NotLeaderError(Exception):
    """Raised when a node that is not Leader is asked to do a Leader's work.

    Only a Leader appends a client command (REPL-1, CLIENT-6) or replicates its
    log (REPL-2). Nothing changes when it is raised.
    """


class CommittedEntryConflictError(Exception):
    """Raised when an AppendEntries would change an entry this node has committed.

    Committed entries are never overwritten: the election and commit rules
    guarantee a legitimate Leader already holds every one of them. Seeing this
    means one of those rules has failed, so the node refuses the RPC with
    nothing changed rather than rewrite history it may already have applied.
    """
