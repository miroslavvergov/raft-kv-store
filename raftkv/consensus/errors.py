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
