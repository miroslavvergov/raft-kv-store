"""Exceptions raised by the pure Raft consensus core."""


class IllegalTransition(Exception):
    """Raised when a requested role change is not one of STATE-3's edges.

    STATE-3's five legal edges are: Follower to Candidate, Candidate to
    Candidate, Candidate to Leader, Candidate to Follower, and Leader to
    Follower. STATE-3 is a closure statement — these are the *only*
    edges that may ever occur — so an attempted illegal transition is
    treated as a programming error in the caller, not a recoverable
    runtime condition, and `NodeState` is left completely unchanged when
    it's raised.
    """
