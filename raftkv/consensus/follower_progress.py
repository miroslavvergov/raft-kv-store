"""A Leader's per-follower replication-progress tracking (nextIndex)."""


class FollowerProgress:
    """The Leader's own view of one Follower's replication progress.

    REPL-6 speaks of the Leader's "stored nextIndex for that Follower" —
    a value that persists and evolves across an entire term as
    AppendEntries RPCs to this one follower succeed or get rejected. One
    FollowerProgress holds that value for one follower.

    Attributes:
        next_index: The index of the next log entry the Leader will send
            to this follower.
    """

    def __init__(self, next_index: int) -> None:
        """Start tracking a follower at a given nextIndex.

        Per §5.3, a newly elected Leader initializes every follower's
        nextIndex to one past its own last log index (11, for the
        10-entry leader log in Figure 7) — computing that starting value
        is the caller's job; this constructor just records it.

        Args:
            next_index: The nextIndex to start tracking for this
                follower.
        """
        self.next_index = next_index

    def record_rejection(self) -> None:
        """Decrement nextIndex after this follower rejects an AppendEntries.

        Implements REPL-6: "whenever a Leader's AppendEntries RPC is
        rejected under REPL-5, the Leader shall decrement its stored
        nextIndex for that Follower." The floor at 1 exists because
        `Log.matches` already treats a `prev_log_index` of 0 as
        automatically satisfied — nextIndex can never usefully fall
        below 1, since index 0 needs no agreement check at all, and
        decrementing past it would just repeat an
        already-guaranteed-to-succeed probe forever.

        This is only half of the repair loop REPL-6 and REPL-7 describe
        together: the retrying itself — checking `Log.matches` again
        with the new, lower `next_index`, and calling this again on
        another rejection — is REPL-7's job, and is therefore the
        caller's responsibility, not this method's.
        """
        self.next_index = max(1, self.next_index - 1)
