"""A Leader's per-follower replication progress: nextIndex and matchIndex."""


class FollowerProgress:
    """How far one Follower's log is known to match the Leader's.

    Holds the two values a Leader tracks, in memory only, for one
    follower:

    - `next_index` — the index of the next entry the Leader will send.
      It is a guess, lowered on every rejection (REPL-6) until the
      Follower's log agrees with the Leader's (REPL-5).
    - `match_index` — the highest index known to be identical on the
      Follower and the Leader. It is never a guess: it moves only when
      the Follower has actually acknowledged a successful AppendEntries
      (REPL-16), which is what lets the Leader count the Follower toward
      the majority APPLY-1 requires before an entry can be committed.

    Both values change only through `record_success` and
    `record_rejection` (DD-25), and together they keep one invariant
    after every call: `match_index < next_index`. Everything up to
    `match_index` is already confirmed, so the next entry the Leader
    sends is always past it.

    A FollowerProgress knows nothing about terms. In the Leader, every
    FollowerProgress belongs to a `Leadership`, which creates a fresh one
    for each follower whenever the node wins an election and passes on
    only replies to RPCs sent in that election's term — so every reply
    that reaches this record belongs to the current leadership.

    Attributes:
        next_index: The index of the next log entry the Leader will send
            to this follower. Always greater than `match_index`.
        match_index: The highest log index confirmed to match on this
            follower (0 if none yet). Never decreases.
    """

    def __init__(self, next_index: int) -> None:
        """Start tracking a follower, with nothing confirmed yet.

        `match_index` always starts at 0 (REPL-15), because nothing has
        been confirmed yet. A `Leadership` starts `next_index` at one past
        the Leader's own last log index (REPL-14) — 11, for a Leader
        holding 10 entries — optimistically assuming the follower already
        has everything, and letting rejections walk it back.

        Args:
            next_index: The nextIndex to start tracking this follower
                from. At least 1.
        """
        self._next_index = next_index
        self._match_index = 0

    @property
    def next_index(self) -> int:
        return self._next_index

    @property
    def match_index(self) -> int:
        return self._match_index

    def record_success(self, prev_log_index: int, entry_count: int) -> None:
        """Record that this follower accepted an AppendEntries RPC.

        The accepted RPC proves the follower's log now matches the
        Leader's through the last entry that RPC carried:
        `prev_log_index + entry_count`, both taken from that specific
        RPC (REPL-16). `match_index` moves up to that index, and
        `next_index` moves to the index right after it.

        Neither value ever moves down here (REPL-17). Responses can arrive
        late or more than once — FAIL-2 retries an RPC that timed out, and
        the network may deliver replies out of order — so a success for an
        older RPC that carried fewer entries can arrive after a newer
        one. Taking the larger value keeps `match_index` from falling
        back to something already surpassed, so it only ever increases.

        Args:
            prev_log_index: The `prev_log_index` the accepted RPC was
                sent with.
            entry_count: How many entries the accepted RPC carried — 0
                for a heartbeat.
        """
        self._match_index = max(self._match_index, prev_log_index + entry_count)
        self._next_index = max(self._next_index, self._match_index + 1)

    def record_rejection(self) -> None:
        """Lower nextIndex by one after this follower rejects an AppendEntries.

        Implements REPL-6: "whenever a Leader's AppendEntries RPC is
        rejected under REPL-5, the Leader shall decrement its stored
        nextIndex for that Follower." It never goes below
        `match_index + 1`: every index up to `match_index` is already
        confirmed to match, so a rejection reaching below it can only be
        a late reply to an older RPC, and probing there again would only
        resend entries the follower already holds. With nothing confirmed
        yet, that floor is 1 — the lowest index a probe ever needs,
        since `Log.matches` accepts a `prev_log_index` of 0
        unconditionally.

        The retrying itself — sending AppendEntries again from the new,
        lower `next_index` — is REPL-7's job, and the caller's.
        """
        self._next_index = max(self._match_index + 1, self._next_index - 1)
