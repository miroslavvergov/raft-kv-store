"""A Leader's per-Follower replication progress: nextIndex and matchIndex."""


class FollowerProgress:
    """A Leader's in-memory replication progress for one Follower (DD-25).

    `next_index` is a guess, lowered on each rejection (REPL-6) until the logs
    agree (REPL-5). `match_index` is never a guess: it rises only on an
    acknowledged success (REPL-16), which is what counts the Follower toward an
    entry's commit majority (APPLY-1). Both change only through
    `record_success` and `record_rejection`, which keep
    `match_index < next_index`. It holds no term: its `Leadership` passes on
    only replies to RPCs sent in its own term.

    Attributes:
        next_index: The index of the next entry to send; always above
            `match_index`.
        match_index: The highest index known to match on the Follower; 0 until
            one is confirmed. Never decreases (REPL-17).
    """

    def __init__(self, next_index: int) -> None:
        """Start tracking a Follower, with `match_index` 0 (REPL-15).

        Args:
            next_index: The first index to send; at least 1. `Leadership` passes
                one past the Leader's last log index (REPL-14), assuming the
                Follower has everything until rejections walk it back.
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
        """Record that the Follower accepted an AppendEntries.

        The accepted RPC proves the logs match through
        `prev_log_index + entry_count` (REPL-16): `match_index` rises to that index
        and `next_index` to at least `match_index + 1`. A success never lowers
        either, so a late, duplicated, or reordered reply (FAIL-2) cannot move them
        back; `match_index` never decreases at all (REPL-17).

        Args:
            prev_log_index: The accepted RPC's `prev_log_index`.
            entry_count: How many entries it carried; 0 for a heartbeat.
        """
        self._match_index = max(self._match_index, prev_log_index + entry_count)
        self._next_index = max(self._next_index, self._match_index + 1)

    def record_rejection(self) -> None:
        """Lower `next_index` by one after the Follower rejects an AppendEntries (REPL-6).

        Never below `match_index + 1`: indexes up to `match_index` are confirmed, so
        a rejection reaching below is a late reply to an older RPC. With nothing
        confirmed the floor is 1, the lowest probe needed, since `Log.matches`
        accepts `prev_log_index` 0. Resending from the new `next_index` (REPL-7) is
        the caller's job.
        """
        self._next_index = max(self._match_index + 1, self._next_index - 1)
