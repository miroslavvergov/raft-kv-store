"""A Leader's per-Follower replication progress: nextIndex, matchIndex, and what it has answered."""

from raftkv.consensus.append_entries import AppendEntriesRequest


class FollowerProgress:
    """A Leader's in-memory replication progress for one Follower (DD-25).

    `next_index` is a guess, lowered on each rejection of the probe now
    outstanding (REPL-6, DD-27) until the logs agree (REPL-5). `match_index` is
    never a guess: it rises only on an acknowledged success (REPL-16), which is
    what counts the Follower toward an entry's commit majority (APPLY-1). Both
    change only through `record_success` and `record_rejection`, which keep
    `match_index < next_index`. It holds no term: its `Leadership` passes on only
    replies to RPCs sent in its own term.

    It also remembers the request built last, with its number, until the
    Follower answers it, and the number of the newest request the Follower has
    answered, so that `Leadership` can tell which requests a read's confirmation
    may count (CLIENT-8, DD-34).

    Attributes:
        next_index: The index of the next entry to send; always above
            `match_index`.
        match_index: The highest index known to match on the Follower; 0 until
            one is confirmed. Never decreases (REPL-17).
        answered_request: The number of the newest request the Follower has
            answered in this term; 0 until one is.
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
        self._awaiting_answer: AppendEntriesRequest | None = None
        self._awaiting_number = 0
        self._answered_request = 0

    @property
    def next_index(self) -> int:
        return self._next_index

    @property
    def match_index(self) -> int:
        return self._match_index

    @property
    def answered_request(self) -> int:
        return self._answered_request

    def record_built(self, request: AppendEntriesRequest, request_number: int) -> None:
        """Remember `request`, the one now awaiting an answer, and its number.

        Args:
            request: The AppendEntries just built for this Follower.
            request_number: Its number among every request the leadership has built.
        """
        self._awaiting_answer = request
        self._awaiting_number = request_number

    def record_reply(self, request: AppendEntriesRequest) -> None:
        """Record that the Follower answered `request`, if it is the one now awaiting an answer.

        Only the request built last counts: an answer to an older one, or a second
        answer to the same one, changes nothing. The request is matched by identity,
        not equality: two heartbeats to a caught-up Follower are equal in every
        field, so equality could not tell an answer to the older from one to the newer.

        Args:
            request: The AppendEntries the Follower answered: the very object given to
                `record_built`. An equal copy is not recognized.
        """
        if request is not self._awaiting_answer:
            return
        self._answered_request = self._awaiting_number
        self._awaiting_answer = None

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

    def record_rejection(self, prev_log_index: int) -> bool:
        """Lower `next_index` by one if the rejected AppendEntries is the probe now outstanding.

        The outstanding probe is the one sent from `next_index`, so its
        `prev_log_index` is `next_index - 1`. A rejection of any other probe, a
        duplicate or a late reply to one already backed off from, changes nothing
        (FAIL-1, DD-27), so each round of repair lowers `next_index` exactly once
        (REPL-6). Never below `match_index + 1`: indexes up to `match_index` are
        confirmed. With nothing confirmed the floor is 1, since `Log.matches`
        accepts `prev_log_index` 0. Resending from the new `next_index` (REPL-7) is
        the caller's job.

        Args:
            prev_log_index: The rejected RPC's `prev_log_index`.

        Returns:
            True if `next_index` was lowered, so a resend reaches one entry further
            back; False if nothing changed.
        """
        if prev_log_index != self._next_index - 1 or self._next_index <= self._match_index + 1:
            return False
        self._next_index -= 1
        return True
