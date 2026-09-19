"""One election win: the Leader's term and its progress with every Follower."""

from typing import Iterable

from raftkv.consensus.follower_progress import FollowerProgress


class Leadership:
    """Everything a Leader tracks about its Followers for one term of leadership.

    A node creates a Leadership when it wins an election and discards it
    when it stops being Leader; winning again later creates a new one. It
    holds the Leader-level fact that applies to all Followers at once —
    the term this leadership was won in — and one `FollowerProgress` per
    Follower (DD-25).

    Creating it is what resets replication progress on every election.
    Each Follower starts at `next_index` one past the Leader's own last
    log index (REPL-14), optimistically assuming the Follower already has
    everything and letting rejections walk it back, and at `match_index`
    0 (REPL-15), since nothing has been confirmed in this term yet.
    Nothing carries over from an earlier leadership of the same node:
    another Leader may have rewritten the Followers' logs, and this
    node's own log, in between.

    Replies are counted only if they answer an AppendEntries sent in this
    leadership's term (REPL-16). A reply can be delayed in the network
    for a long time. By the time a reply to a term-5 request arrives,
    this node may have lost leadership, had its own entries and the
    Follower's overwritten by another Leader, and won again in term 7
    with different entries at the same positions. That reply describes a
    log that no longer exists, and counting it could make the Leader
    believe an entry is stored on a majority of nodes when it is not. The
    term compared is the one the RPC was sent in, not the term written on
    the reply: a request delayed until the Follower has already moved on
    to the Leader's newer term comes back rejected with that newer term
    on it, and would otherwise look current. Every other term is ignored,
    not only earlier ones.

    The Follower records never leave this object. The only way to change
    one is `record_success` or `record_rejection`, which check the term
    first; everything else sees them only through the `next_index` and
    `match_index` lookups. A stale reply therefore cannot be counted,
    even by a caller that doesn't know it could be stale.

    Separately, a reply carrying a term higher than this leadership's
    means the node has been replaced as Leader and must step down. That
    concerns the node's role and term, and is handled by
    `NodeState.handle_observed_term`, not here.

    Attributes:
        term: The term in which this node won the election.
        followers: The IDs of the nodes this Leader replicates to.
    """

    def __init__(self, term: int, followers: Iterable[int], last_log_index: int) -> None:
        """Start a leadership, with every Follower's progress reset.

        Args:
            term: The term in which this node won the election.
            followers: The IDs of every other node in the cluster, not
                including this node itself.
            last_log_index: The index of the last entry in this node's
                own log at the moment it won (0 for an empty log).
        """
        self._term = term
        self._progress = {
            follower: FollowerProgress(next_index=last_log_index + 1)
            for follower in followers
        }

    @property
    def term(self) -> int:
        return self._term

    @property
    def followers(self) -> frozenset[int]:
        return frozenset(self._progress)

    def next_index(self, follower: int) -> int:
        """Return the index of the next log entry to send to a Follower.

        Args:
            follower: The Follower's node ID.

        Returns:
            That Follower's current `next_index`.

        Raises:
            KeyError: If `follower` is not one of this leadership's
                Followers.
        """
        return self._progress[follower].next_index

    def match_index(self, follower: int) -> int:
        """Return the highest log index confirmed to match on a Follower.

        Args:
            follower: The Follower's node ID.

        Returns:
            That Follower's current `match_index` (0 if nothing has been
            confirmed in this leadership yet).

        Raises:
            KeyError: If `follower` is not one of this leadership's
                Followers.
        """
        return self._progress[follower].match_index

    def record_success(
        self, follower: int, sent_in_term: int, prev_log_index: int, entry_count: int
    ) -> bool:
        """Record that a Follower accepted an AppendEntries — if it belongs to this leadership.

        If `sent_in_term` is not this leadership's term, the reply is
        ignored: no Follower is looked up and nothing changes (REPL-16).
        Otherwise the success is recorded on that Follower's progress:
        `match_index` rises to `prev_log_index + entry_count` unless it is
        already higher (REPL-16, REPL-17), and `next_index` moves to just
        past it. Every other Follower is untouched.

        Args:
            follower: The node ID of the Follower that replied.
            sent_in_term: The term the accepted RPC was sent in.
            prev_log_index: The `prev_log_index` the accepted RPC was
                sent with.
            entry_count: How many entries the accepted RPC carried — 0
                for a heartbeat.

        Returns:
            True if the reply was counted, False if it was ignored
            because the RPC was sent in a different term.

        Raises:
            KeyError: If the RPC was sent in this term but `follower` is
                not one of this leadership's Followers.
        """
        if sent_in_term != self._term:
            return False
        self._progress[follower].record_success(prev_log_index, entry_count)
        return True

    def record_rejection(self, follower: int, sent_in_term: int) -> bool:
        """Record that a Follower rejected an AppendEntries — if it belongs to this leadership.

        If `sent_in_term` is not this leadership's term, the reply is
        ignored: no Follower is looked up and nothing changes (REPL-16).
        Otherwise that Follower's `next_index` drops by one (REPL-6), but
        never to or below its `match_index`. Every other Follower is
        untouched. Sending AppendEntries again from the lower
        `next_index` (REPL-7) is the caller's job.

        Args:
            follower: The node ID of the Follower that replied.
            sent_in_term: The term the rejected RPC was sent in.

        Returns:
            True if the reply was counted, False if it was ignored
            because the RPC was sent in a different term.

        Raises:
            KeyError: If the RPC was sent in this term but `follower` is
                not one of this leadership's Followers.
        """
        if sent_in_term != self._term:
            return False
        self._progress[follower].record_rejection()
        return True
