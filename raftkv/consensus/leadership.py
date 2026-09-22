"""One election win: the Leader's term and its progress with every Follower."""

from collections.abc import Iterable

from raftkv.consensus.append_entries import AppendEntriesRequest
from raftkv.consensus.follower_progress import FollowerProgress
from raftkv.consensus.log import Log


class Leadership:
    """A Leader's progress with each Follower for one term of leadership (DD-25).

    Created on each election win, discarded on leaving the Leader role; nothing
    carries over from an earlier leadership, since another Leader may have
    rewritten the logs in between. Each Follower starts at `next_index` one past
    the Leader's last log index (REPL-14) and `match_index` 0 (REPL-15).

    A reply counts only if its RPC was sent in this leadership's term (REPL-16, DD-25):
    an older reply describes logs that may since have been overwritten and could
    fake a commit majority. The send term is compared, not the reply's term,
    because a delayed request can come back rejected with the current term on
    it. The Follower records are private, and only `record_success` and
    `record_rejection` change them, after that check, so no caller can count a
    stale reply. The caller handles a higher term in a reply with
    `NodeState.handle_observed_term`.

    Attributes:
        term: The term in which this node won the election.
        followers: The IDs of the nodes this Leader replicates to.
    """

    def __init__(self, term: int, followers: Iterable[int], last_log_index: int) -> None:
        """Start a leadership with every Follower's progress reset.

        Args:
            term: The term in which this node won the election.
            followers: The IDs of every other cluster member, not this node.
            last_log_index: This node's last log index when it won; 0 if empty.
        """
        self._term = term
        self._progress = {
            follower: FollowerProgress(next_index=last_log_index + 1) for follower in followers
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

        Raises:
            KeyError: If `follower` is not one of this leadership's Followers.
        """
        return self._progress[follower].next_index

    def match_index(self, follower: int) -> int:
        """Return the highest log index confirmed to match on a Follower; 0 if none yet.

        Args:
            follower: The Follower's node ID.

        Raises:
            KeyError: If `follower` is not one of this leadership's Followers.
        """
        return self._progress[follower].match_index

    def record_success(
        self, follower: int, sent_in_term: int, prev_log_index: int, entry_count: int
    ) -> bool:
        """Record a Follower's AppendEntries success, if the RPC was sent in this term.

        A reply to an RPC from any other term is ignored before the Follower is
        looked up (REPL-16). Otherwise only that Follower changes: `match_index`
        rises to `prev_log_index + entry_count` unless already higher (REPL-17),
        and `next_index` to at least `match_index + 1`.

        Args:
            follower: The ID of the Follower that replied.
            sent_in_term: The term the accepted RPC was sent in.
            prev_log_index: The accepted RPC's `prev_log_index`.
            entry_count: How many entries it carried; 0 for a heartbeat.

        Returns:
            True if counted; False if ignored as sent in another term.

        Raises:
            KeyError: If `sent_in_term` is this term but `follower` is not one of
                this leadership's Followers.
        """
        if sent_in_term != self._term:
            return False
        self._progress[follower].record_success(prev_log_index, entry_count)
        return True

    def record_rejection(self, follower: int, sent_in_term: int) -> bool:
        """Record a Follower's AppendEntries rejection, if the RPC was sent in this term.

        A reply to an RPC from any other term is ignored before the Follower is
        looked up (DD-25). Otherwise only that Follower changes: its `next_index`
        drops by one (REPL-6), never to `match_index` or below. Resending from the
        lower `next_index` (REPL-7) is the caller's job.

        Args:
            follower: The ID of the Follower that replied.
            sent_in_term: The term the rejected RPC was sent in.

        Returns:
            True if counted; False if ignored as sent in another term.

        Raises:
            KeyError: If `sent_in_term` is this term but `follower` is not one of
                this leadership's Followers.
        """
        if sent_in_term != self._term:
            return False
        self._progress[follower].record_rejection()
        return True

    def append_entries_request_for(
        self, follower: int, log: Log, leader_id: int, commit_index: int
    ) -> AppendEntriesRequest:
        """Return the AppendEntries to send a Follower now (REPL-2, REPL-3, REPL-4).

        It carries every entry from the Follower's `next_index` to the end of `log`,
        with `prev_log_index` and `prev_log_term` naming the entry just before
        `next_index`, so an accepted RPC leaves the Follower's log matching this one
        from `next_index` on, and a caught-up Follower gets a heartbeat. Built in this
        leadership's term, which is what an answer is later checked against (REPL-16,
        DD-25).

        Args:
            follower: The Follower to send to.
            log: The Leader's log.
            leader_id: The Leader's node ID.
            commit_index: The Leader's commit index.

        Raises:
            KeyError: If `follower` is not one of this leadership's Followers.
        """
        next_index = self._progress[follower].next_index
        return AppendEntriesRequest(
            term=self._term,
            leader_id=leader_id,
            prev_log_index=next_index - 1,
            prev_log_term=log.term_at(next_index - 1),
            entries=log.entries_from(next_index),
            leader_commit=commit_index,
        )

    def commit_index_after(self, commit_index: int, log: Log, majority: int) -> int:
        """Return the Leader's commit index given what its Followers have confirmed.

        The highest index on at least `majority` nodes, the Leader counting with
        its whole log (APPLY-1), is committed only if its entry is from this
        leadership's term (APPLY-2). An earlier term's entry on a majority can
        still be overwritten by a later Leader, so it commits only when a
        current-term entry after it does (APPLY-3). Never below `commit_index`.

        Args:
            commit_index: The Leader's commit index so far.
            log: The Leader's log.
            majority: How many members make a strict majority (`Cluster.majority`).
        """
        held = sorted(
            [log.last_index, *(progress.match_index for progress in self._progress.values())],
            reverse=True,
        )
        # NOTE: the `majority`-th highest index is on at least `majority` nodes, and no higher
        # index is. Terms never fall along a log, so if this entry is from an earlier term,
        # every entry before it is too, and nothing new can commit.
        on_a_majority = held[majority - 1]
        if on_a_majority <= commit_index or log.term_at(on_a_majority) != self._term:
            return commit_index
        return on_a_majority
