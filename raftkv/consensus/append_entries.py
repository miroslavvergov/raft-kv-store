"""The AppendEntries RPC: what a Leader sends, and what a Follower answers."""

from dataclasses import dataclass

from raftkv.consensus.log import LogEntry


@dataclass(frozen=True)
class AppendEntriesRequest:
    """A Leader's AppendEntries to one Follower: entries to store, and how far it may commit.

    Carries the position of the entry before `entries` (REPL-3), which is what
    REPL-5's check compares, and the Leader's own commit index (REPL-4), which is
    what REPL-13 caps. Empty `entries` is the heartbeat REPL-9 sends; it still
    carries both, so a heartbeat both confirms agreement and advances commitment.

    Attributes:
        term: The Leader's `current_term`.
        leader_id: The Leader's node ID.
        prev_log_index: The 1-based index of the entry preceding `entries`; 0 when
            they start the log.
        prev_log_term: The term of that entry; 0 when `prev_log_index` is 0.
        entries: The entries to store, in order, from `prev_log_index + 1`; empty
            for a heartbeat. Stored as a tuple, so a list passed in cannot be
            changed afterwards through the caller's reference.
        leader_commit: The Leader's `commit_index` when it sent this RPC.

    Raises:
        ValueError: If `term` is below 1; any index or term is negative; exactly
            one of `prev_log_index` and `prev_log_term` is 0; or the entries'
            terms fall, start below `prev_log_term`, or exceed `term`. A negative
            `prev_log_index` would otherwise be read as an offset from the log's
            end, and falling terms would break every check that reads a log's
            highest term from one index.
    """

    term: int
    leader_id: int
    prev_log_index: int = 0
    prev_log_term: int = 0
    entries: tuple[LogEntry, ...] = ()
    leader_commit: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(self.entries))
        if self.term < 1:
            raise ValueError(f"term must be at least 1, got {self.term}")
        # NOTE: 0 is legal for both, meaning "no entry precedes these", which is how an empty
        # log accepts its first entries; only a negative value is rejected.
        if self.prev_log_index < 0 or self.prev_log_term < 0:
            raise ValueError(
                f"prev_log_index and prev_log_term cannot be negative, got "
                f"{self.prev_log_index} and {self.prev_log_term}"
            )
        if (self.prev_log_index == 0) != (self.prev_log_term == 0):
            raise ValueError(
                f"prev_log_term is 0 exactly when prev_log_index is, got "
                f"{self.prev_log_index} and {self.prev_log_term}"
            )
        if self.leader_commit < 0:
            raise ValueError(f"leader_commit cannot be negative, got {self.leader_commit}")
        terms = [self.prev_log_term, *(entry.term for entry in self.entries), self.term]
        if terms != sorted(terms):
            raise ValueError(
                f"entry terms must not fall and must lie from prev_log_term to term, got {terms}"
            )

    @property
    def last_new_index(self) -> int:
        """The index of the last entry this RPC covers; `prev_log_index` for a heartbeat."""
        return self.prev_log_index + len(self.entries)

    def commit_index_after(self, commit_index: int) -> int:
        """Return a Follower's commit index after it accepts this RPC (REPL-13).

        The Leader's commit index, capped at `last_new_index`: a Follower commits
        only entries it holds, and `leader_commit` can already name entries a later
        RPC will bring. Never below `commit_index`, so a delayed RPC covering fewer
        entries cannot uncommit what is committed.

        Args:
            commit_index: The Follower's commit index before this RPC.
        """
        return max(commit_index, min(self.leader_commit, self.last_new_index))


@dataclass(frozen=True)
class AppendEntriesResponse:
    """A Follower's answer to one AppendEntriesRequest.

    Carries no index: a Leader advances `match_index` from the request it sent,
    never from the answer (REPL-16).

    Attributes:
        term: The Follower's `current_term` after handling the RPC. A Leader that
            sees a higher term catches up and steps down (STATE-4, STATE-5).
        success: Whether the Follower stored the entries. False means its log
            failed REPL-5's check at `prev_log_index`, the RPC's term was behind
            its own, or it is itself Leader of that term, which a correct cluster
            never produces.
    """

    term: int
    success: bool
