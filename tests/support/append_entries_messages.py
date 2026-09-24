"""AppendEntries messages, built by keyword, shared by the consensus and node tests.

`LEADER_ID` and `LEADER_TERM` are the Leader the divergent-log fixtures describe: it holds
`LEADER_TERMS` and leads a term above every entry term in those logs, so its entries may
overwrite any of them.
"""

from raftkv.consensus import AppendEntriesRequest, AppendEntriesResponse

LEADER_ID = 8
LEADER_TERM = 8


def accepted(*, term):
    """Return a Follower's answer accepting an AppendEntries, carrying `term`."""
    return AppendEntriesResponse(term=term, success=True)


def rejected(*, term):
    """Return a Follower's answer rejecting an AppendEntries, carrying `term`."""
    return AppendEntriesResponse(term=term, success=False)


def append_entries(
    *,
    term=LEADER_TERM,
    leader=LEADER_ID,
    prev_log_index=0,
    prev_log_term=0,
    entries=(),
    leader_commit=0,
):
    """Return a Leader's AppendEntries; it starts an empty log and commits nothing by default."""
    return AppendEntriesRequest(
        term=term,
        leader_id=leader,
        prev_log_index=prev_log_index,
        prev_log_term=prev_log_term,
        entries=entries,
        leader_commit=leader_commit,
    )


def heartbeat(*, term=LEADER_TERM, leader=LEADER_ID, prev_log_index=0, prev_log_term=0, commit=0):
    """Return the entry-less AppendEntries REPL-9 sends at a fixed interval."""
    return append_entries(
        term=term,
        leader=leader,
        prev_log_index=prev_log_index,
        prev_log_term=prev_log_term,
        leader_commit=commit,
    )


def append_entries_at(leader_log, next_index, *, term=LEADER_TERM, leader=LEADER_ID, commit=0):
    """Return the AppendEntries a Leader holding `leader_log` sends a Follower at `next_index`.

    Args:
        leader_log: The Leader's own log.
        next_index: The Leader's `next_index` for that Follower.
        term: The term the Leader leads.
        leader: The Leader's node ID.
        commit: The Leader's commit index.
    """
    prev_log_index = next_index - 1
    return append_entries(
        term=term,
        leader=leader,
        prev_log_index=prev_log_index,
        prev_log_term=leader_log.term_at(prev_log_index),
        entries=leader_log.entries_from(next_index),
        leader_commit=commit,
    )
