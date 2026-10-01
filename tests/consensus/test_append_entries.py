"""Tier 1 tests for the AppendEntries messages: their fields, and REPL-13's commit rule.

Pure: no disk, no node. REPL-3 (the previous entry travels with the RPC), REPL-4 (so does the
Leader's commit index), REPL-13 (what a Follower does with it), DD-34 (it carries nothing more
for a read).
"""

import dataclasses

import pytest

from raftkv.consensus import AppendEntriesRequest, LogEntry
from tests.support.append_entries_messages import append_entries, heartbeat


def entries(*terms):
    """Return one entry per term, commanded by position, as a Leader would send them."""
    return [LogEntry(term=term, command=f"cmd{i + 1}") for i, term in enumerate(terms)]


# --- The request's fields (REPL-3, REPL-4) --------------------------------------------


def test_a_request_carries_only_what_replication_needs_and_nothing_for_a_read():
    # A Leader confirms a read by pairing each answer with the request it answers, so the RPC
    # needs no field of its own for that (DD-34).
    names = [field.name for field in dataclasses.fields(AppendEntriesRequest)]

    assert names == [
        "term",
        "leader_id",
        "prev_log_index",
        "prev_log_term",
        "entries",
        "leader_commit",
    ]


def test_a_request_is_immutable_and_keeps_its_own_copy_of_the_entries():
    sent = entries(4, 4)
    request = append_entries(entries=sent)
    sent.append(LogEntry(term=4, command="sneaked in"))  # the sender changes its list afterwards

    assert request.entries == tuple(entries(4, 4))
    assert request == append_entries(entries=tuple(entries(4, 4)))  # a tuple or a list alike
    with pytest.raises(AttributeError):
        request.term = 99


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("term", 0),
        ("term", -1),
        # A negative previous index would read the log from its end instead of rejecting.
        ("prev_log_index", -1),
        ("prev_log_term", -1),
        ("leader_commit", -1),
    ],
)
def test_a_request_with_an_impossible_index_or_term_is_refused(field, value):
    with pytest.raises(ValueError, match=field.split("_")[0]):
        append_entries(**{field: value})


@pytest.mark.parametrize(
    ("options", "why"),
    [
        ({"prev_log_index": 0, "prev_log_term": 3}, "no entry precedes the first, so no term"),
        ({"prev_log_index": 3, "prev_log_term": 0}, "every real entry has a term of at least 1"),
        ({"prev_log_index": 1, "prev_log_term": 2, "entries": entries(1)}, "terms never fall"),
        ({"term": 2, "entries": [LogEntry(3, "x")]}, "no entry is newer than its Leader"),
        ({"entries": [LogEntry(2, "a"), LogEntry(1, "b")]}, "terms never fall within the entries"),
        ({"entries": [LogEntry(1, "a", 5), LogEntry(1, "b", 4)]}, "cluster time never falls"),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_a_request_no_correct_leader_could_send_is_refused(options, why):
    with pytest.raises(ValueError):
        append_entries(**options)


def test_entries_may_share_a_cluster_time():
    # Entries appended within one tick carry the same time: it never falls, but need not rise.
    request = append_entries(entries=[LogEntry(1, "a", 5), LogEntry(1, "b", 5)])
    assert [entry.cluster_time for entry in request.entries] == [5, 5]


def test_zero_is_legal_for_the_previous_entry_because_no_entry_precedes_the_first():
    request = append_entries(prev_log_index=0, prev_log_term=0, entries=entries(1))
    assert (request.prev_log_index, request.prev_log_term) == (0, 0)


@pytest.mark.parametrize(
    ("prev_log_index", "entry_terms", "expected"),
    [
        (0, [], 0),  # an empty log's heartbeat covers nothing
        (0, [1, 1], 2),  # the first two entries
        (7, [], 7),  # a heartbeat covers everything through the previous entry
        (7, [8, 8, 8], 10),
    ],
)
def test_last_new_index_is_the_previous_index_plus_the_entries_carried(
    prev_log_index, entry_terms, expected
):
    request = append_entries(
        prev_log_index=prev_log_index,
        prev_log_term=1 if prev_log_index else 0,
        entries=entries(*entry_terms),
    )
    assert request.last_new_index == expected


# --- REPL-13: what a Follower commits after accepting ---------------------------------


@pytest.mark.parametrize(
    ("leader_commit", "expected", "why"),
    [
        (0, 0, "a Leader that has committed nothing commits nothing here"),
        (3, 3, "below the last entry carried, so the Leader's value is taken as is"),
        (5, 5, "exactly the last entry carried"),
        (9, 5, "above the last entry carried, so it is capped: the rest has not arrived"),
    ],
)
def test_commit_index_is_the_leaders_capped_at_the_last_entry_carried(leader_commit, expected, why):
    # prev_log_index 2 plus three entries: this RPC covers the log through index 5.
    request = append_entries(
        prev_log_index=2, prev_log_term=1, entries=entries(4, 4, 4), leader_commit=leader_commit
    )
    assert request.commit_index_after(0) == expected, why


def test_a_heartbeat_commits_through_its_previous_entry():
    # No entries, but the match at index 4 proves the Follower holds everything through it.
    assert heartbeat(prev_log_index=4, prev_log_term=2, commit=4).commit_index_after(0) == 4


def test_a_heartbeat_is_still_capped_by_what_the_follower_holds():
    assert heartbeat(prev_log_index=4, prev_log_term=2, commit=9).commit_index_after(0) == 4


def test_a_delayed_rpc_covering_fewer_entries_never_uncommits_what_is_committed():
    # An old RPC arrives after the Follower already committed through index 7. Its cap is 2, but
    # entries committed by a majority can never be taken back.
    stale = heartbeat(prev_log_index=2, prev_log_term=1, commit=9)
    assert stale.commit_index_after(7) == 7


def test_commit_index_never_moves_backwards_over_a_sequence_of_rpcs():
    committed = 0
    for request in [
        append_entries(prev_log_index=0, prev_log_term=0, entries=entries(1, 1), leader_commit=2),
        append_entries(prev_log_index=2, prev_log_term=1, entries=entries(1), leader_commit=3),
        heartbeat(prev_log_index=1, prev_log_term=1, commit=3),  # delayed, covers only index 1
        heartbeat(prev_log_index=3, prev_log_term=1, commit=3),
    ]:
        advanced = request.commit_index_after(committed)
        assert advanced >= committed
        committed = advanced
    assert committed == 3
