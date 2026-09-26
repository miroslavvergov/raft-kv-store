"""Tier 1 tests for the Leader's commit rule: Leadership.commit_index_after.

APPLY-1 (an entry on a majority is a candidate), APPLY-2 (only a current-term entry commits by
counting), APPLY-3 (an earlier term's entry commits only with a later current-term one).
"""

import random

import pytest

from raftkv.consensus import Cluster, Leadership, Log, LogEntry
from tests.support.divergent_logs import make_log

LEADER = 1


def leadership(term, cluster, log, *, confirmed=None):
    """Return `LEADER`'s Leadership with each Follower confirmed through `confirmed[follower]`.

    Args:
        term: The term the Leader leads.
        cluster: The cluster; every member but `LEADER` is a Follower.
        log: The Leader's log.
        confirmed: The index each Follower has acknowledged, by Follower; 0 if absent.
    """
    lead = Leadership(term, cluster.peers_of(LEADER), log.last_index, cluster_time=0)
    for follower, index in (confirmed or {}).items():
        assert lead.record_success(follower, term, prev_log_index=0, entry_count=index)
    return lead


def committed(term, cluster, log, *, confirmed=None, commit_index=0):
    """Return the Leader's commit index after the Followers confirmed what `confirmed` says."""
    lead = leadership(term, cluster, log, confirmed=confirmed)
    return lead.commit_index_after(commit_index, log, cluster.majority)


THREE = Cluster([1, 2, 3])
FOUR = Cluster([1, 2, 3, 4])
FIVE = Cluster([1, 2, 3, 4, 5])


# --- APPLY-1: an entry on a majority, counting the Leader's own copy -------------------


def test_nothing_commits_before_any_follower_confirms():
    # The Leader alone holds its entries: 1 of 3 is not a majority.
    assert committed(1, THREE, make_log([1, 1])) == 0


def test_one_follower_confirming_makes_a_majority_of_three():
    # The Leader's own copy counts: with one Follower's, that is 2 of 3 nodes.
    assert committed(1, THREE, make_log([1, 1]), confirmed={2: 2}) == 2


@pytest.mark.parametrize(
    ("cluster", "confirmed", "expected", "why"),
    [
        (FIVE, {2: 3}, 0, "Leader and one Follower: 2 of 5"),
        (FIVE, {2: 3, 3: 3}, 3, "Leader and two Followers: 3 of 5"),
        (FOUR, {2: 3}, 0, "Leader and one Follower: 2 of 4 is half, not a majority"),
        (FOUR, {2: 3, 3: 3}, 3, "Leader and two Followers: 3 of 4"),
    ],
)
def test_a_majority_is_strictly_more_than_half(cluster, confirmed, expected, why):
    assert committed(1, cluster, make_log([1, 1, 1]), confirmed=confirmed) == expected, why


def test_a_single_node_cluster_commits_the_leaders_entries_at_once():
    assert committed(1, Cluster([1]), make_log([1, 1, 1])) == 3


def test_it_commits_the_highest_index_a_majority_holds_not_the_highest_anyone_holds():
    # Follower 2 has everything; Follower 3 only the first two; Follower 4 and 5 nothing.
    # Index 2 is on 1, 2, 3 (a majority of 5); index 4 is only on 1 and 2.
    assert committed(1, FIVE, make_log([1, 1, 1, 1]), confirmed={2: 4, 3: 2}) == 2


def test_the_commit_index_never_goes_down():
    # Already committed through 3; the confirmations now shown only reach 2.
    assert committed(1, THREE, make_log([1, 1, 1]), confirmed={2: 2}, commit_index=3) == 3


# --- APPLY-2 and APPLY-3: only the Leader's own term commits by counting ---------------


def test_an_earlier_terms_entry_on_a_majority_is_not_committed():
    # Leading term 4, the Leader holds entries from terms 1 and 2. Follower 2 has both, so
    # index 2 is on a majority of three, but it is from term 2, not term 4.
    assert committed(4, THREE, make_log([1, 2]), confirmed={2: 2}) == 0


def test_earlier_entries_commit_together_with_a_current_term_entry_after_them():
    # The Leader's own term-4 entry reaches a majority, and everything before it commits too.
    assert committed(4, THREE, make_log([1, 2, 4]), confirmed={2: 3}) == 3


def test_the_current_term_entry_must_itself_be_on_the_majority():
    # Follower 2 has the old entries but not the term-4 one: still nothing commits.
    assert committed(4, THREE, make_log([1, 2, 4]), confirmed={2: 2}) == 0


def test_a_new_leaders_empty_entry_is_what_lets_earlier_entries_commit():
    # With no client command, the empty entry is the only current-term entry there is.
    log = Log([*make_log([1, 2]), LogEntry.empty(4)])
    assert committed(4, THREE, log, confirmed={2: 2}) == 0
    assert committed(4, THREE, log, confirmed={2: 3}) == 3


# --- Why APPLY-2 exists: an earlier term's entry on a majority can still be erased -----
#
# Five nodes. Node 1 led term 2 and sent its entry 2 only to node 2. Node 5 led term 3 and
# wrote a different entry 2 only to itself. Node 1, leading again in term 4, now copies its
# term-2 entry 2 to node 3: that entry is on nodes 1, 2 and 3, a majority.

LEADER_TERM = 4
LOGS_AT_TERM_4 = {
    1: make_log([1, 2]),
    2: make_log([1, 2]),
    3: make_log([1, 2]),  # just received it from node 1
    4: make_log([1]),
    5: Log([LogEntry(1, "cmd1"), LogEntry(3, "node 5's entry")]),
}


def node_5_wins_term_5_and_repairs_node_3(logs):
    """Replay what happens if node 1 then crashes: node 5 runs for term 5.

    Returns:
        The voters that grant node 5 their vote, and node 3's log after node 5's AppendEntries.
    """
    candidate = logs[5].last_position
    voters = {n for n in (2, 3, 4) if candidate.is_at_least_as_up_to_date_as(logs[n].last_position)}
    # Node 5 repairs node 3: they agree at index 1, and node 5 sends its own entry 2.
    repaired = logs[3].after_append_entries(1, [logs[5].entry_at(2)])
    return voters, repaired


def test_an_earlier_terms_entry_on_a_majority_can_still_be_erased():
    voters, node_3_after = node_5_wins_term_5_and_repairs_node_3(LOGS_AT_TERM_4)
    # Node 5's last entry (term 3) beats their last entries (term 2 or 1): all three grant, so
    # node 5 wins with 4 of 5, and its entry 2 replaces the term-2 entry on node 3.
    assert voters == {2, 3, 4}
    assert FIVE.is_majority(voters | {5})
    assert node_3_after.entry_at(2).command == "node 5's entry"


def test_a_leader_does_not_commit_an_earlier_terms_entry_that_could_still_be_erased():
    assert committed(LEADER_TERM, FIVE, LOGS_AT_TERM_4[1], confirmed={2: 2, 3: 2}) == 0


def test_once_the_leaders_own_entry_is_on_a_majority_node_5_can_no_longer_win():
    # Node 1 appends its term-4 empty entry and it reaches nodes 2 and 3 too: now it commits,
    # and node 5's term-3 log is behind theirs, so it cannot collect a majority.
    logs = dict(LOGS_AT_TERM_4)
    for n in (1, 2, 3):
        logs[n] = Log([*logs[n], LogEntry.empty(LEADER_TERM)])
    assert committed(LEADER_TERM, FIVE, logs[1], confirmed={2: 3, 3: 3}) == 3
    voters, _ = node_5_wins_term_5_and_repairs_node_3(logs)
    assert voters == {4}
    assert not FIVE.is_majority(voters | {5})


@pytest.mark.negative_control
def test_negative_control_committing_by_count_alone_loses_a_committed_entry(monkeypatch):
    # Proves test_the_leader_does_not_commit_that_entry can fail: with the term check removed,
    # the Leader commits entry 2, which node 5 then erases.
    def count_only(lead, commit_index, log, majority):
        held = sorted(
            [log.last_index, *(lead.match_index(f) for f in lead.followers)], reverse=True
        )
        return max(commit_index, held[majority - 1])

    monkeypatch.setattr(Leadership, "commit_index_after", count_only)
    assert committed(LEADER_TERM, FIVE, LOGS_AT_TERM_4[1], confirmed={2: 2, 3: 2}) == 2
    _, node_3_after = node_5_wins_term_5_and_repairs_node_3(LOGS_AT_TERM_4)
    assert node_3_after.entry_at(2) != LOGS_AT_TERM_4[1].entry_at(2)


# --- Invariants over random confirmations ----------------------------------------------


@pytest.mark.parametrize("seed", range(40))
def test_every_commit_is_on_a_majority_and_from_the_leaders_term(seed):
    rng = random.Random(seed)
    cluster = Cluster(range(1, rng.choice([3, 4, 5, 6, 7]) + 1))
    term = rng.randint(2, 6)
    # Terms never fall along a log, and the Leader's own entries come last.
    log = make_log(sorted(rng.randint(1, term) for _ in range(rng.randint(1, 8))))
    confirmed = {f: rng.randint(0, log.last_index) for f in cluster.peers_of(LEADER)}
    before = rng.randint(0, log.last_index)

    after = committed(term, cluster, log, confirmed=confirmed, commit_index=before)

    assert after >= before
    if after > before:
        holders = {LEADER} | {f for f, index in confirmed.items() if index >= after}
        assert cluster.is_majority(holders)
        assert log.term_at(after) == term
