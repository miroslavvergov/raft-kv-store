"""Tests for the election-trace checker: a clean election passes, and each rule flags a breach.

A clean verdict on a real trace therefore means something.
"""

import json

from tests.election_traces.checker import check_election_trace, main


def node_event(name, *, node, term, vote, role="follower", last_log=(0, 0), msg=None, prop=None):
    """Return a node's trace entry, as the recorder writes it; `last_log` is (term, index)."""
    fields = {
        "name": name,
        "nid": node,
        "role": role,
        "state": {"term": term, "vote": vote},
        "log": {"term": last_log[0], "index": last_log[1]},
    }
    if msg is not None:
        fields["msg"] = msg
    if prop is not None:
        fields["prop"] = prop
    return {"source": "node", "event": fields}


def node_starts(node, *, peers, term=0, vote=None):
    """Return a node's InitState entry."""
    return node_event("InitState", node=node, term=term, vote=vote, prop={"peers": peers})


def request_msg(*, candidate, voter, term, last_log=(0, 0)):
    """Return a RequestVote's `msg` fields; `last_log` is the Candidate's (term, index)."""
    return {
        "type": "RequestVote",
        "from": candidate,
        "to": voter,
        "term": term,
        "logTerm": last_log[0],
        "index": last_log[1],
    }


def grant_msg(*, voter, candidate, term):
    """Return the `msg` fields of an answer granting the vote."""
    return _response_msg(voter, candidate, term, reject=False)


def refusal_msg(*, voter, candidate, term):
    """Return the `msg` fields of an answer refusing the vote."""
    return _response_msg(voter, candidate, term, reject=True)


def _response_msg(voter, candidate, term, reject):
    return {
        "type": "RequestVoteResponse",
        "from": voter,
        "to": candidate,
        "term": term,
        "reject": reject,
    }


def clean_election():
    """Return the trace of three nodes in which node 1 wins term 1 with node 2's vote."""
    return [
        node_starts(1, peers=[2, 3]),
        node_starts(2, peers=[1, 3]),
        node_starts(3, peers=[1, 2]),
        node_event("BecomeCandidate", node=1, term=1, vote=1, role="candidate"),
        node_event(
            "SendRequestVoteRequest",
            node=1,
            term=1,
            vote=1,
            role="candidate",
            msg=request_msg(candidate=1, voter=2, term=1),
        ),
        node_event(
            "ReceiveRequestVoteRequest",
            node=2,
            term=0,
            vote=None,
            msg=request_msg(candidate=1, voter=2, term=1),
        ),
        node_event("BecomeFollower", node=2, term=1, vote=1),
        node_event(
            "SendRequestVoteResponse",
            node=2,
            term=1,
            vote=1,
            msg=grant_msg(voter=2, candidate=1, term=1),
        ),
        node_event(
            "ReceiveRequestVoteResponse",
            node=1,
            term=1,
            vote=1,
            role="candidate",
            msg=grant_msg(voter=2, candidate=1, term=1),
            prop={"sentInTerm": 1},
        ),
        node_event("BecomeLeader", node=1, term=1, vote=1, role="leader"),
    ]


def test_a_clean_election_breaks_no_rule():
    verdict = check_election_trace(clean_election())
    assert verdict.problems == []
    assert verdict.leaders_by_term == {1: [1]}


def test_harness_entries_are_ignored():
    delivery = {"name": "Deliver", "msg": request_msg(candidate=1, voter=3, term=1)}
    entries = clean_election() + [
        {"source": "net", "event": delivery},
        {"source": "state", "event": {"name": "ClusterState", "nodes": {}}},
    ]
    assert check_election_trace(entries).problems == []


# --- Rule 1: at most one Leader per term ----------------------------------------------


def test_two_leaders_in_one_term_are_flagged():
    entries = clean_election() + [
        node_event("BecomeCandidate", node=3, term=1, vote=3, role="candidate"),
        node_event(
            "ReceiveRequestVoteResponse",
            node=3,
            term=1,
            vote=3,
            role="candidate",
            msg=grant_msg(voter=2, candidate=3, term=1),
            prop={"sentInTerm": 1},
        ),
        node_event("BecomeLeader", node=3, term=1, vote=3, role="leader"),
    ]
    assert "term 1 had 2 leaders: [1, 3]" in check_election_trace(entries).problems


# --- Rule 2: at most one vote per node per term ---------------------------------------


def test_a_vote_for_two_candidates_in_one_term_is_flagged():
    entries = clean_election() + [
        node_event(
            "ReceiveRequestVoteRequest",
            node=2,
            term=1,
            vote=1,
            msg=request_msg(candidate=3, voter=2, term=1),
        ),
        node_event(
            "SendRequestVoteResponse",
            node=2,
            term=1,
            vote=3,
            msg=grant_msg(voter=2, candidate=3, term=1),
        ),
    ]
    assert "node 2 voted for [1, 3] in term 1" in check_election_trace(entries).problems


def test_a_vote_seen_only_in_a_nodes_state_counts():
    # Node 2 granted node 1 in term 1, then persisted a vote for node 3 that it never answered.
    entries = clean_election() + [node_event("PersistVote", node=2, term=1, vote=3)]
    assert "node 2 voted for [1, 3] in term 1" in check_election_trace(entries).problems


# --- Rule 3: votes only for an up-to-date log, answering a request of that term -------


def test_a_vote_for_a_candidate_whose_log_is_behind_is_flagged():
    entries = [
        node_starts(1, peers=[2, 3]),
        node_starts(2, peers=[1, 3], term=2),
        node_event(
            "ReceiveRequestVoteRequest",
            node=2,
            term=2,
            vote=None,
            last_log=(2, 5),
            msg=request_msg(candidate=1, voter=2, term=3, last_log=(1, 9)),
        ),
        node_event(
            "SendRequestVoteResponse",
            node=2,
            term=3,
            vote=1,
            last_log=(2, 5),
            msg=grant_msg(voter=2, candidate=1, term=3),
        ),
    ]
    assert check_election_trace(entries).problems == [
        "node 2 voted for 1 in term 3 although the candidate's log (term 1, index 9) "
        "is behind its own (term 2, index 5)"
    ]


def test_a_vote_answering_a_request_from_another_term_is_flagged():
    entries = [
        node_starts(2, peers=[1, 3], term=5),
        node_event(
            "ReceiveRequestVoteRequest",
            node=2,
            term=5,
            vote=None,
            msg=request_msg(candidate=1, voter=2, term=3),
        ),
        node_event(
            "SendRequestVoteResponse",
            node=2,
            term=5,
            vote=1,
            msg=grant_msg(voter=2, candidate=1, term=5),
        ),
    ]
    assert check_election_trace(entries).problems == [
        "node 2 granted a term-5 vote to 1, but its requests were for term(s) [3]"
    ]


# --- Rule 4: a Leader only with a majority of this term's grants ----------------------


def test_a_leader_without_a_majority_is_flagged():
    entries = [
        node_starts(1, peers=[2, 3, 4, 5]),
        node_event("BecomeCandidate", node=1, term=1, vote=1, role="candidate"),
        node_event(
            "ReceiveRequestVoteResponse",
            node=1,
            term=1,
            vote=1,
            role="candidate",
            msg=grant_msg(voter=2, candidate=1, term=1),
            prop={"sentInTerm": 1},
        ),
        node_event("BecomeLeader", node=1, term=1, vote=1, role="leader"),
    ]
    assert check_election_trace(entries).problems == [
        "node 1 became leader of term 1 with votes from [1, 2] — not a majority of 5"
    ]


def test_a_grant_from_an_earlier_election_does_not_count_toward_a_majority():
    entries = [
        node_starts(1, peers=[2, 3]),
        node_event("BecomeCandidate", node=1, term=2, vote=1, role="candidate"),
        node_event(
            "ReceiveRequestVoteResponse",
            node=1,
            term=2,
            vote=1,
            role="candidate",
            msg=grant_msg(voter=2, candidate=1, term=1),
            prop={"sentInTerm": 1},
        ),
        node_event("BecomeLeader", node=1, term=2, vote=1, role="leader"),
    ]
    assert check_election_trace(entries).problems == [
        "node 1 became leader of term 2 with votes from [1] — not a majority of 3"
    ]


def test_a_grant_from_a_non_member_does_not_count_toward_a_majority():
    entries = [
        node_starts(1, peers=[2, 3]),
        node_event("BecomeCandidate", node=1, term=1, vote=1, role="candidate"),
        node_event(
            "ReceiveRequestVoteResponse",
            node=1,
            term=1,
            vote=1,
            role="candidate",
            msg=grant_msg(voter=9, candidate=1, term=1),
            prop={"sentInTerm": 1},
        ),
        node_event("BecomeLeader", node=1, term=1, vote=1, role="leader"),
    ]
    assert check_election_trace(entries).problems == [
        "node 1 became leader of term 1 with votes from [1] — not a majority of 3"
    ]


def test_only_a_voters_first_answer_counts_toward_a_majority():
    # Node 2 refused, then granted in the same term: only the refusal counts.
    entries = [
        node_starts(1, peers=[2, 3]),
        node_event("BecomeCandidate", node=1, term=1, vote=1, role="candidate"),
        node_event(
            "ReceiveRequestVoteResponse",
            node=1,
            term=1,
            vote=1,
            role="candidate",
            msg=refusal_msg(voter=2, candidate=1, term=1),
            prop={"sentInTerm": 1},
        ),
        node_event(
            "ReceiveRequestVoteResponse",
            node=1,
            term=1,
            vote=1,
            role="candidate",
            msg=grant_msg(voter=2, candidate=1, term=1),
            prop={"sentInTerm": 1},
        ),
        node_event("BecomeLeader", node=1, term=1, vote=1, role="leader"),
    ]
    assert check_election_trace(entries).problems == [
        "node 1 became leader of term 1 with votes from [1] — not a majority of 3"
    ]


def test_a_leader_whose_cluster_size_is_unknown_is_flagged():
    # No InitState for node 1, so nothing says how many votes a majority needs.
    entries = [
        node_event("BecomeCandidate", node=1, term=1, vote=1, role="candidate"),
        node_event(
            "ReceiveRequestVoteResponse",
            node=1,
            term=1,
            vote=1,
            role="candidate",
            msg=grant_msg(voter=2, candidate=1, term=1),
            prop={"sentInTerm": 1},
        ),
        node_event("BecomeLeader", node=1, term=1, vote=1, role="leader"),
    ]
    assert check_election_trace(entries).problems == [
        "node 1 became leader of term 1, but its cluster size is unknown (no InitState)"
    ]


# --- Rule 5: terms never go down; a vote never changes within its term ----------------


def test_a_term_that_goes_down_across_a_restart_is_flagged():
    entries = clean_election() + [node_starts(2, peers=[1, 3], term=0, vote=None)]
    problems = check_election_trace(entries).problems
    assert "node 2's term went down from 1 to 0 (InitState)" in problems


def test_a_vote_that_changes_within_its_term_is_flagged():
    entries = clean_election() + [node_event("BecomeFollower", node=2, term=1, vote=None)]
    assert (
        "node 2's vote in term 1 changed from 1 to None (BecomeFollower)"
        in check_election_trace(entries).problems
    )


def test_a_first_vote_in_a_term_is_not_flagged_as_a_change():
    entries = [
        node_starts(2, peers=[1, 3], term=4),
        node_event(
            "SendRequestVoteResponse",
            node=2,
            term=4,
            vote=3,
            msg=refusal_msg(voter=2, candidate=3, term=4),
        ),
    ]
    assert check_election_trace(entries).problems == []


# --- Rule 6: every answered vote is on disk -------------------------------------------


def test_an_answer_whose_vote_was_not_on_disk_is_flagged():
    disk_check = {
        "name": "DiskCheck",
        "nid": 2,
        "ok": False,
        "disk": {"term": 0, "vote": None},
        "memory": {"term": 1, "vote": 1},
    }
    entries = clean_election() + [{"source": "disk", "event": disk_check}]
    assert check_election_trace(entries).problems == [
        "node 2 answered with term 1, vote 1 in memory, but its file held term 0, vote None"
    ]


# --- Re-checking a directory of traces ------------------------------------------------


def write_trace(path, entries, negative_control):
    """Write `entries` to `path` as a `.jsonl` trace, headed as the recorder heads it."""
    header = {"test": {"id": path.stem, "result": "passed", "negativeControl": negative_control}}
    path.write_text("\n".join(json.dumps(line) for line in [header, *entries]) + "\n")


def test_rechecking_a_directory_fails_only_on_unexpected_problems(tmp_path, capsys):
    broken = clean_election() + [node_event("BecomeFollower", node=2, term=1, vote=None)]
    write_trace(tmp_path / "clean.jsonl", clean_election(), negative_control=False)
    write_trace(tmp_path / "control.jsonl", broken, negative_control=True)
    assert main(tmp_path) == 0

    write_trace(tmp_path / "broken.jsonl", broken, negative_control=False)
    assert main(tmp_path) == 1
    assert "PROBLEMS  broken" in capsys.readouterr().out


# --- Rules 7-9: committed entries agree, survive into every new Leader, never go back --


def commit_event(node, *, term, commit, entries, role="follower"):
    """Return a node's Commit entry; `entries` are the newly committed `[index, term, command]`."""
    return node_event(
        "Commit",
        node=node,
        term=term,
        vote=None,
        role=role,
        prop={"commit": commit, "entries": entries},
    )


def leader_starts_and_wins(node, *, term, log, peers=(2, 3)):
    """Return a node's InitState and its BecomeLeader for `term`; `log` is [term, command] pairs."""
    return [
        node_starts(node, peers=list(peers)),
        node_event(
            "BecomeLeader",
            node=node,
            term=term,
            vote=node,
            role="leader",
            prop={"next": {}, "match": {}, "log": log},
        ),
    ]


def test_nodes_committing_the_same_entries_break_no_rule():
    entries = [
        commit_event(1, term=1, commit=2, entries=[[1, 1, ""], [2, 1, "x=5"]], role="leader"),
        commit_event(2, term=1, commit=1, entries=[[1, 1, ""]]),
        commit_event(2, term=1, commit=2, entries=[[2, 1, "x=5"]]),
    ]
    assert check_election_trace(entries).problems == []


def test_two_nodes_committing_different_commands_at_one_index_is_flagged():
    entries = [
        commit_event(1, term=1, commit=2, entries=[[1, 1, ""], [2, 1, "x=5"]], role="leader"),
        commit_event(2, term=1, commit=2, entries=[[1, 1, ""], [2, 1, "x=6"]]),
    ]
    assert check_election_trace(entries).problems == [
        "node 2 committed (term 1, 'x=6') at index 2, but node 1 committed (term 1, 'x=5') there"
    ]


def test_two_nodes_committing_entries_of_different_terms_at_one_index_is_flagged():
    entries = [
        commit_event(1, term=2, commit=1, entries=[[1, 2, "x=5"]], role="leader"),
        commit_event(3, term=3, commit=1, entries=[[1, 3, "x=5"]]),
    ]
    assert check_election_trace(entries).problems == [
        "node 3 committed (term 3, 'x=5') at index 1, but node 1 committed (term 2, 'x=5') there"
    ]


def test_a_conflict_in_an_earlier_entry_of_a_multi_entry_commit_is_flagged():
    # The conflict is at index 2, not at the newest entry the Commit event reports.
    entries = [
        commit_event(1, term=1, commit=2, entries=[[1, 1, ""], [2, 1, "x=5"]], role="leader"),
        commit_event(2, term=1, commit=3, entries=[[1, 1, ""], [2, 1, "x=6"], [3, 1, "y=7"]]),
    ]
    assert check_election_trace(entries).problems == [
        "node 2 committed (term 1, 'x=6') at index 2, but node 1 committed (term 1, 'x=5') there"
    ]


def test_a_new_leader_holding_every_committed_entry_breaks_no_rule():
    entries = [
        commit_event(1, term=1, commit=2, entries=[[1, 1, ""], [2, 1, "x=5"]], role="leader"),
        *leader_starts_and_wins(2, term=2, log=[[1, ""], [1, "x=5"], [2, ""]], peers=(1, 3)),
    ]
    problems = check_election_trace(entries).problems
    # The win itself breaks rule 4 here (no votes in this short trace), so check only rule 8.
    assert not [p for p in problems if "holding" in p]


def test_a_new_leader_missing_a_committed_entry_is_flagged():
    entries = [
        commit_event(1, term=1, commit=2, entries=[[1, 1, ""], [2, 1, "x=5"]], role="leader"),
        *leader_starts_and_wins(2, term=2, log=[[1, ""], [2, ""]], peers=(1, 3)),
    ]
    assert (
        "node 2 became leader of term 2 holding [2, ''] at index 2, where (term 1, 'x=5') was "
        "committed in term 1" in check_election_trace(entries).problems
    )


def test_a_new_leader_whose_log_is_too_short_for_a_committed_entry_is_flagged():
    entries = [
        commit_event(1, term=1, commit=2, entries=[[1, 1, ""], [2, 1, "x=5"]], role="leader"),
        *leader_starts_and_wins(2, term=2, log=[[1, ""]], peers=(1, 3)),
    ]
    assert (
        "node 2 became leader of term 2 holding None at index 2, where (term 1, 'x=5') was "
        "committed in term 1" in check_election_trace(entries).problems
    )


def test_a_commit_index_that_goes_down_while_running_is_flagged():
    entries = [
        node_starts(2, peers=[1, 3]),
        commit_event(2, term=1, commit=3, entries=[[1, 1, ""], [2, 1, "a"], [3, 1, "b"]]),
        commit_event(2, term=1, commit=2, entries=[]),
    ]
    assert check_election_trace(entries).problems == ["node 2's commit index went down from 3 to 2"]


def test_a_commit_index_relearned_from_zero_after_a_restart_is_not_flagged():
    entries = [
        node_starts(2, peers=[1, 3]),
        commit_event(2, term=1, commit=3, entries=[[1, 1, ""], [2, 1, "a"], [3, 1, "b"]]),
        node_starts(2, peers=[1, 3], term=1),
        commit_event(2, term=1, commit=1, entries=[[1, 1, ""]]),
    ]
    assert check_election_trace(entries).problems == []


def test_a_leader_of_an_earlier_term_missing_a_later_terms_commit_is_not_flagged():
    # Node 1 ran for term 1 and node 2 granted, but the grant was delayed in the network. Node 3
    # won term 2 and committed. The grant then arrives and node 1 becomes a legitimate term-1
    # Leader without node 3's entry: rule 8 says nothing about a Leader of an earlier term.
    entries = [
        commit_event(3, term=2, commit=1, entries=[[1, 2, ""]], role="leader"),
        *leader_starts_and_wins(1, term=1, log=[], peers=(2, 3)),
    ]
    assert not [p for p in check_election_trace(entries).problems if "holding" in p]


def test_a_leader_of_the_same_term_as_the_commit_must_still_hold_it():
    entries = [
        commit_event(3, term=2, commit=1, entries=[[1, 2, "x=5"]], role="leader"),
        *leader_starts_and_wins(1, term=2, log=[[2, "other"]], peers=(2, 3)),
    ]
    assert (
        "node 1 became leader of term 2 holding [2, 'other'] at index 1, where (term 2, 'x=5') "
        "was committed in term 2" in check_election_trace(entries).problems
    )


def test_a_leader_of_an_earlier_term_missing_an_old_entry_committed_later_is_not_flagged():
    # The entries are from term 1, but a term-3 Leader is the first to commit them. What rule 8
    # compares is the term they were committed in, not the term they were written in, so a
    # term-2 Leader that lacks them is not flagged.
    entries = [
        commit_event(3, term=3, commit=2, entries=[[1, 1, ""], [2, 1, "x=5"]], role="leader"),
        *leader_starts_and_wins(1, term=2, log=[], peers=(2, 3)),
    ]
    assert not [p for p in check_election_trace(entries).problems if "holding" in p]
