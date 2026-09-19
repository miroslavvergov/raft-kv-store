"""Tests for the election-trace checker: a clean election passes, and each
of its six rules is shown to fail on a trace that breaks it — so a clean
verdict on a real trace means something.
"""

import json

from tests.election_traces.checker import check_election_trace, main


def event(name, nid, term, vote, role="follower", log=(0, 0), msg=None, prop=None):
    fields = {
        "name": name,
        "nid": nid,
        "role": role,
        "state": {"term": term, "vote": vote},
        "log": {"term": log[0], "index": log[1]},
    }
    if msg is not None:
        fields["msg"] = msg
    if prop is not None:
        fields["prop"] = prop
    return {"source": "node", "event": fields}


def request(sender, receiver, term, log=(0, 0)):
    return {"type": "RequestVote", "from": sender, "to": receiver, "term": term,
            "logTerm": log[0], "index": log[1]}


def answer(sender, receiver, term, granted):
    return {"type": "RequestVoteResponse", "from": sender, "to": receiver, "term": term,
            "reject": not granted}


def init(nid, peers, term=0, vote=None):
    return event("InitState", nid, term, vote, prop={"peers": peers})


def clean_election():
    """Three nodes; node 1 wins term 1 with node 2's vote."""
    return [
        init(1, [2, 3]),
        init(2, [1, 3]),
        init(3, [1, 2]),
        event("BecomeCandidate", 1, 1, 1, role="candidate"),
        event("SendRequestVoteRequest", 1, 1, 1, role="candidate", msg=request(1, 2, 1)),
        event("ReceiveRequestVoteRequest", 2, 0, None, msg=request(1, 2, 1)),
        event("BecomeFollower", 2, 1, 1),
        event("SendRequestVoteResponse", 2, 1, 1, msg=answer(2, 1, 1, True)),
        event("ReceiveRequestVoteResponse", 1, 1, 1, role="candidate",
              msg=answer(2, 1, 1, True), prop={"sentInTerm": 1}),
        event("BecomeLeader", 1, 1, 1, role="leader"),
    ]


def test_a_clean_election_breaks_no_rule():
    verdict = check_election_trace(clean_election())
    assert verdict.problems == []
    assert verdict.leaders_by_term == {1: [1]}


def test_harness_entries_are_ignored():
    entries = clean_election() + [
        {"source": "net", "event": {"name": "Deliver", "msg": request(1, 3, 1)}},
        {"source": "state", "event": {"name": "ClusterState", "nodes": {}}},
    ]
    assert check_election_trace(entries).problems == []


def test_rule_1_two_leaders_in_one_term():
    entries = clean_election() + [
        event("BecomeCandidate", 3, 1, 3, role="candidate"),
        event("ReceiveRequestVoteResponse", 3, 1, 3, role="candidate",
              msg=answer(2, 3, 1, True), prop={"sentInTerm": 1}),
        event("BecomeLeader", 3, 1, 3, role="leader"),
    ]
    assert "term 1 had 2 leaders: [1, 3]" in check_election_trace(entries).problems


def test_rule_2_two_candidates_given_a_vote_in_one_term():
    entries = clean_election() + [
        event("ReceiveRequestVoteRequest", 2, 1, 1, msg=request(3, 2, 1)),
        event("SendRequestVoteResponse", 2, 1, 3, msg=answer(2, 3, 1, True)),
    ]
    assert "node 2 voted for [1, 3] in term 1" in check_election_trace(entries).problems


def test_rule_3_a_vote_for_a_candidate_whose_log_is_behind():
    entries = [
        init(1, [2, 3]),
        init(2, [1, 3], term=2),
        event("ReceiveRequestVoteRequest", 2, 2, None, log=(2, 5),
              msg=request(1, 2, 3, log=(1, 9))),
        event("SendRequestVoteResponse", 2, 3, 1, log=(2, 5), msg=answer(2, 1, 3, True)),
    ]
    problems = check_election_trace(entries).problems
    assert problems == [
        "node 2 voted for 1 in term 3 although the candidate's log (term 1, index 9) "
        "is behind its own (term 2, index 5)"
    ]


def test_rule_3_a_vote_answering_a_request_from_another_term():
    entries = [
        init(2, [1, 3], term=5),
        event("ReceiveRequestVoteRequest", 2, 5, None, msg=request(1, 2, 3)),
        event("SendRequestVoteResponse", 2, 5, 1, msg=answer(2, 1, 5, True)),
    ]
    assert check_election_trace(entries).problems == [
        "node 2 granted a term-5 vote to 1, but its requests were for term(s) [3]"
    ]


def test_rule_4_a_leader_without_a_majority():
    entries = [
        init(1, [2, 3, 4, 5]),
        event("BecomeCandidate", 1, 1, 1, role="candidate"),
        event("ReceiveRequestVoteResponse", 1, 1, 1, role="candidate",
              msg=answer(2, 1, 1, True), prop={"sentInTerm": 1}),
        event("BecomeLeader", 1, 1, 1, role="leader"),
    ]
    assert check_election_trace(entries).problems == [
        "node 1 became leader of term 1 with votes from [1, 2] — not a majority of 5"
    ]


def test_rule_4_a_grant_from_an_earlier_election_does_not_count():
    entries = [
        init(1, [2, 3]),
        event("BecomeCandidate", 1, 2, 1, role="candidate"),
        event("ReceiveRequestVoteResponse", 1, 2, 1, role="candidate",
              msg=answer(2, 1, 1, True), prop={"sentInTerm": 1}),
        event("BecomeLeader", 1, 2, 1, role="leader"),
    ]
    assert check_election_trace(entries).problems == [
        "node 1 became leader of term 2 with votes from [1] — not a majority of 3"
    ]


def test_rule_5_a_term_that_goes_down_across_a_restart():
    entries = clean_election() + [init(2, [1, 3], term=0, vote=None)]
    problems = check_election_trace(entries).problems
    assert "node 2's term went down from 1 to 0 (InitState)" in problems


def test_rule_5_a_vote_that_changes_within_its_term():
    entries = clean_election() + [event("BecomeFollower", 2, 1, None)]
    assert (
        "node 2's vote in term 1 changed from 1 to None (BecomeFollower)"
        in check_election_trace(entries).problems
    )


def test_rule_5_a_first_vote_in_a_term_is_not_a_change():
    entries = [init(2, [1, 3], term=4), event("SendRequestVoteResponse", 2, 4, 3,
                                              msg=answer(2, 3, 4, False))]
    assert check_election_trace(entries).problems == []


def test_rule_6_an_answer_whose_vote_was_not_on_disk():
    entries = clean_election() + [{
        "source": "disk",
        "event": {"name": "DiskCheck", "nid": 2, "ok": False,
                  "disk": {"term": 0, "vote": None}, "memory": {"term": 1, "vote": 1}},
    }]
    assert check_election_trace(entries).problems == [
        "node 2 answered with term 1, vote 1 in memory, but its file held term 0, vote None"
    ]


def write_trace(path, entries, negative_control):
    header = {"test": {"id": path.stem, "result": "passed", "negativeControl": negative_control}}
    path.write_text("\n".join(json.dumps(line) for line in [header, *entries]) + "\n")


def test_rechecking_a_directory_fails_only_on_unexpected_problems(tmp_path, capsys):
    broken = clean_election() + [event("BecomeFollower", 2, 1, None)]
    write_trace(tmp_path / "clean.jsonl", clean_election(), negative_control=False)
    write_trace(tmp_path / "control.jsonl", broken, negative_control=True)
    assert main(tmp_path) == 0

    write_trace(tmp_path / "broken.jsonl", broken, negative_control=False)
    assert main(tmp_path) == 1
    assert "PROBLEMS  broken" in capsys.readouterr().out
