"""Re-check election safety from a recorded trace alone.

The counterpart of etcd's TLA+ trace validation: it reads only what was
recorded — the nodes' trace events and the harness's disk checks — and
re-derives from them, independently of the tests' own assertions, whether
the run broke any of the rules below. It imports nothing from `raftkv`, so
a bug in the code under test cannot also hide inside the check.

Rules, per trace:

1. At most one node becomes Leader in any term (Election Safety).
2. A node votes for at most one Candidate in any term, itself included
   (ELECT-8).
3. A node grants its vote only to a Candidate whose last log entry is at
   least as up to date as its own: a later last term, or the same last
   term and at least as long (ELECT-9, ELECT-10).
4. A node becomes Leader only after grants from a strict majority of the
   cluster, itself included, answering requests sent in that same term
   (ELECT-11, ELECT-12).
5. A node's term never goes down, and once it has voted in a term its
   vote never changes while that term lasts — including across restarts,
   which is what persisting the term and vote guarantees (PERSIST-1,
   PERSIST-2, STATE-6).
6. Every vote read back from a node's file matches the vote the node held
   when it answered (PERSIST-2).

Run on a directory of traces: `python -m tests.election_traces.checker test-traces/elections`.
"""

import json
import pathlib
import sys
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class TraceVerdict:
    """What the checker found in one trace.

    Attributes:
        problems: One sentence per broken rule, in the order found.
        leaders_by_term: Every node that became Leader, per term.
    """

    problems: list[str] = field(default_factory=list)
    leaders_by_term: dict[int, list[int]] = field(default_factory=dict)


def _at_least_as_up_to_date(candidate, voter):
    """ELECT-10 on (last term, last index) pairs."""
    return candidate[0] > voter[0] or (candidate[0] == voter[0] and candidate[1] >= voter[1])


def check_election_trace(entries):
    """Check one trace's entries against the rules in this module's docstring.

    Args:
        entries: The trace entries in the order they were recorded, each
            a dict with a `source` and an `event`, as written to the
            `.jsonl` file. Entries without an `event` are skipped.

    Returns:
        A TraceVerdict.
    """
    verdict = TraceVerdict()
    leaders = defaultdict(set)
    votes = defaultdict(set)
    peers = {}
    requests_seen = {}
    grants_received = defaultdict(set)
    last_term_and_vote = {}

    for entry in entries:
        event = entry.get("event")
        if event is None:
            continue
        name = event["name"]

        if entry["source"] == "disk":
            if not event["ok"]:
                verdict.problems.append(
                    f"node {event['nid']} answered with term {event['memory']['term']}, "
                    f"vote {event['memory']['vote']} in memory, but its file held term "
                    f"{event['disk']['term']}, vote {event['disk']['vote']}"
                )
            continue
        if entry["source"] != "node":
            continue

        node, term, vote = event["nid"], event["state"]["term"], event["state"]["vote"]
        own_log = (event["log"]["term"], event["log"]["index"])
        msg = event.get("msg")

        # Rule 5: term never goes down; a vote never changes within its term.
        if node in last_term_and_vote:
            last_term, last_vote = last_term_and_vote[node]
            if term < last_term:
                verdict.problems.append(
                    f"node {node}'s term went down from {last_term} to {term} ({name})"
                )
            elif term == last_term and last_vote is not None and vote != last_vote:
                verdict.problems.append(
                    f"node {node}'s vote in term {term} changed from {last_vote} to {vote} ({name})"
                )
        last_term_and_vote[node] = (term, vote)

        if name == "InitState":
            peers[node] = event.get("prop", {}).get("peers", [])
        elif name == "BecomeCandidate":
            votes[(node, term)].add(node)
        elif name == "ReceiveRequestVoteRequest":
            requests_seen[(node, msg["from"], msg["term"])] = (msg["logTerm"], msg["index"])
        elif name == "SendRequestVoteResponse" and not msg["reject"]:
            candidate = msg["to"]
            votes[(node, msg["term"])].add(candidate)  # rule 2
            candidate_log = requests_seen.get((node, candidate, msg["term"]))
            if candidate_log is None:
                other_terms = sorted(
                    t for (n, c, t) in requests_seen if (n, c) == (node, candidate)
                )
                verdict.problems.append(
                    f"node {node} granted a term-{msg['term']} vote to {candidate}, but "
                    + (f"its requests were for term(s) {other_terms}" if other_terms
                       else "no request from it had arrived")
                )
            elif not _at_least_as_up_to_date(candidate_log, own_log):  # rule 3
                verdict.problems.append(
                    f"node {node} voted for {candidate} in term {msg['term']} although the "
                    f"candidate's log (term {candidate_log[0]}, index {candidate_log[1]}) is "
                    f"behind its own (term {own_log[0]}, index {own_log[1]})"
                )
        elif name == "ReceiveRequestVoteResponse" and not msg["reject"]:
            if event.get("prop", {}).get("sentInTerm") == msg["term"]:
                grants_received[(node, msg["term"])].add(msg["from"])
        elif name == "BecomeLeader":
            leaders[term].add(node)
            granted = grants_received[(node, term)] | {node}
            cluster_size = len(peers.get(node, [])) + 1
            if 2 * len(granted) <= cluster_size:  # rule 4
                verdict.problems.append(
                    f"node {node} became leader of term {term} with votes from {sorted(granted)} "
                    f"— not a majority of {cluster_size}"
                )

    for term, nodes in sorted(leaders.items()):  # rule 1
        if len(nodes) > 1:
            verdict.problems.append(f"term {term} had {len(nodes)} leaders: {sorted(nodes)}")
    for (node, term), candidates in sorted(votes.items()):  # rule 2
        if len(candidates) > 1:
            verdict.problems.append(
                f"node {node} voted for {sorted(candidates)} in term {term}"
            )
    verdict.leaders_by_term = {term: sorted(nodes) for term, nodes in sorted(leaders.items())}
    return verdict


def read_trace(path):
    """Read a `.jsonl` trace: its header, then its entries."""
    lines = pathlib.Path(path).read_text().splitlines()
    header = json.loads(lines[0])["test"]
    return header, [json.loads(line) for line in lines[1:]]


def main(directory):
    """Re-check every trace in `directory`; return 1 if any unexpected problem is found."""
    unexpected = 0
    for path in sorted(pathlib.Path(directory).glob("*.jsonl")):
        header, entries = read_trace(path)
        verdict = check_election_trace(entries)
        expected = header["negativeControl"]
        status = "ok" if not verdict.problems else ("expected" if expected else "PROBLEMS")
        print(f"{status:<9} {header['id']}")
        for problem in verdict.problems:
            print(f"            - {problem}")
        if verdict.problems and not expected:
            unexpected += 1
    print(f"\n{unexpected} trace(s) with unexpected problems")
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "test-traces/elections"))
