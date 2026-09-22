"""Re-check election safety from a recorded trace alone.

It reads only the nodes' trace events and the harness's disk checks, and imports nothing from
`raftkv`, so a bug in the code under test cannot also hide in the check. Rules, per trace:

1. At most one node becomes Leader in any term.
2. A node votes for at most one Candidate in any term, itself included, whether the vote shows in
   an answer or only in the node's state (ELECT-8).
3. A node grants its vote only to a Candidate whose last log entry is at least as up to date as
   its own: a later last term, or the same last term and at least as long (ELECT-9, ELECT-10).
4. A node becomes Leader only after grants from a strict majority of its cluster, itself
   included, counting only members and each voter's first answer to a request sent in that term
   (ELECT-11, ELECT-12). A Leader whose cluster size is unknown, with no InitState, is flagged.
5. A node's term never goes down, and once it has voted in a term, that vote never changes while
   the term lasts, across restarts too (PERSIST-1, PERSIST-2, STATE-6).
6. Every vote read back from a node's file matches the vote it held when it answered (PERSIST-2).
7. Every node that commits an index commits the same entry there, term and command.
8. A new Leader's log holds every entry already committed, in its own term or in an earlier
   one. A Leader of an earlier term need not: a vote cast in that term can arrive after a later
   term has committed, which legitimately elects a Leader that is behind.
9. A node's commit index never goes down while it runs; a restart resets it, as it is not
   persisted.

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
    """Return whether `candidate`'s (last term, last index) is at least as up to date as `voter`'s.

    ELECT-10: a later last term wins; between equal last terms, the longer log wins.
    """
    (candidate_term, candidate_index), (voter_term, voter_index) = candidate, voter
    return candidate_term > voter_term or (
        candidate_term == voter_term and candidate_index >= voter_index
    )


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
    first_answers = defaultdict(dict)  # (candidate, sent in term) -> {voter: granted}
    last_term_and_vote = {}
    committed = {}  # index -> (entry term, command, the node and term that first committed it)
    last_commit = {}  # node -> its commit index since it last started

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
        # NOTE: nothing resets a node's last term and vote when it restarts, so what it reloads
        # is compared with what it held before the crash (PERSIST-1, PERSIST-2).
        if node in last_term_and_vote:
            last_term, last_vote = last_term_and_vote[node]
            if term < last_term:
                verdict.problems.append(
                    f"node {node}'s term went down from {last_term} to {term} ({name})"
                )
            # NOTE: a term's first vote, cast from no vote, is not a change; only losing or
            # switching a vote while the term lasts breaks the rule.
            elif term == last_term and last_vote is not None and vote != last_vote:
                verdict.problems.append(
                    f"node {node}'s vote in term {term} changed from {last_vote} to {vote} ({name})"
                )
        last_term_and_vote[node] = (term, vote)
        # NOTE: a vote reaches the state without ever reaching an answer — a self-vote, or a
        # vote persisted by a cancelled caller — and still uses up the term's one vote (ELECT-8).
        if vote is not None:
            votes[(node, term)].add(vote)  # rule 2: every vote the node's state shows

        if name == "InitState":
            peers[node] = event.get("prop", {}).get("peers", [])
            last_commit[node] = 0  # rule 9: commitment is relearned after every start
        elif name == "Commit":
            commit = event["prop"]["commit"]
            if commit < last_commit.get(node, 0):  # rule 9
                verdict.problems.append(
                    f"node {node}'s commit index went down from {last_commit[node]} to {commit}"
                )
            last_commit[node] = max(commit, last_commit.get(node, 0))
            for index, entry_term, command in event["prop"]["entries"]:  # rule 7
                first = committed.setdefault(index, (entry_term, command, node, term))
                if (entry_term, command) != first[:2]:
                    verdict.problems.append(
                        f"node {node} committed (term {entry_term}, {command!r}) at index "
                        f"{index}, but node {first[2]} committed (term {first[0]}, "
                        f"{first[1]!r}) there"
                    )
        elif name == "ReceiveRequestVoteRequest":
            requests_seen[(node, msg["from"], msg["term"])] = (msg["logTerm"], msg["index"])
        elif name == "SendRequestVoteResponse" and not msg["reject"]:
            candidate = msg["to"]
            votes[(node, msg["term"])].add(candidate)  # rule 2
            candidate_log = requests_seen.get((node, candidate, msg["term"]))
            # NOTE: a grant carries the term of the request it answers, so a grant with no
            # request of that term is a vote cast on nothing, not a gap in the trace.
            if candidate_log is None:
                other_terms = sorted(
                    t for (n, c, t) in requests_seen if (n, c) == (node, candidate)
                )
                verdict.problems.append(
                    f"node {node} granted a term-{msg['term']} vote to {candidate}, but "
                    + (
                        f"its requests were for term(s) {other_terms}"
                        if other_terms
                        else "no request from it had arrived"
                    )
                )
            elif not _at_least_as_up_to_date(candidate_log, own_log):  # rule 3
                verdict.problems.append(
                    f"node {node} voted for {candidate} in term {msg['term']} although the "
                    f"candidate's log (term {candidate_log[0]}, index {candidate_log[1]}) is "
                    f"behind its own (term {own_log[0]}, index {own_log[1]})"
                )
        elif name == "ReceiveRequestVoteResponse":
            sent_in_term = event.get("prop", {}).get("sentInTerm")
            members = {node, *peers.get(node, [])}
            if sent_in_term == msg["term"] and msg["from"] in members:
                # NOTE: `setdefault` keeps the voter's first answer, so a duplicated reply
                # cannot turn a refusal into a grant (FAIL-2).
                first_answers[(node, sent_in_term)].setdefault(msg["from"], not msg["reject"])
        elif name == "BecomeLeader":  # rule 4
            leaders[term].add(node)
            leader_log = event.get("prop", {}).get("log")
            # NOTE: a commit from a later term is skipped, not flagged (rule 8); it is checked
            # against the next Leader of that term or later instead.
            if leader_log is not None:
                for index, (entry_term, command, _, committed_in) in sorted(committed.items()):
                    if committed_in > term:
                        continue
                    held = leader_log[index - 1] if index <= len(leader_log) else None
                    if held != [entry_term, command]:
                        verdict.problems.append(
                            f"node {node} became leader of term {term} holding {held} at index "
                            f"{index}, where (term {entry_term}, {command!r}) was committed in "
                            f"term {committed_in}"
                        )
            # NOTE: with no InitState the cluster size is unknown, so the majority check cannot
            # run and the win is flagged; rule 1 counts it above either way.
            if node not in peers:
                verdict.problems.append(
                    f"node {node} became leader of term {term}, but its cluster size is "
                    "unknown (no InitState)"
                )
                continue
            answers = first_answers[(node, term)]
            # NOTE: a Candidate sends itself no request, so its own vote never arrives as an
            # answer and is added here (ELECT-11).
            granted = {voter for voter, grant in answers.items() if grant} | {node}
            cluster_size = len(peers[node]) + 1
            if 2 * len(granted) <= cluster_size:
                verdict.problems.append(
                    f"node {node} became leader of term {term} with votes from {sorted(granted)} "
                    f"— not a majority of {cluster_size}"
                )

    for term, nodes in sorted(leaders.items()):  # rule 1
        if len(nodes) > 1:
            verdict.problems.append(f"term {term} had {len(nodes)} leaders: {sorted(nodes)}")
    for (node, term), candidates in sorted(votes.items()):  # rule 2
        if len(candidates) > 1:
            verdict.problems.append(f"node {node} voted for {sorted(candidates)} in term {term}")
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
