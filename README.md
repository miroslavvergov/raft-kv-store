# raft-kv-store

A small distributed key-value store replicated with a from-scratch
implementation of the Raft consensus algorithm (Ongaro & Ousterhout, 2014),
written in Python 3 with asyncio.

Full specification: see the accompanying A4 Software Requirements
Specification for the complete requirements, design decisions, and test
strategy this project implements against.

## Development setup

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pytest -q

## Tracing elections

    pytest --trace-elections

Records, for every test that runs a node, what each node decided and what
the test's simulated network did, and writes it to `test-traces/elections/`
(git-ignored, emptied at the start of each traced run). Use it when you want
to see *why* a test passed or failed, not just *that* it did.

- `<test>.log` — for reading. Each node's own log lines, in the format etcd's
  raft uses (`2 [logterm: 0, index: 0, vote: 0] cast RequestVote for 1 ...`;
  `vote: 0` means no vote cast yet), interleaved with what the harness did:
  `net` (a message delivered, dropped, or duplicated), `clock` (an election
  timeout fires), `crash` (a node restarts from its file), `disk` (a vote
  read back from a node's file), `state` (every node after the step).
- `<test>.jsonl` — for tools. The same run as structured events in the shape
  of etcd's `TracingEvent` (`BecomeCandidate`, `ReceiveRequestVoteRequest`,
  `SendRequestVoteResponse`, `BecomeLeader`, ...), each carrying the node's
  term, vote, role, and last log entry.

Every trace is also re-checked by an independent checker that reads only the
trace (`tests/election_traces/checker.py`, the counterpart of etcd's TLA+
trace validation): at most one Leader per term, at most one vote per node
per term, votes only for up-to-date logs, Leaders only with a majority,
terms and votes that survive restarts, and votes on disk before answering.
A test whose trace breaks a rule fails; tests marked `negative_control`
break one on purpose, and the summary at the end shows what the checker
found in them. To re-check saved traces later:

    python -m tests.election_traces.checker test-traces/elections

Without the option nothing is recorded, and tracing costs one level check
per call.

## Status

Environment and repository scaffold only — implementation has not started yet.
