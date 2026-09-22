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
    ruff format raftkv tests && ruff check raftkv tests   # settings in ruff.toml

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
  timeout fires), `cmd` (a client command reaches a node), `crash` (a node
  restarts from its file), `disk` (a vote read back from a node's file),
  `state` (every node after the step: role, term, vote, last log index, and
  commit index).
- `<test>.jsonl` — for tools. The same run as structured events in the shape
  of etcd's `TracingEvent` (`BecomeCandidate`, `ReceiveRequestVoteRequest`,
  `SendRequestVoteResponse`, `BecomeLeader`, `Replicate`, `Commit`, ...), each
  carrying the node's term, vote, role, and last log entry.

Every trace is also re-checked by an independent checker that reads only the
trace (`tests/election_traces/checker.py`): at most one Leader per term, at
most one vote per node per term, votes only for up-to-date logs, Leaders only
with a majority, terms and votes that survive restarts, votes on disk before
answering, every node committing the same entry at each index, every new
Leader holding every entry committed in its term or earlier, and commit
indexes that never go down while a node runs. A passing test whose trace breaks a rule fails at teardown;
tests marked `negative_control` break one on purpose, and the summary at the
end shows what the checker found in them. To re-check saved traces later:

    python -m tests.election_traces.checker test-traces/elections

Without the option nothing is recorded, and tracing costs a logger level
check per call.

## Known deviations from the specification

- **REPL-6** says a Leader decrements `nextIndex` whenever an AppendEntries is
  rejected. A Leader here decrements only for the rejection answering the probe
  currently outstanding for that Follower; a duplicated or superseded rejection
  changes nothing. Taken literally the two requirements conflict, since FAIL-1
  requires handling the same RPC twice to have no further effect, and one
  duplicated packet would otherwise walk `nextIndex` back an extra step and
  resend entries for nothing.
- **A new Leader appends an empty entry** in its own term on winning, so that
  entries from earlier terms can commit without waiting for a client write
  (APPLY-3). No requirement asks for it, and the command it carries — the empty
  string — is reserved, which DD-21's "opaque string" does not anticipate.

## Status

Built and tested: the replicated log's consistency check and repair, durable
term, vote, and log in SQLite (persisted before every answer), leader election
(voting, vote counting, stepping down), and replication: a Leader appends
client commands and an empty entry on winning, sends each Follower what it
lacks, backs off on rejection, and commits what a majority holds from its own
term; a Follower accepts entries and learns the commit index. Not yet built:
applying committed entries, timers, the HTTP transport, and the key-value
store.
