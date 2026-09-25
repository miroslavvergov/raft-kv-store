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

## Layout

    raftkv/
      consensus/   pure Raft rules: log, roles, voting, replication, commit (no I/O)
      node/        DurableNodeState: runs those rules and persists before acting
      storage/     SQLite storage of term, vote, and log
      runtime/     RaftNode: the clock, the RPCs a node sends, and applying
      kvstore/     the key-value state machine committed entries are applied to
      tracing/     log lines and trace events for every node decision
    tests/
      consensus/ node/ storage/ kvstore/ tracing/   one folder per package
      cluster/     several nodes, every message delivered by the test, with safety checks
      runtime/     RaftNode alone, and clusters that run themselves on a ticked clock
      traces/      the trace recorder and checker behind --trace-elections
      support/     message builders, log fixtures, store doubles, in-memory network

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
  timeout fires), `cmd` (a client command reaches a node), `apply` (a node
  applies committed entries to its state machine), `crash` (a node
  restarts from its file), `disk` (a vote read back from a node's file),
  `state` (every node after the step: role, term, vote, last log index, and
  commit index).
- `<test>.jsonl` — for tools. The same run as structured events in the shape
  of etcd's `TracingEvent` (`BecomeCandidate`, `ReceiveRequestVoteRequest`,
  `SendRequestVoteResponse`, `BecomeLeader`, `Replicate`, `Commit`, `Apply`, ...),
  each
  carrying the node's term, vote, role, and last log entry.

Every trace is also re-checked by an independent checker that reads only the
trace (`tests/traces/checker.py`): at most one Leader per term, at
most one vote per node per term, votes only for up-to-date logs, Leaders only
with a majority, terms and votes that survive restarts, votes on disk before
answering, every node committing the same entry at each index, every new
Leader holding every entry committed in its term or earlier, commit indexes
that never go down while a node runs, and every node applying the same command
at each index, one index at a time. A passing test whose trace breaks a rule
fails at teardown;
tests marked `negative_control` break one on purpose, and the summary at the
end shows what the checker found in them. To re-check saved traces later:

    python -m tests.traces.checker test-traces/elections

Without the option nothing is recorded, and tracing costs a logger level
check per call.

## Where the design narrows a requirement

Each of these reads a requirement more narrowly than its literal wording, and a
design decision in the specification records why.

- **REPL-6** says a Leader decrements `nextIndex` whenever an AppendEntries is
  rejected. A Leader here decrements only for the rejection answering the probe
  currently outstanding for that Follower; a duplicated or superseded rejection
  changes nothing (DD-27). Taken literally, REPL-6 conflicts with FAIL-1, which
  requires handling the same RPC twice to have no further effect.
- **REPL-8** says a Follower overwrites any conflicting entry. A node here
  refuses, and stops, rather than change an entry it has already committed
  (DD-29); a correct cluster never asks it to.
- **FAIL-2** says an unanswered RPC is retried identically. A Candidate does
  resend the identical RequestVote, but a Leader sends its *current*
  AppendEntries at the next heartbeat instead: it starts where the lost one
  did, since `nextIndex` moves only on an answer, and carries the same entries
  plus any appended since (DD-30).
- **ELECT-2** counts only the Leader's AppendEntries and a granted vote as
  reasons to hold off an election. A node here also restarts its election
  timeout whenever its role or term changes, even on a refused vote that
  carries a newer term (DD-9).

## Status

Built and tested: the replicated log's consistency check and repair, durable
term, vote, and log in SQLite (persisted before every answer), leader election
(voting, vote counting, stepping down), and replication: a Leader appends
client commands and an empty entry on winning, sends each Follower what it
lacks, backs off on rejection, and commits what a majority holds from its own
term; a Follower accepts entries and learns the commit index. Committed
entries are applied in order to a key-value state machine, which every node
rebuilds by replaying its log after a restart, and a Leader knows whether it
has committed in its own term, one of the three conditions a linearizable read
waits on (CLIENT-8, CLIENT-9, CLIENT-10). A `RaftNode` runs each node by itself
on a clock counted in ticks: election timeouts, heartbeats, sending and
resending RPCs, and applying what commits, over any `Transport`; the tests run
whole clusters of them over an in-memory network. Not yet built: a `propose()`
that waits for commit and apply, a `read_barrier()` for linearizable reads,
request-ID deduplication, the HTTP transport, the client API, a node entry
point configured from environment variables, and the Docker packaging.
