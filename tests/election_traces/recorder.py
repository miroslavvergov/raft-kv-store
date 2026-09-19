"""Collects each test's election trace and writes it out once the test is over.

Used by `pytest --trace-elections` (see tests/conftest.py). While a test
runs, the recorder only appends log records to a list: nothing is
formatted or written, so recording adds no disk I/O to the event loop and
no `await`, and cannot reorder anything the test does. Once the test has
finished — its fixtures torn down, its nodes stopped — the records are
written as two files in the traces directory:

- `<test>.log`, for reading: every node's log line in etcd's raft log
  format, interleaved with what the test harness did (`net`, `clock`,
  `crash`, `disk`, `state`), numbered in the order it happened.
- `<test>.jsonl`, for machines: a header line, then every trace event in
  the same order — the nodes' events in etcd's `TracingEvent` shape and the
  harness's own — each with the number of the `.log` line it follows.

Each trace is then re-checked by `checker.check_election_trace`, and the
verdict is written at the top of the `.log` file.
"""

import json
import logging
import pathlib
import re
import shutil

from raftkv.tracing import LOG_LINES, TRACE_EVENTS
from tests.election_traces.checker import check_election_trace

HARNESS = "tests.cluster"

_LEGEND = """\
# Lines marked "raft" are a node's own log, in etcd's raft log format; each
# starts with the node's ID, and "vote: 0" means no vote cast yet.
# net = a message delivered, dropped, or duplicated · clock = an election timeout
# fires · crash = a node restarts from its file · disk = a vote read back from a
# node's file · state = every running node after the step (t = term, v = vote)."""


class ElectionTraceRecorder(logging.Handler):
    """A logging handler that records one test at a time and writes its trace files.

    Attributes:
        directory: Where trace files are written.
        results: For every trace written: the test's ID, its verdict,
            whether it is a negative control, and the `.log` file's path.
    """

    def __init__(self, directory):
        super().__init__(level=logging.DEBUG)
        self.directory = pathlib.Path(directory)
        self.results = []
        self._records = None

    def install(self):
        """Empty the traces directory and start receiving every node's and the harness's records."""
        shutil.rmtree(self.directory, ignore_errors=True)
        self.directory.mkdir(parents=True)
        for name in (LOG_LINES, TRACE_EVENTS, HARNESS):
            logger = logging.getLogger(name)
            logger.setLevel(logging.DEBUG)
            logger.addHandler(self)

    def uninstall(self):
        for name in (LOG_LINES, TRACE_EVENTS, HARNESS):
            logger = logging.getLogger(name)
            logger.removeHandler(self)
            logger.setLevel(logging.NOTSET)

    def emit(self, record):
        if self._records is not None:
            self._records.append(record)

    def begin(self):
        """Start recording a new test."""
        self._records = []

    def finish(self, test_id, outcome, negative_control):
        """Stop recording, check the trace, and write its files.

        Args:
            test_id: The test's pytest node ID.
            outcome: "passed", "failed", or "skipped".
            negative_control: Whether the test deliberately breaks a
                safety rule, so problems in its trace are expected.

        Returns:
            The checker's TraceVerdict, or None if the test recorded
            nothing (it ran no node).
        """
        records, self._records = self._records, None
        if not records:
            return None
        lines, entries = self._render(records)
        verdict = check_election_trace(entries)
        base = self.directory / _file_name(test_id)

        header = {"id": test_id, "result": outcome, "negativeControl": negative_control}
        with open(base.with_suffix(".jsonl"), "w") as jsonl:
            jsonl.write(json.dumps({"test": header}, sort_keys=True) + "\n")
            for entry in entries:
                jsonl.write(json.dumps(entry, sort_keys=True) + "\n")

        leaders = ", ".join(
            f"term {term} → node {nodes[0] if len(nodes) == 1 else nodes}"
            for term, nodes in verdict.leaders_by_term.items()
        )
        summary = [
            f"# test:     {test_id}",
            f"# result:   {outcome}",
            f"# leaders:  {leaders or 'none'}",
            "# checker:  "
            + ("no problems" if not verdict.problems else f"{len(verdict.problems)} problem(s)"
               + (" — expected: this is a negative control" if negative_control else "")),
        ]
        summary += [f"#   - {problem}" for problem in verdict.problems]
        base.with_suffix(".log").write_text(
            "\n".join(summary) + "\n#\n" + _LEGEND + "\n\n" + "\n".join(lines) + "\n"
        )

        self.results.append((test_id, verdict, negative_control, base.with_suffix(".log")))
        return verdict

    @staticmethod
    def _render(records):
        """Turn records into numbered `.log` lines and `.jsonl` entries, in recorded order."""
        lines, entries = [], []
        for record in records:
            if record.name == TRACE_EVENTS:
                entries.append({"seq": len(entries) + 1, "logLine": len(lines), "source": "node",
                                "event": record.trace_event.as_dict()})
                continue
            source = "raft" if record.name == LOG_LINES else record.trace_source
            lines.append(f"{len(lines) + 1:>4}  {source:<5}  {record.getMessage()}")
            harness_event = getattr(record, "trace_event", None)
            if harness_event is not None:
                entries.append({"seq": len(entries) + 1, "logLine": len(lines), "source": source,
                                "event": harness_event})
        return lines, entries


def _file_name(test_id):
    """Turn `tests/persistence/test_x.py::test_y[3]` into `test_x__test_y-3`."""
    path, _, name = test_id.partition("::")
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{pathlib.Path(path).stem}__{name}").strip("-")
