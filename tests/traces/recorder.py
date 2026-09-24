"""Record each test's election trace in memory, then write and check it once the test is over.

Recording only appends log records, so it adds no I/O or `await` to the test. Each trace becomes
`<test>.log`, the numbered node log lines and harness steps headed by the checker's verdict, and
`<test>.jsonl`, a header and then every trace event with the number of the `.log` line it follows.
"""

import contextlib
import json
import logging
import pathlib
import re
import shutil
from typing import NamedTuple

from raftkv.tracing import LOG_LINES_LOGGER, TRACE_EVENTS_LOGGER
from tests.traces.checker import TraceVerdict, check_election_trace

HARNESS = "tests.cluster"

_LEGEND = """\
# Lines marked "raft" are a node's own log, in etcd's raft log format; each
# starts with the node's ID, and "vote: 0" means no vote cast yet.
# net = a message delivered, dropped, or duplicated · clock = an election timeout
# fires · cmd = a client command reaches a node · apply = a node applies committed
# entries to its state machine · crash = a node restarts from its file · disk = a
# vote read back from a node's file · state = every running node
# after the step (t = term, v = vote, log = last log index, c = commit index)."""


class TraceResult(NamedTuple):
    """One written trace.

    Attributes:
        test_id: The test's pytest node ID.
        verdict: The checker's verdict on the trace.
        negative_control: Whether the test breaks a safety rule on purpose.
        log_path: The trace's `.log` file.
    """

    test_id: str
    verdict: TraceVerdict
    negative_control: bool
    log_path: pathlib.Path

    @property
    def test_name(self):
        """Return the test's name without its file path."""
        return self.test_id.split("::")[-1]


class ElectionTraceRecorder(logging.Handler):
    """A logging handler that records one test at a time and writes its trace files.

    Attributes:
        directory: Where trace files are written.
        results: A TraceResult for every trace written.
    """

    def __init__(self, directory):
        super().__init__(level=logging.DEBUG)
        self.directory = pathlib.Path(directory)
        self.results = []
        self._records = None
        self._names_used = set()

    def install(self):
        """Empty the traces directory and start receiving the nodes' and the harness's records."""
        shutil.rmtree(self.directory, ignore_errors=True)
        self.directory.mkdir(parents=True)
        for name in (LOG_LINES_LOGGER, TRACE_EVENTS_LOGGER, HARNESS):
            logger = logging.getLogger(name)
            logger.setLevel(logging.DEBUG)
            logger.addHandler(self)

    def uninstall(self):
        """Stop receiving records and restore the loggers' levels."""
        for name in (LOG_LINES_LOGGER, TRACE_EVENTS_LOGGER, HARNESS):
            logger = logging.getLogger(name)
            logger.removeHandler(self)
            # NOTE: this runs after the last test, when caplog has already restored every level
            # it set, so clearing to NOTSET leaves these loggers as `install` found them.
            logger.setLevel(logging.NOTSET)

    def emit(self, record):
        """Keep `record` if a test is being recorded."""
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
            negative_control: Whether the test breaks a safety rule on purpose, so problems
                in its trace are expected.

        Returns:
            The checker's TraceVerdict, or None if the test ran no node.
        """
        records, self._records = self._records, None
        if not records:
            return None
        lines, entries = render(records)
        verdict = check_election_trace(entries)
        # NOTE: sanitizing can map two test IDs to one file name, so a suffix keeps a later
        # trace from erasing an earlier one.
        name = self._unused_name(_file_name(test_id))
        log_path = self.directory / f"{name}.log"

        header = {"id": test_id, "result": outcome, "negativeControl": negative_control}
        with open(self.directory / f"{name}.jsonl", "w") as jsonl:
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
            f"# checker:  {_checker_summary(verdict, negative_control)}",
        ]
        summary += [f"#   - {problem}" for problem in verdict.problems]
        log_path.write_text(
            "\n".join(summary) + "\n#\n" + _LEGEND + "\n\n" + "\n".join(lines) + "\n"
        )

        self.results.append(TraceResult(test_id, verdict, negative_control, log_path))
        return verdict

    def _unused_name(self, name):
        """Return `name`, or `name-2`, `name-3`, ... if an earlier trace of this run has it."""
        candidate, suffix = name, 1
        while candidate in self._names_used:
            suffix += 1
            candidate = f"{name}-{suffix}"
        self._names_used.add(candidate)
        return candidate


@contextlib.contextmanager
def collecting_trace_records():
    """Collect the nodes' trace events and the harness's steps while the block runs.

    Yields the list the records are appended to, ready for `render`. Both loggers are enabled
    for the block, then restored to their earlier levels.
    """
    handler = _RecordList()
    loggers = [logging.getLogger(name) for name in (TRACE_EVENTS_LOGGER, HARNESS)]
    # NOTE: caplog or a running recorder may already hold these loggers at a level, which the
    # restore below keeps.
    levels = [logger.level for logger in loggers]
    for logger in loggers:
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
    try:
        yield handler.records
    finally:
        for logger, level in zip(loggers, levels, strict=True):
            logger.removeHandler(handler)
            logger.setLevel(level)


class _RecordList(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def render(records):
    """Turn log records into numbered `.log` lines and `.jsonl` entries, in recorded order.

    Returns:
        The `.log` lines, and the entries the checker reads.
    """
    lines, entries = [], []
    for record in records:
        if record.name == TRACE_EVENTS_LOGGER:
            entries.append(
                {
                    "seq": len(entries) + 1,
                    # NOTE: a trace event adds no .log line, so logLine names the line it follows.
                    "logLine": len(lines),
                    "source": "node",
                    "event": record.trace_event.as_dict(),
                }
            )
            continue
        source = "raft" if record.name == LOG_LINES_LOGGER else record.trace_source
        lines.append(f"{len(lines) + 1:>4}  {source:<5}  {record.getMessage()}")
        harness_event = getattr(record, "trace_event", None)
        if harness_event is not None:
            entries.append(
                {
                    "seq": len(entries) + 1,
                    "logLine": len(lines),
                    "source": source,
                    "event": harness_event,
                }
            )
    return lines, entries


def _checker_summary(verdict, negative_control):
    """Return the text of the `.log` header's `# checker:` line."""
    if not verdict.problems:
        return "no problems"
    expected = " — expected: this is a negative control" if negative_control else ""
    return f"{len(verdict.problems)} problem(s){expected}"


def _file_name(test_id):
    """Turn `tests/node/test_x.py::test_y[3]` into `node.test_x__test_y-3`."""
    path, _, name = test_id.partition("::")
    parts = path.removesuffix(".py").split("/")
    if parts[0] == "tests":
        parts = parts[1:]
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", ".".join(parts) + "__" + name).strip("-")
