"""Project-wide pytest setup: the `--trace-elections` option.

`pytest --trace-elections` records, for every test that runs a node, what
each node decided — in etcd's raft log format and as etcd-style trace
events — interleaved with everything the test harness did, and writes it to
`test-traces/elections/`. Each trace is then re-checked for election-safety
violations by an independent checker (tests/election_traces/checker.py). A
test whose trace breaks a rule fails at teardown, unless it is marked
`negative_control`, meaning it breaks one on purpose.

Without the option, nothing is recorded: the nodes' loggers stay disabled
and every tracing call returns after a single level check.
"""

import pytest

from tests.election_traces.recorder import ElectionTraceRecorder

_RECORDER = pytest.StashKey[ElectionTraceRecorder]()
_OUTCOME = pytest.StashKey[str]()


def pytest_addoption(parser):
    parser.addoption(
        "--trace-elections",
        action="store_true",
        default=False,
        help="write every node's decisions and every message of each test to "
        "test-traces/elections/, and re-check each trace for election-safety violations",
    )


def pytest_configure(config):
    if config.getoption("--trace-elections"):
        recorder = ElectionTraceRecorder(config.rootpath / "test-traces" / "elections")
        recorder.install()
        config.stash[_RECORDER] = recorder


def pytest_unconfigure(config):
    if _RECORDER in config.stash:
        config.stash[_RECORDER].uninstall()


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    if _RECORDER in item.config.stash:
        item.config.stash[_RECORDER].begin()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    if report.when == "call" or report.outcome != "passed":
        item.stash[_OUTCOME] = report.outcome
    return report


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item, nextitem):
    if _RECORDER not in item.config.stash:
        return (yield)
    try:
        return (yield)
    finally:
        outcome = item.stash.get(_OUTCOME, "passed")
        negative_control = item.get_closest_marker("negative_control") is not None
        verdict = item.config.stash[_RECORDER].finish(item.nodeid, outcome, negative_control)
        if verdict and verdict.problems and not negative_control and outcome == "passed":
            pytest.fail(
                "the election trace breaks a safety rule:\n  - "
                + "\n  - ".join(verdict.problems),
                pytrace=False,
            )


def pytest_terminal_summary(terminalreporter, config):
    if _RECORDER not in config.stash:
        return
    recorder = config.stash[_RECORDER]
    clean = [r for r in recorder.results if not r[1].problems]
    expected = [r for r in recorder.results if r[1].problems and r[2]]
    unexpected = [r for r in recorder.results if r[1].problems and not r[2]]
    controls_silent = [r for r in recorder.results if r[2] and not r[1].problems]

    terminalreporter.section("election traces")
    terminalreporter.write_line(
        f"{len(recorder.results)} traces written to {recorder.directory} "
        "(.log to read, .jsonl for tools)"
    )
    terminalreporter.write_line(f"checker: {len(clean)} traces break no rule")
    for test_id, verdict, _, _ in expected:
        terminalreporter.write_line(
            f"checker: negative control {test_id.split('::')[-1]} "
            "shows the violation it was built to cause:"
        )
        for problem in verdict.problems:
            terminalreporter.write_line(f"           - {problem}")
    for test_id, _, _, _ in controls_silent:
        terminalreporter.write_line(
            f"checker: negative control {test_id.split('::')[-1]} "
            "leaves no violation visible in its trace"
        )
    for test_id, verdict, _, path in unexpected:
        terminalreporter.write_line(
            f"checker: UNEXPECTED problems in {test_id} — see {path}", red=True
        )
