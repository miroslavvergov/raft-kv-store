"""Project-wide pytest setup: shared fixtures and the `--trace-elections` option.

With `--trace-elections`, every test that runs a node writes its trace to `test-traces/elections/`,
and the checker (tests/traces/checker.py) re-checks it. A passing test whose trace breaks
a rule fails at teardown, unless it is marked `negative_control`.
"""

import logging

import pytest

from raftkv.tracing import LOG_LINES_LOGGER, TRACE_EVENTS_LOGGER
from tests.cluster.in_process_cluster import InProcessCluster
from tests.traces.recorder import ElectionTraceRecorder, failing_on_broken_rules

_RECORDER = pytest.StashKey[ElectionTraceRecorder]()
_OUTCOME = pytest.StashKey[str]()


@pytest.fixture
def db_path(tmp_path):
    """Return the path of a node's SQLite file, not yet created."""
    return str(tmp_path / "node.db")


@pytest.fixture
async def start_cluster(tmp_path, request):
    """Return a function that starts an InProcessCluster; every cluster is stopped at teardown.

    The function takes the member IDs and, by keyword: `preload`, each node's starting file as
    InProcessCluster.preload's arguments; `down`, the nodes left stopped; and InProcessCluster's
    `check_votes_on_disk` and `session_timeout`. The test's election trace is re-checked at
    teardown, and the test fails if it breaks a rule, unless it is marked `negative_control`.
    """
    clusters = []

    async def start(member_ids, *, preload=None, down=(), **options):
        cluster = InProcessCluster(tmp_path, member_ids, **options)
        clusters.append(cluster)
        for node_id, starting_file in (preload or {}).items():
            await cluster.preload(node_id, **starting_file)
        for node_id in cluster.member_ids:
            if node_id not in down:
                await cluster.start(node_id)
        return cluster

    with failing_on_broken_rules(request.node):
        yield start
        for cluster in clusters:
            await cluster.stop_all()


def _set_tracing_level(caplog, level):
    caplog.set_level(level, logger=LOG_LINES_LOGGER)
    caplog.set_level(level, logger=TRACE_EVENTS_LOGGER)
    return caplog


@pytest.fixture
def tracing_on(caplog):
    """Enable the nodes' log lines and trace events; return `caplog`, which captures them."""
    return _set_tracing_level(caplog, logging.DEBUG)


@pytest.fixture
def tracing_off(caplog):
    """Disable the nodes' log lines and trace events; return `caplog`."""
    return _set_tracing_level(caplog, logging.WARNING)


@pytest.fixture(params=["tracing on", "tracing off"])
def tracing_on_or_off(request, caplog):
    """Run the test once with tracing on and once with it off; return `caplog`."""
    return _set_tracing_level(
        caplog, logging.DEBUG if request.param == "tracing on" else logging.WARNING
    )


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
    # NOTE: recording starts before pytest's own setup hook, so the nodes a fixture starts
    # while setting up are already in the test's trace.
    if _RECORDER in item.config.stash:
        item.config.stash[_RECORDER].begin()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    """Remember the call phase's outcome, or an earlier phase's failure or skip."""
    report = yield
    if report.when == "call" or report.outcome != "passed":
        item.stash[_OUTCOME] = report.outcome
    return report


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item, nextitem):
    """Write the test's trace after its own teardown, and fail it if the trace breaks a rule.

    Only a passing test that is not a negative control, and whose own teardown raised nothing,
    is failed; an error from that teardown is never replaced.
    """
    if _RECORDER not in item.config.stash:
        return (yield)
    # NOTE: the finally writes the trace even when the test's own teardown raises, which then
    # propagates past the check below.
    try:
        result = yield
    finally:
        outcome = item.stash.get(_OUTCOME, "passed")
        negative_control = item.get_closest_marker("negative_control") is not None
        verdict = item.config.stash[_RECORDER].finish(item.nodeid, outcome, negative_control)
    if verdict and verdict.problems and not negative_control and outcome == "passed":
        pytest.fail(
            "the election trace breaks a safety rule:\n  - " + "\n  - ".join(verdict.problems),
            pytrace=False,
        )
    return result


def pytest_terminal_summary(terminalreporter, config):
    if _RECORDER not in config.stash:
        return
    recorder = config.stash[_RECORDER]
    results = recorder.results
    clean = [r for r in results if not r.verdict.problems]
    expected = [r for r in results if r.verdict.problems and r.negative_control]
    unexpected = [r for r in results if r.verdict.problems and not r.negative_control]
    silent_controls = [r for r in results if r.negative_control and not r.verdict.problems]

    terminalreporter.section("election traces")
    terminalreporter.write_line(
        f"{len(results)} traces written to {recorder.directory} (.log to read, .jsonl for tools)"
    )
    terminalreporter.write_line(f"checker: {len(clean)} traces break no rule")
    for result in expected:
        terminalreporter.write_line(
            f"checker: negative control {result.test_name} "
            "shows the violation it was built to cause:"
        )
        for problem in result.verdict.problems:
            terminalreporter.write_line(f"           - {problem}")
    for result in silent_controls:
        terminalreporter.write_line(
            f"checker: negative control {result.test_name} leaves no violation visible in its trace"
        )
    for result in unexpected:
        terminalreporter.write_line(
            f"checker: UNEXPECTED problems in {result.test_id} — see {result.log_path}", red=True
        )
