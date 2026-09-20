"""Persistence-test fixtures: a node's file, clusters, and a lock that never blocks."""

import pytest

from raftkv.persistence import DurableNodeState
from tests.election_traces.checker import check_election_trace
from tests.election_traces.recorder import collecting_trace_records, render
from tests.persistence.in_process_cluster import InProcessCluster
from tests.persistence.store_doubles import NoLock


@pytest.fixture
def db_path(tmp_path):
    """Return the path of a node's SQLite file, not yet created."""
    return str(tmp_path / "node.db")


@pytest.fixture
def without_the_lock(monkeypatch):
    """Replace every new DurableNodeState's DD-8 lock with one that never blocks.

    For negative controls only: the one place tests touch a private attribute.
    """
    original_init = DurableNodeState.__init__

    def init_without_lock(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._lock = NoLock()

    # NOTE: patching the class, not one instance, strips the lock from every node a helper or
    # a restart builds later, since `load` builds them out of the test's reach.
    monkeypatch.setattr(DurableNodeState, "__init__", init_without_lock)


@pytest.fixture
async def start_cluster(tmp_path, request):
    """Return a function that starts an InProcessCluster; every cluster is stopped at teardown.

    The function takes the member IDs and, by keyword: `preload`, each node's starting file as
    InProcessCluster.preload's arguments; `down`, the nodes left stopped; and
    `check_votes_on_disk`. The test's election trace is re-checked at teardown, and the test
    fails if it breaks a rule, unless it is marked `negative_control`.
    """
    clusters = []

    async def start(member_ids, *, preload=None, down=(), check_votes_on_disk=True):
        cluster = InProcessCluster(tmp_path, member_ids, check_votes_on_disk=check_votes_on_disk)
        clusters.append(cluster)
        for node_id, starting_file in (preload or {}).items():
            await cluster.preload(node_id, **starting_file)
        for node_id in cluster.member_ids:
            if node_id not in down:
                await cluster.start(node_id)
        return cluster

    # NOTE: the trace is collected and re-checked here too, so every cluster test is checked
    # without --trace-elections.
    with collecting_trace_records() as records:
        yield start
        for cluster in clusters:
            await cluster.stop_all()
    if request.node.get_closest_marker("negative_control") is not None:
        return
    _, entries = render(records)
    problems = check_election_trace(entries).problems
    if problems:
        pytest.fail(
            "the election trace breaks a safety rule:\n  - " + "\n  - ".join(problems),
            pytrace=False,
        )
