"""Runtime-test fixtures: clusters of nodes that run themselves, re-checked from their trace."""

import pytest

from tests.runtime.runtime_cluster import RuntimeCluster
from tests.traces.recorder import failing_on_broken_rules


@pytest.fixture
async def start_runtime_cluster(tmp_path, request):
    """Return a function that starts a RuntimeCluster; every cluster is stopped at teardown.

    The function takes the member IDs and, by keyword, RuntimeCluster's `timing` and `seed`.
    The test's election trace is re-checked at teardown, and the test fails if it breaks a rule,
    unless it is marked `negative_control`.
    """
    clusters = []

    async def start(member_ids, **options):
        directory = tmp_path / f"cluster-{len(clusters)}"
        directory.mkdir()
        cluster = RuntimeCluster(directory, member_ids, **options)
        clusters.append(cluster)
        await cluster.start_all()
        return cluster

    with failing_on_broken_rules(request.node):
        yield start
        for cluster in clusters:
            await cluster.stop_all()
