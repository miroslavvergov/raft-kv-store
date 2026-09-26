"""Cluster-test fixtures: a three-node cluster with a Leader already elected."""

import pytest


@pytest.fixture
async def three_nodes(start_cluster):
    """Return a running cluster of fresh nodes 1, 2, and 3, with node 1 elected Leader."""
    cluster = await start_cluster([1, 2, 3])
    await cluster.run_election(1)
    assert cluster.leaders() == {1}
    return cluster
