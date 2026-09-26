"""Tier 2 tests that whole clusters of real nodes end up with the same key-value map.

This is what the rest of the system exists to provide: a command a client gives one node is
committed, applied on every node in the same order, and survives crashes and leader changes.
APPLY-4, APPLY-5, APPLY-6, APPLY-7, DD-12, DD-26, DD-28.
"""

import pathlib

import pytest

from raftkv.consensus import Log, LogEntry, Role
from raftkv.kvstore import Put
from tests.support.kv_client import KvClient


@pytest.fixture
async def client(three_nodes):
    """Return a client whose session node 1 has committed; its entry is index 2."""
    return KvClient(await three_nodes.open_session(1))


# --- The whole point --------------------------------------------------------------------


async def test_a_command_given_to_the_leader_reaches_every_nodes_map(three_nodes, client):
    await three_nodes.commit_everywhere(1, client.put("x", "5"))

    assert three_nodes.maps() == {1: {"x": "5"}, 2: {"x": "5"}, 3: {"x": "5"}}


async def test_every_node_applies_the_same_commands_in_the_same_order(three_nodes, client):
    for key, value in (("x", "1"), ("y", "2"), ("x", "3")):
        await three_nodes.commit_everywhere(1, client.put(key, value))

    # The last write to x wins everywhere, which only holds if the order was the same.
    assert three_nodes.maps() == {n: {"x": "3", "y": "2"} for n in (1, 2, 3)}


async def test_a_node_that_was_down_catches_up_and_ends_with_the_same_map(three_nodes, client):
    await three_nodes.stop(3)
    await three_nodes.commit_everywhere(1, client.put("x", "5"))
    await three_nodes.commit_everywhere(1, client.put("y", "6"))
    assert three_nodes.maps() == {1: {"x": "5", "y": "6"}, 2: {"x": "5", "y": "6"}}

    await three_nodes.start(3)
    assert three_nodes.maps()[3] == {}  # it comes back with nothing applied
    await three_nodes.replicate(1, 3)
    await three_nodes.replicate(1, 3)
    await three_nodes.apply_on(3)

    assert three_nodes.maps()[3] == {"x": "5", "y": "6"}


async def test_a_restarted_node_rebuilds_its_map_by_replaying_its_log(three_nodes, client):
    await three_nodes.commit_everywhere(1, client.put("x", "5"))
    assert three_nodes.maps()[2] == {"x": "5"}

    await three_nodes.restart(2)
    assert (three_nodes.maps()[2], three_nodes.nodes[2].last_applied) == ({}, 0)

    await three_nodes.replicate(1, 2)
    await three_nodes.apply_on(2)
    assert three_nodes.maps()[2] == {"x": "5"}


async def test_a_committed_command_survives_a_change_of_leader(three_nodes, client):
    await three_nodes.commit_everywhere(1, client.put("x", "5"))
    await three_nodes.stop(1)

    await three_nodes.run_election(2)
    await three_nodes.commit_everywhere(2, client.put("y", "6"))

    assert three_nodes.maps() == {2: {"x": "5", "y": "6"}, 3: {"x": "5", "y": "6"}}


async def test_the_leaders_empty_entry_never_reaches_a_state_machine(three_nodes, client):
    # Every node's log starts with node 1's empty entry, and no map ever holds it.
    await three_nodes.commit_everywhere(1, client.put("x", "5"))

    assert three_nodes.nodes[1].log.entry_at(1).is_empty
    assert all(node.last_applied == 3 for node in three_nodes.nodes.values())
    assert all(map_ == {"x": "5"} for map_ in three_nodes.maps().values())


async def test_a_leader_answers_no_read_until_it_has_committed_in_its_own_term(three_nodes, client):
    # CLIENT-10 at cluster level: node 2 wins with node 1's committed entry in its log, but
    # may not answer a read until its own empty entry commits.
    await three_nodes.commit_everywhere(1, client.put("x", "5"))
    await three_nodes.stop(1)
    await three_nodes.run_election(2)

    assert three_nodes.nodes[2].role is Role.LEADER
    assert three_nodes.nodes[2].has_committed_in_current_term is False

    await three_nodes.replicate(2, 3)
    assert three_nodes.nodes[2].has_committed_in_current_term is True


@pytest.mark.negative_control
async def test_negative_control_applying_past_the_commit_index_diverges(
    three_nodes, client, monkeypatch
):
    # Proves the harness notices a broken apply loop: node 3 applies everything in its log,
    # committed or not (APPLY-5 removed). It applies an entry the cluster has not committed, so
    # its last_applied runs past its commit index and the harness catches that.
    # It reaches into the node's privates on purpose: the replacement is apply_committed with
    # the commit bound removed, which is the whole point of the control.
    await three_nodes.append_command(1, client.put("x", "5"))
    await three_nodes.replicate(1, 3)  # node 3 stores it, uncommitted

    async def apply_everything(node):
        while node._last_applied < node.log.last_index:
            entry = node.log.entry_at(node._last_applied + 1)
            if not entry.is_empty:
                node._apply(node._last_applied + 1, entry.cluster_time, entry.command)
            node._last_applied += 1
        return 0

    monkeypatch.setattr(type(three_nodes.nodes[3]), "apply_committed", apply_everything)
    with pytest.raises(AssertionError, match="applied through"):
        await three_nodes.apply_on(3)


@pytest.mark.negative_control
async def test_negative_control_two_nodes_applying_different_commands_is_caught(
    three_nodes, client
):
    # Proves the harness's applied-command check can fire. Reaching into _log and
    # _last_applied is deliberate: no public path can forge a log that disagrees at an index
    # another node has already applied, which is exactly what this check exists to catch.
    await three_nodes.commit_everywhere(1, client.put("x", "5"))
    node = three_nodes.nodes[3]
    forged = LogEntry(1, Put(client.client_id, 2, "y", "9").encode())
    node._log = Log([node.log.entry_at(1), node.log.entry_at(2), forged])
    node._last_applied = 2

    with pytest.raises(AssertionError, match="at index 3, where"):
        await three_nodes.apply_on(3)


def test_the_raft_layer_never_imports_the_kv_layer():
    # DD-12: the Raft layers reach the state machine only through the callback they were
    # handed, so none may import the KV Store layer.
    root = pathlib.Path(__file__).resolve().parents[2] / "raftkv"
    raft_packages = [p for p in root.iterdir() if p.is_dir() and (p / "__init__.py").exists()]
    offenders = [
        f"{source.relative_to(root.parent)}: {line.strip()}"
        for package in raft_packages
        if package.name != "kvstore"
        for source in package.glob("*.py")
        for line in source.read_text().splitlines()
        if line.startswith(("import ", "from ")) and "kvstore" in line
    ]
    assert offenders == []
