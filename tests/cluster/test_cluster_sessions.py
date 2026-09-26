"""Tier 2 tests that a client's retried request takes effect once on every node, whatever happens.

The retry is a new log entry: every node applies the log in order, recognizes the repeat by its
session and request number, and skips it, each on its own, so all reach the same map. The
harness also asserts, after every step, that no node applies a request twice, that a request
taking no effect leaves the map alone, and that nodes at the same point hold the same state.
FAIL-4, FAIL-5, FAIL-6, APPLY-6, DD-15, DD-32.
"""

import pytest

from raftkv.consensus import Log, LogEntry
from raftkv.kvstore import OpenSession, Put
from tests.support.kv_client import KvClient


def times_in_log(node, command):
    """Return how many entries of `node`'s log carry `command`."""
    return sum(entry.command == command for entry in node.log)


async def test_a_retried_put_does_not_undo_a_later_put_on_any_node(three_nodes):
    # A's put commits, but A never hears so; B then puts x=2; A's retry arrives last.
    a = KvClient(await three_nodes.open_session(1))
    b = KvClient(await three_nodes.open_session(1))
    await three_nodes.commit_everywhere(1, a.put("x", "1"))
    await three_nodes.commit_everywhere(1, b.put("x", "2"))

    await three_nodes.commit_everywhere(1, a.retry())

    assert three_nodes.maps() == {n: {"x": "2"} for n in (1, 2, 3)}
    assert three_nodes.times_applied(a.client_id, 1) == {1: 1, 2: 1, 3: 1}
    assert times_in_log(three_nodes.nodes[1], a.last) == 2  # the retry is in the log too


async def test_a_retry_through_a_new_leader_takes_effect_once_when_the_first_attempt_committed(
    three_nodes,
):
    # The Leader commits A's put on itself and node 2, then crashes before answering; B's put
    # lands before A's retry, so a retry applied again would show in the map.
    a = KvClient(await three_nodes.open_session(1))
    b = KvClient(await three_nodes.open_session(1))
    await three_nodes.append_command(1, a.put("x", "1"))
    await three_nodes.replicate(1, 2)
    assert three_nodes.nodes[1].commit_index == 4
    await three_nodes.stop(1)

    await three_nodes.run_election(2)
    await three_nodes.commit_everywhere(2, b.put("x", "2"))
    await three_nodes.commit_everywhere(2, a.retry())

    assert three_nodes.maps() == {2: {"x": "2"}, 3: {"x": "2"}}
    assert three_nodes.times_applied(a.client_id, 1) == {2: 1, 3: 1}
    assert times_in_log(three_nodes.nodes[2], a.last) == 2


async def test_a_retry_takes_effect_once_when_the_first_attempt_was_lost_with_its_leader(
    three_nodes,
):
    # A's put reaches only the Leader, which crashes; the retry is the first copy to commit.
    a = KvClient(await three_nodes.open_session(1))
    first = a.put("x", "1")
    await three_nodes.append_command(1, first)
    await three_nodes.stop(1)
    await three_nodes.run_election(2)
    await three_nodes.commit_everywhere(2, a.retry())

    await three_nodes.start(1)  # its uncommitted copy is overwritten by node 2's log
    await three_nodes.commit_everywhere(2, a.put("y", "2"))

    assert three_nodes.maps() == {n: {"x": "1", "y": "2"} for n in (1, 2, 3)}
    assert three_nodes.times_applied(a.client_id, 1) == {1: 1, 2: 1, 3: 1}
    assert times_in_log(three_nodes.nodes[1], first) == 1


async def test_a_late_copy_of_an_answered_request_is_refused_on_every_node(three_nodes):
    a = KvClient(await three_nodes.open_session(1))
    first = a.put("x", "1")
    await three_nodes.commit_everywhere(1, first)
    await three_nodes.commit_everywhere(1, a.put("x", "2"))

    await three_nodes.commit_everywhere(1, first)  # arrives after request 2 was applied

    assert three_nodes.maps() == {n: {"x": "2"} for n in (1, 2, 3)}
    assert three_nodes.times_applied(a.client_id, 1) == {1: 1, 2: 1, 3: 1}


async def test_a_restarted_node_skips_the_same_duplicates_when_it_replays_its_log(three_nodes):
    a = KvClient(await three_nodes.open_session(1))
    b = KvClient(await three_nodes.open_session(1))
    await three_nodes.commit_everywhere(1, a.put("x", "1"))
    await three_nodes.commit_everywhere(1, b.put("x", "2"))
    await three_nodes.commit_everywhere(1, a.retry())

    await three_nodes.restart(3)
    await three_nodes.replicate(1, 3)
    await three_nodes.apply_on(3)

    assert three_nodes.maps()[3] == {"x": "2"}
    assert three_nodes.times_applied(a.client_id, 1)[3] == 1


async def test_a_put_claiming_a_session_whose_opening_was_lost_is_refused(three_nodes):
    # A client may use its ID only once its OpenSession has committed: this one never did, and
    # its index went to the next Leader's empty entry, so no session carries that ID.
    await three_nodes.replicate_to_all(1)  # every node holds node 1's empty entry
    lost = await three_nodes.append_command(1, OpenSession().encode())
    await three_nodes.stop(1)
    await three_nodes.run_election(2)
    assert three_nodes.nodes[2].log.entry_at(lost.index).is_empty

    await three_nodes.commit_everywhere(2, Put(lost.index, 1, "x", "1").encode())

    assert three_nodes.maps() == {2: {}, 3: {}}
    assert all(three_nodes.kv[n].session(lost.index) is None for n in three_nodes.nodes)


# --- Expiry: the same sessions forgotten at the same entry, everywhere (DD-15, DD-32) -----


async def test_an_idle_session_expires_at_the_same_entry_on_every_node_and_on_replay(
    start_cluster,
):
    cluster = await start_cluster([1, 2, 3], session_timeout=10)
    await cluster.run_election(1)
    a = KvClient(await cluster.open_session(1))
    await cluster.commit_everywhere(1, a.put("x", "1"))
    cluster.advance_cluster_time(1, 11)  # more than the timeout passes on the Leader's clock

    # B's opening is stamped 11. Every node holds A's session until it applies that entry,
    # and none holds it after.
    b_opening = await cluster.append_command(1, OpenSession().encode())
    await cluster.replicate_to_all(1)
    await cluster.replicate_to_all(1)
    assert all(node.last_applied == b_opening.index - 1 for node in cluster.nodes.values())
    assert all(cluster.kv[n].session(a.client_id) is not None for n in cluster.nodes)
    await cluster.apply_everywhere()
    assert all(node.last_applied == b_opening.index for node in cluster.nodes.values())
    assert all(cluster.kv[n].session(a.client_id) is None for n in cluster.nodes)

    b = KvClient(b_opening.index)
    await cluster.commit_everywhere(1, b.put("x", "2"))
    await cluster.commit_everywhere(1, a.retry())  # too late: refused, and x stays "2"

    for kv in cluster.kv.values():
        assert kv.session(b.client_id) is not None
        assert kv.as_dict() == {"x": "2"}
    assert cluster.times_applied(a.client_id, 1) == {1: 1, 2: 1, 3: 1}

    await cluster.restart(2)
    await cluster.replicate(1, 2)
    await cluster.apply_on(2)
    assert cluster.kv[2].session(a.client_id) is None
    assert cluster.kv[2].sessions == cluster.kv[1].sessions


# --- The harness's own checks can fail -------------------------------------------------


@pytest.mark.negative_control
async def test_negative_control_a_node_applying_a_retry_again_is_caught(three_nodes, monkeypatch):
    # Node 3's store forgets the request number just before the retry arrives, so it applies
    # the retry a second time. Reaching into its session table is the point of the control.
    a = KvClient(await three_nodes.open_session(1))
    await three_nodes.commit_everywhere(1, a.put("x", "1"))
    store = three_nodes.kv[3]
    apply_once = store.apply

    def apply_forgetting_the_number(index, cluster_time, command):
        if command == a.last:
            store._sessions.record(a.client_id, 0, None, cluster_time)
        return apply_once(index, cluster_time, command)

    monkeypatch.setattr(store, "apply", apply_forgetting_the_number)
    with pytest.raises(AssertionError, match="applied a client request twice"):
        await three_nodes.commit_everywhere(1, a.retry())


@pytest.mark.negative_control
async def test_negative_control_a_retry_that_changes_the_map_is_caught(three_nodes, monkeypatch):
    # Node 3 answers the retry with the stored result, but writes the value again first.
    a = KvClient(await three_nodes.open_session(1))
    b = KvClient(await three_nodes.open_session(1))
    await three_nodes.commit_everywhere(1, a.put("x", "1"))
    await three_nodes.commit_everywhere(1, b.put("x", "2"))
    store = three_nodes.kv[3]
    apply_once = store.apply

    def rewrite_then_apply(index, cluster_time, command):
        if command == a.last:
            store._values["x"] = "1"
        return apply_once(index, cluster_time, command)

    monkeypatch.setattr(store, "apply", rewrite_then_apply)
    with pytest.raises(AssertionError, match="which took no effect"):
        await three_nodes.commit_everywhere(1, a.retry())


@pytest.mark.negative_control
async def test_negative_control_a_falling_cluster_time_is_caught(three_nodes):
    # Forged through _log: no correct path can build a log whose times fall.
    three_nodes.nodes[3]._log = Log([LogEntry(1, "", 5), LogEntry(1, "x", 3)])
    with pytest.raises(AssertionError, match="cluster times fall"):
        three_nodes.assert_log_safety()
