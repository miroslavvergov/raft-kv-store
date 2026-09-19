"""Tier 1 tests for `traced`: arguments reach the method and its report however they are passed.

Only coroutine methods and `__init__` can be decorated, and a failing report never changes a call.
"""

import asyncio
import logging

import pytest

from raftkv.tracing import NodeSnapshot, NodeTracer, traced


def node_class(report):
    """Return a node class whose coroutine methods `act` and `wait_forever` are traced."""

    class Node:
        node_id = 7

        @traced(report)
        async def act(self, first, second, third=3):
            return first + second + third

        @traced(report)
        async def wait_forever(self):
            await asyncio.Event().wait()

    return Node


def failing_report(*args, **kwargs):
    """Stand in for a report with a bug: always raise."""
    raise RuntimeError("the report has a bug")


async def test_the_report_gets_every_argument_positionally_however_it_was_passed(
    tracing_on, monkeypatch
):
    monkeypatch.setattr(NodeSnapshot, "of", lambda node: "snapshot")
    reports = []

    def report(tracer, before, after, *args, result, error):
        reports.append((args, result, error))

    node = node_class(report)()
    assert await node.act(1, second=2) == 6
    assert await node.act(first=1, second=2, third=4) == 7
    assert reports == [((1, 2, 3), 6, None), ((1, 2, 4), 7, None)]


async def test_with_tracing_off_keyword_arguments_still_reach_the_method(tracing_off):
    def report(*args, **kwargs):
        raise AssertionError("reported while tracing was off")

    node = node_class(report)()
    assert await node.act(first=1, second=2) == 6


def test_a_synchronous_method_other_than_init_cannot_be_decorated():
    def step(node):
        return None

    with pytest.raises(TypeError, match="step"):
        traced(NodeTracer.report_started)(step)


async def test_a_report_that_raises_leaves_the_result_unchanged(tracing_on, monkeypatch):
    monkeypatch.setattr(NodeSnapshot, "of", lambda node: "snapshot")
    node = node_class(failing_report)()
    assert await node.act(1, second=2) == 6
    [failure] = [r for r in tracing_on.records if r.levelno == logging.ERROR]
    assert failure.getMessage() == "reporting with failing_report failed"


async def test_a_report_that_raises_leaves_a_cancellation_unchanged(tracing_on, monkeypatch):
    monkeypatch.setattr(NodeSnapshot, "of", lambda node: "snapshot")
    waiting = asyncio.create_task(node_class(failing_report)().wait_forever())
    await asyncio.sleep(0)  # let it start waiting
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
