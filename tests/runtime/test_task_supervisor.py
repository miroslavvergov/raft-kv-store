"""Tests for TaskSupervisor: a node's tasks and calls in progress, and stopping both."""

import asyncio

import pytest

from raftkv.runtime import NodeStoppedError
from raftkv.runtime.task_supervisor import TaskSupervisor
from tests.support.store_doubles import let_other_tasks_run
from tests.support.waiting import within_bound


async def forever():
    await asyncio.Event().wait()


async def test_a_task_started_after_closing_never_runs():
    supervisor = TaskSupervisor()
    await supervisor.close()
    ran = []

    async def work():
        ran.append(True)

    assert supervisor.spawn(work()) is None
    await let_other_tasks_run()
    assert ran == []


async def test_closing_cancels_every_task_counted_or_not():
    supervisor = TaskSupervisor()
    counted = supervisor.spawn(forever())
    clock = supervisor.spawn(forever(), counted=False)

    await within_bound(supervisor.close())

    assert counted.cancelled() and clock.cancelled()
    assert not supervisor.busy


async def test_idle_waits_for_tasks_started_while_waiting():
    supervisor = TaskSupervisor()
    done = []

    async def second():
        done.append("second")

    async def first():
        supervisor.spawn(second())
        done.append("first")

    supervisor.spawn(first())
    await within_bound(supervisor.idle())

    assert done == ["first", "second"]


async def test_a_task_that_raises_is_kept_as_the_failure_and_closes_the_supervisor():
    supervisor = TaskSupervisor()
    other = supervisor.spawn(forever())

    async def broken():
        raise ValueError("broken")

    supervisor.spawn(broken())
    await within_bound(supervisor.idle())

    assert isinstance(supervisor.failure, ValueError)
    assert supervisor.closed and other.cancelled()


async def test_a_failure_is_kept_even_when_closing_began_first():
    # The closing task's first step is queued before the failed task's done-callback, so
    # closing has begun by the time the failure is reported.
    supervisor = TaskSupervisor()

    async def broken():
        raise ValueError("broken")

    supervisor.spawn(broken())
    await within_bound(asyncio.create_task(supervisor.close()))

    assert isinstance(supervisor.failure, ValueError)


async def test_only_the_first_failure_is_kept():
    supervisor = TaskSupervisor()
    supervisor.fail(ValueError("first"))
    supervisor.fail(KeyError("second"))
    assert str(supervisor.failure) == "first"


async def test_a_call_is_refused_once_closed():
    supervisor = TaskSupervisor()
    await supervisor.close()
    with pytest.raises(NodeStoppedError), supervisor.call():
        pass


async def test_closing_waits_for_every_call_in_progress():
    supervisor = TaskSupervisor()
    release = asyncio.Event()

    async def call():
        with supervisor.call():
            await release.wait()

    calling = asyncio.create_task(call())
    await let_other_tasks_run()
    closing = asyncio.create_task(supervisor.close())
    await let_other_tasks_run()
    assert not closing.done()

    release.set()
    await within_bound(asyncio.gather(calling, closing))
