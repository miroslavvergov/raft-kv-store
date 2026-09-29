"""The tasks a running node owns, the calls it is serving, and stopping both."""

import asyncio
import contextlib
from collections.abc import Callable, Coroutine, Iterator
from typing import Any

from raftkv.runtime.errors import NodeStoppedError


class TaskSupervisor:
    """Runs a node's background tasks and counts its calls in progress, until closed.

    A task's unexpected error is kept as `failure` and closes the supervisor, so a
    node never runs on after a part it depends on has failed. Once closed, no new
    task starts and no new call is admitted. `close` cancels every task and waits
    for them and for every call still in progress, so nothing uses the node's
    store once it returns. It calls `on_close` once, as it first closes.

    Attributes:
        closed: Whether the supervisor is closed: `close` was called, or something
            failed.
        failure: The first unexpected error of a task, or one reported through
            `fail`; None if there was none.
        busy: Whether a counted task is still running.
    """

    def __init__(self, on_close: Callable[[], None] | None = None) -> None:
        """Create an open supervisor with no tasks and no calls in progress.

        Args:
            on_close: Called once, synchronously, as the supervisor first closes,
                whether `close` was called or something failed.
        """
        self._on_close = on_close
        self._counted: set[asyncio.Task] = set()
        self._uncounted: set[asyncio.Task] = set()
        self._calls_in_progress = 0
        self._no_calls = asyncio.Event()
        self._no_calls.set()
        self._closed = False
        self._failure: BaseException | None = None

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def failure(self) -> BaseException | None:
        return self._failure

    @property
    def busy(self) -> bool:
        return bool(self._counted)

    def spawn(
        self, coroutine: Coroutine[Any, Any, None], *, counted: bool = True
    ) -> asyncio.Task | None:
        """Run `coroutine` as a task, unless closed.

        Args:
            coroutine: The work to run.
            counted: Whether `busy` and `idle` wait for the task; False for a task
                that never ends on its own, such as a clock.

        Returns:
            The task, or None if the supervisor is closed; the coroutine is then
            closed without running.
        """
        if self._closed:
            coroutine.close()
            return None
        task = asyncio.get_running_loop().create_task(coroutine)
        (self._counted if counted else self._uncounted).add(task)
        task.add_done_callback(self._task_done)
        return task

    @contextlib.contextmanager
    def call(self) -> Iterator[None]:
        """Count one call in progress for as long as the `with` block runs.

        Raises:
            NodeStoppedError: If the supervisor is closed.
        """
        self.check_open()
        self._calls_in_progress += 1
        self._no_calls.clear()
        try:
            yield
        finally:
            self._calls_in_progress -= 1
            if self._calls_in_progress == 0:
                self._no_calls.set()

    def check_open(self) -> None:
        """Raise unless the supervisor is open.

        Raises:
            NodeStoppedError: If it is closed.
        """
        if self._closed:
            raise NodeStoppedError("the node is stopped")

    def fail(self, error: BaseException) -> None:
        """Keep `error` as the failure, unless one is kept already, and close.

        Tasks are cancelled at once; calls in progress run to their end.
        """
        if self._failure is None:
            self._failure = error
        self._begin_closing()

    async def idle(self) -> None:
        """Wait until no counted task is left, including any started meanwhile."""
        while self._counted:
            await asyncio.wait(set(self._counted))

    async def close(self) -> None:
        """Close, cancel every task, and wait for the tasks and every call in progress.

        Must not be awaited from within a call this supervisor counts, which would
        wait for itself.
        """
        self._begin_closing()
        tasks = [*self._counted, *self._uncounted]
        await asyncio.gather(*tasks, return_exceptions=True)
        await self._no_calls.wait()

    def _begin_closing(self) -> None:
        first_time = not self._closed
        self._closed = True
        for task in [*self._counted, *self._uncounted]:
            task.cancel()
        if first_time and self._on_close is not None:
            self._on_close()

    def _task_done(self, task: asyncio.Task) -> None:
        self._counted.discard(task)
        self._uncounted.discard(task)
        # NOTE: kept even once closed: an error a task raised before `close` cancelled it is
        # still the reason the node could not go on.
        if not task.cancelled() and task.exception() is not None:
            self.fail(task.exception())
