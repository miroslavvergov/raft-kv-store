"""Waiting in tests with a bound, so broken code fails a test instead of hanging it."""

import asyncio

# NOTE: far above anything a passing test needs, and far below a hung run's patience.
BOUND_SECONDS = 5.0


async def eventually(condition, *, timeout=BOUND_SECONDS):
    """Wait until `condition()` holds, checking every millisecond.

    Waits in real time, not bare turns of the event loop, since a write finishes on
    aiosqlite's thread.

    Raises:
        AssertionError: If `condition()` does not hold within `timeout` seconds.
    """
    try:
        async with asyncio.timeout(timeout):
            while not condition():
                await asyncio.sleep(0.001)
    except TimeoutError:
        raise AssertionError(f"condition did not hold within {timeout} seconds") from None


async def within_bound(awaitable, *, timeout=BOUND_SECONDS):
    """Await `awaitable`, failing the test if it takes longer than `timeout` seconds.

    Raises:
        AssertionError: If it does not finish in time.
    """
    try:
        async with asyncio.timeout(timeout):
            return await awaitable
    except TimeoutError:
        raise AssertionError(f"did not finish within {timeout} seconds") from None
