"""Stand-ins for a node's store and lock, shared by the persistence tests.

Each store is a real SqliteStore with one behavior added: recording every
write, failing every write, or holding every write in flight until the
test releases it. NoLock replaces DD-8's lock in negative controls.
"""

import asyncio

from raftkv.persistence import SqliteStore
from tests.divergent_logs import make_log


async def reload(path):
    """Reopen the file from scratch, as a restarted node would, and load it."""
    async with SqliteStore(path) as store:
        return await store.load()


async def seed_log(path, terms):
    """Write a log with the given entry terms straight to a node's file."""
    async with SqliteStore(path) as store:
        await store.save_log_from(1, list(make_log(terms)))


class RecordingStore(SqliteStore):
    """A real SqliteStore that also records every write it is asked to make."""

    def __init__(self, path):
        super().__init__(path)
        self.writes = []

    async def save_term_and_vote(self, current_term, voted_for):
        self.writes.append(("term_and_vote", current_term, voted_for))
        await super().save_term_and_vote(current_term, voted_for)

    async def save_log_from(self, index, entries):
        self.writes.append(("log_from", index, [e.term for e in entries]))
        await super().save_log_from(index, entries)


class FailingStore(SqliteStore):
    """A store whose every write fails, as a full or broken disk would."""

    async def save_term_and_vote(self, current_term, voted_for):
        raise OSError("disk full")

    async def save_log_from(self, index, entries):
        raise OSError("disk full")


class GatedStore(SqliteStore):
    """A store that holds each write in flight until `release` is set.

    `entered` is set as soon as a write arrives, so a test knows exactly
    when one is pending instead of guessing with sleeps.
    """

    def __init__(self, path):
        super().__init__(path)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.writes = []

    async def wait_for_write(self, timeout=5):
        """Wait until a write is being held.

        Returns as soon as one arrives. If none arrives — code that skips
        its write — the test fails after `timeout` seconds instead of
        waiting forever.
        """
        await asyncio.wait_for(self.entered.wait(), timeout)

    async def save_term_and_vote(self, current_term, voted_for):
        self.writes.append((current_term, voted_for))
        self.entered.set()
        await self.release.wait()
        await super().save_term_and_vote(current_term, voted_for)

    async def save_log_from(self, index, entries):
        self.writes.append(("log_from", index))
        self.entered.set()
        await self.release.wait()
        await super().save_log_from(index, entries)


class NoLock:
    """Stands in for DD-8's lock in a negative control: never blocks anyone."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False
