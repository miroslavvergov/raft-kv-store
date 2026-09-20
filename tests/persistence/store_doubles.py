"""Store doubles and disk helpers shared by the persistence tests.

Each store is a real SqliteStore that also records, fails, or holds in flight every write,
before or after it commits. NoLock stands in for DD-8's lock in negative controls.
"""

import asyncio

from raftkv.consensus import Role
from raftkv.persistence import SqliteStore
from tests.divergent_logs import make_log
from tests.vote_messages import granted


async def reload(path):
    """Reopen the file from scratch, as a restarted node would, and load it."""
    async with SqliteStore(path) as store:
        return await store.load()


async def term_and_vote_on_disk(path):
    """Return the (current_term, voted_for) a freshly reopened file holds."""
    persisted = await reload(path)
    return persisted.current_term, persisted.voted_for


async def seed_log(path, terms):
    """Write a log with the given entry terms straight to a node's file."""
    async with SqliteStore(path) as store:
        await store.replace_log_from(1, list(make_log(terms)))


async def let_other_tasks_run():
    """Give every ready task ten event-loop turns; no real time passes."""
    # NOTE: a task held on a write or a lock stays held through all ten turns, so a test can
    # assert it has not finished.
    for _ in range(10):
        await asyncio.sleep(0)


async def win_election(durable):
    """Start an election and grant it votes, lowest peer first, until the node leads.

    Returns:
        The election's RequestVote.
    """
    request = await durable.start_election()
    for peer in sorted(durable.peers):
        if durable.role is Role.LEADER:
            break
        await durable.handle_vote_response(peer, request.term, granted(term=request.term))
    assert durable.role is Role.LEADER
    return request


class RecordingStore(SqliteStore):
    """A real SqliteStore that also records every write it is asked to make.

    Attributes:
        writes: ("term_and_vote", term, vote) or ("replace_log_from", index, [terms]), in order.
    """

    def __init__(self, path):
        super().__init__(path)
        self.writes = []

    async def save_term_and_vote(self, current_term, voted_for):
        # NOTE: the write is recorded before it runs, so one that fails or never returns still
        # shows, and GatedStore can hold it here before it commits.
        await self._record(("term_and_vote", current_term, voted_for))
        await super().save_term_and_vote(current_term, voted_for)

    async def replace_log_from(self, index, entries):
        await self._record(("replace_log_from", index, [e.term for e in entries]))
        await super().replace_log_from(index, entries)

    async def _record(self, write):
        self.writes.append(write)


class FailingStore(SqliteStore):
    """A store whose every write fails, as a full or broken disk would."""

    async def save_term_and_vote(self, current_term, voted_for):
        raise OSError("disk full")

    async def replace_log_from(self, index, entries):
        raise OSError("disk full")


class GatedStore(RecordingStore):
    """A recording store that holds each write in flight, uncommitted, until `release` is set.

    A write is recorded, and `entered` set, as soon as it arrives, so a test knows exactly when
    one is pending instead of guessing with sleeps. `write_task` is the task running the latest
    write.
    """

    def __init__(self, path):
        super().__init__(path)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.write_task = None

    async def wait_for_write(self, timeout=5):
        """Wait until a write is being held; fail the test after `timeout` seconds without one."""
        await asyncio.wait_for(self.entered.wait(), timeout)

    async def _record(self, write):
        await super()._record(write)
        self.write_task = asyncio.current_task()
        # NOTE: neither event is ever cleared: `entered` stays set after the first write, and
        # once `release` is set every later write goes straight through.
        self.entered.set()
        await self.release.wait()


class CommitThenHoldStore(SqliteStore):
    """A store that commits each write, then holds it in flight until `release` is set.

    As a disk that is slow to confirm: `committed` is set once a write is on disk, and
    `write_task` is the task still running it.
    """

    def __init__(self, path):
        super().__init__(path)
        self.committed = asyncio.Event()
        self.release = asyncio.Event()
        self.write_task = None

    async def wait_for_commit(self, timeout=5):
        """Wait until a write is on disk; fail the test after `timeout` seconds without one."""
        await asyncio.wait_for(self.committed.wait(), timeout)

    async def save_term_and_vote(self, current_term, voted_for):
        await super().save_term_and_vote(current_term, voted_for)
        await self._hold()

    async def replace_log_from(self, index, entries):
        await super().replace_log_from(index, entries)
        await self._hold()

    async def _hold(self):
        self.write_task = asyncio.current_task()
        self.committed.set()
        await self.release.wait()


class NoLock:
    """Stands in for DD-8's lock in a negative control: never blocks anyone."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False
