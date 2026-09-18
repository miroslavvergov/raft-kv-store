"""Tier 1 persistence round-trip tests for SqliteStore (PERSIST-1 through
PERSIST-6, DD-6, DD-7, DD-20, DD-21, DD-23): each of current_term,
voted_for, and the log written, then reloaded from a freshly reopened file,
and asserted on its own.
"""

import sqlite3
from contextlib import closing

import pytest

from raftkv.consensus import Log, LogEntry
from raftkv.persistence import SqliteStore
from tests.figure_7 import make_log


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "node.db")


async def reload(path):
    """Reopen the file from scratch, as a restarted node would, and load it."""
    async with SqliteStore(path) as store:
        return await store.load()


# --- A brand-new file --------------------------------------------------------


async def test_fresh_file_loads_term_zero_no_vote_and_empty_log(db_path):
    persisted = await reload(db_path)
    assert persisted.current_term == 0
    assert persisted.voted_for is None
    assert persisted.log == Log()


# --- current_term and voted_for, each on its own ----------------------------


async def test_current_term_survives_a_reopen(db_path):
    async with SqliteStore(db_path) as store:
        await store.save_term_and_vote(current_term=5, voted_for=None)
    assert (await reload(db_path)).current_term == 5


async def test_voted_for_survives_a_reopen(db_path):
    async with SqliteStore(db_path) as store:
        await store.save_term_and_vote(current_term=5, voted_for=3)
    assert (await reload(db_path)).voted_for == 3


async def test_a_cleared_vote_survives_a_reopen_as_none(db_path):
    async with SqliteStore(db_path) as store:
        await store.save_term_and_vote(current_term=5, voted_for=3)
        await store.save_term_and_vote(current_term=6, voted_for=None)
    persisted = await reload(db_path)
    assert persisted.current_term == 6
    assert persisted.voted_for is None


async def test_term_and_vote_are_one_row_updated_in_place(db_path):
    async with SqliteStore(db_path) as store:
        for term in range(1, 6):
            await store.save_term_and_vote(current_term=term, voted_for=term)
    with closing(sqlite3.connect(db_path)) as raw:
        assert raw.execute("SELECT COUNT(*) FROM node_state").fetchone() == (1,)


async def test_reopening_does_not_reset_persisted_state(db_path):
    async with SqliteStore(db_path) as store:
        await store.save_term_and_vote(current_term=5, voted_for=2)
    await reload(db_path)
    persisted = await reload(db_path)
    assert (persisted.current_term, persisted.voted_for) == (5, 2)


# --- The log, on its own ------------------------------------------------------


async def test_log_survives_a_reopen(db_path):
    log = make_log([1, 1, 2, 3])
    async with SqliteStore(db_path) as store:
        await store.save_log_from(1, list(log))
    assert (await reload(db_path)).log == log


async def test_save_log_from_replaces_everything_from_that_index(db_path):
    async with SqliteStore(db_path) as store:
        await store.save_log_from(1, list(make_log([1, 1, 2, 2])))
        await store.save_log_from(3, [LogEntry(term=3, command="new")])
    persisted = await reload(db_path)
    assert [e.term for e in persisted.log] == [1, 1, 3]
    assert persisted.log[2].command == "new"


async def test_save_log_from_with_no_entries_truncates_the_tail(db_path):
    async with SqliteStore(db_path) as store:
        await store.save_log_from(1, list(make_log([1, 1, 2, 2])))
        await store.save_log_from(3, [])
    assert [e.term for e in (await reload(db_path)).log] == [1, 1]


async def test_command_is_stored_and_reloaded_verbatim(db_path):
    # DD-21 / APPLY-7: the opaque string is written and read back exactly,
    # never decoded or re-encoded on the way through.
    command = '{"op": "put", "key": "x", "value": [1, 2], "request_id": "a1b2"}'
    async with SqliteStore(db_path) as store:
        await store.save_log_from(1, [LogEntry(term=1, command=command)])
    assert (await reload(db_path)).log[0].command == command
    with closing(sqlite3.connect(db_path)) as raw:
        assert raw.execute("SELECT command, typeof(command) FROM log").fetchone() == (
            command,
            "text",
        )


# --- DD-20 at the storage boundary --------------------------------------------


async def test_non_positive_vote_is_rejected_and_nothing_changes(db_path):
    async with SqliteStore(db_path) as store:
        await store.save_term_and_vote(current_term=3, voted_for=2)
        with pytest.raises(sqlite3.IntegrityError):
            await store.save_term_and_vote(current_term=4, voted_for=0)
    persisted = await reload(db_path)
    assert (persisted.current_term, persisted.voted_for) == (3, 2)


async def test_non_integer_vote_is_rejected(db_path):
    # Without STRICT tables, 'n1' would pass CHECK (voted_for > 0) and be
    # stored as text.
    async with SqliteStore(db_path) as store:
        with pytest.raises(sqlite3.IntegrityError):
            await store.save_term_and_vote(current_term=1, voted_for="n1")
    assert (await reload(db_path)).voted_for is None


async def test_negative_term_is_rejected(db_path):
    async with SqliteStore(db_path) as store:
        with pytest.raises(sqlite3.IntegrityError):
            await store.save_term_and_vote(current_term=-1, voted_for=None)


# --- DD-7: a failed write leaves nothing half-done ----------------------------


async def test_failed_log_write_rolls_back_the_delete_too(db_path):
    async with SqliteStore(db_path) as store:
        await store.save_log_from(1, list(make_log([1, 1, 2, 2])))
        bad_entry = LogEntry(term="not-an-int", command="x")
        with pytest.raises(sqlite3.IntegrityError):
            await store.save_log_from(3, [LogEntry(term=3, command="ok"), bad_entry])
    # Neither the delete of entries 3-4 nor the first insert took effect.
    assert [e.term for e in (await reload(db_path)).log] == [1, 1, 2, 2]


async def test_store_remains_usable_after_a_rejected_write(db_path):
    async with SqliteStore(db_path) as store:
        with pytest.raises(sqlite3.IntegrityError):
            await store.save_term_and_vote(current_term=1, voted_for=0)
        await store.save_term_and_vote(current_term=1, voted_for=1)
    assert (await reload(db_path)).voted_for == 1
