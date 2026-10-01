"""Tier 1 tests for PendingReads: reads waiting to start, be confirmed, and be applied through.

CLIENT-8 (a read waits for a majority's confirmation), CLIENT-9 (and for the state machine to
apply through its read index), CLIENT-10 (and starts only once the Leader has committed in its
term), DD-34 (reads wait in arrival order, each tested only as far as it can matter, and a wait
can also end in failure).
"""

import asyncio

import pytest

from raftkv.consensus import NotLeaderError
from raftkv.runtime import NodeStoppedError
from raftkv.runtime.pending_reads import PendingReads
from tests.support.store_doubles import let_other_tasks_run
from tests.support.waiting import within_bound


def never(mark):
    return False


def always(mark):
    return True


def answered_up_to(request_number):
    """Return the confirmation test of a Leader whose newest answered request is numbered so.

    A mark is confirmed when the request answered was built after it, so when it is below.
    """
    return lambda mark: mark < request_number


async def test_a_read_that_has_arrived_has_not_started_and_waits():
    reads = PendingReads()
    future = reads.register()

    assert reads.has_unstarted() and len(reads) == 1
    reads.settle(always, last_applied=100)  # an unstarted read has no read index to serve

    assert not future.done() and len(reads) == 1


async def test_starting_gives_every_unstarted_read_the_same_read_index_and_leaves_none_unstarted():
    reads = PendingReads()
    first, second = reads.register(), reads.register()

    reads.start(mark=3, read_index=9)

    assert not reads.has_unstarted()
    reads.settle(always, last_applied=9)
    assert await within_bound(first) == await within_bound(second) == 9


async def test_a_read_that_arrives_after_a_start_waits_for_the_next_one():
    reads = PendingReads()
    started = reads.register()
    reads.start(mark=3, read_index=9)
    later = reads.register()

    assert reads.has_unstarted()
    reads.settle(always, last_applied=100)
    assert await within_bound(started) == 9
    assert not later.done()

    reads.start(mark=5, read_index=12)
    reads.settle(always, last_applied=100)
    assert await within_bound(later) == 12


async def test_starting_leaves_a_read_that_started_earlier_as_it_was():
    reads = PendingReads()
    earlier = reads.register()
    reads.start(mark=3, read_index=9)
    later = reads.register()
    reads.start(mark=5, read_index=12)

    reads.settle(always, last_applied=9)  # applied through the earlier read's index only

    assert await within_bound(earlier) == 9
    assert not later.done()


@pytest.mark.parametrize(
    ("confirmed", "applied_through", "ends"),
    [
        (False, 9, False),  # applied, but no majority has answered a later request
        (True, 8, False),  # confirmed, but the state machine has not reached the read index
        (False, 8, False),
        (True, 9, True),
        (True, 12, True),  # the state machine may be past the read index
    ],
)
async def test_a_read_ends_only_when_confirmed_and_applied_through_its_read_index(
    confirmed, applied_through, ends
):
    reads = PendingReads()
    future = reads.register()
    reads.start(mark=3, read_index=9)

    reads.settle(lambda mark: confirmed, last_applied=applied_through)

    assert future.done() == ends
    assert len(reads) == (0 if ends else 1)
    if ends:
        assert await within_bound(future) == 9


async def test_each_read_is_confirmed_by_its_own_mark():
    reads = PendingReads()
    early = reads.register()
    reads.start(mark=3, read_index=9)
    late = reads.register()
    reads.start(mark=6, read_index=9)

    reads.settle(answered_up_to(4), last_applied=9)  # built after mark 3, not after mark 6

    assert early.done() and not late.done()
    reads.settle(answered_up_to(7), last_applied=9)
    assert late.done()


# --- Reads end oldest first, and are tested only as far as it can matter --------------------


async def test_a_read_waiting_to_apply_holds_back_the_reads_after_it():
    reads = PendingReads()
    early = reads.register()
    reads.start(mark=3, read_index=5)
    late = reads.register()
    reads.start(mark=6, read_index=9)

    reads.settle(always, last_applied=5)  # applied through the early read's index only
    assert early.done() and not late.done()

    reads.settle(always, last_applied=9)
    assert late.done()


async def test_a_confirmed_read_waiting_to_apply_does_not_confirm_a_later_read():
    reads = PendingReads()
    early = reads.register()
    reads.start(mark=3, read_index=9)
    late = reads.register()
    reads.start(mark=6, read_index=9)

    reads.settle(answered_up_to(4), last_applied=5)  # the early read is confirmed, not applied
    assert not early.done() and not late.done()

    reads.settle(answered_up_to(4), last_applied=9)
    assert early.done() and not late.done()  # a request built after mark 6 is still awaited


async def test_settling_tests_only_the_oldest_read_while_it_is_not_confirmed():
    reads = PendingReads()
    for mark in range(1, 1001):
        reads.register()
        reads.start(mark=mark, read_index=9)
    asked = []

    def ask(mark):
        asked.append(mark)
        return False

    reads.settle(ask, last_applied=9)

    assert asked == [1]


# --- Which reads want another round (CLIENT-8) ---------------------------------------------


async def test_a_started_unconfirmed_read_wants_a_round_from_a_follower_yet_to_answer_after_it():
    reads = PendingReads()
    reads.register()
    reads.start(mark=3, read_index=9)

    assert reads.wants_round(answered_request=0, is_confirmed=never)
    assert reads.wants_round(answered_request=3, is_confirmed=never)  # built before the read


async def test_a_read_does_not_want_a_round_from_a_follower_that_answered_after_it_began():
    reads = PendingReads()
    reads.register()
    reads.start(mark=3, read_index=9)

    assert not reads.wants_round(answered_request=4, is_confirmed=never)


async def test_a_confirmed_read_and_an_unstarted_read_want_no_round():
    reads = PendingReads()
    assert not reads.wants_round(answered_request=0, is_confirmed=never)  # nothing waits

    reads.register()
    assert not reads.wants_round(answered_request=0, is_confirmed=never)  # no mark to answer after

    reads.start(mark=3, read_index=9)
    assert not reads.wants_round(answered_request=0, is_confirmed=always)  # confirmed, not applied


async def test_a_round_is_wanted_for_the_newest_read_when_an_older_one_is_confirmed():
    reads = PendingReads()
    reads.register()
    reads.start(mark=3, read_index=9)
    reads.register()
    reads.start(mark=6, read_index=9)

    assert reads.wants_round(answered_request=0, is_confirmed=answered_up_to(4))


async def test_asking_for_a_round_tests_only_the_newest_read():
    reads = PendingReads()
    for mark in range(1, 1001):
        reads.register()
        reads.start(mark=mark, read_index=9)
    asked = []

    def ask(mark):
        asked.append(mark)
        return False

    assert reads.wants_round(answered_request=0, is_confirmed=ask)

    assert asked == [1000]


# --- Giving up and failing ------------------------------------------------------------------


async def test_a_discarded_read_is_forgotten_and_its_result_dropped():
    reads = PendingReads()
    future = reads.register()
    reads.start(mark=3, read_index=9)

    reads.discard(future)
    reads.settle(always, last_applied=9)

    assert len(reads) == 0
    assert not future.done()


async def test_a_discarded_read_in_the_middle_holds_back_none_of_the_others():
    reads = PendingReads()
    first, middle, last = reads.register(), reads.register(), reads.register()
    reads.start(mark=3, read_index=9)

    reads.discard(middle)
    reads.settle(always, last_applied=9)

    assert first.done() and last.done() and not middle.done()
    assert len(reads) == 0


async def test_failing_all_gives_every_waiting_caller_its_own_exception():
    reads = PendingReads()
    started, unstarted = reads.register(), reads.register()
    reads.start(mark=3, read_index=9)
    unstarted_later = reads.register()

    reads.fail_all(NotLeaderError, "node 1 is no longer Leader")

    errors = []
    for future in (started, unstarted, unstarted_later):
        with pytest.raises(NotLeaderError, match="no longer Leader") as raised:
            await within_bound(future)
        errors.append(raised.value)
    assert len({id(error) for error in errors}) == 3
    assert len(reads) == 0


async def test_failing_all_can_end_reads_with_a_stop():
    reads = PendingReads()
    future = reads.register()

    reads.fail_all(NodeStoppedError, "node 1 stopped")

    with pytest.raises(NodeStoppedError):
        await within_bound(future)


async def test_failing_all_with_nobody_waiting_does_nothing():
    PendingReads().fail_all(NotLeaderError, "no one to tell")


async def test_a_caller_that_gave_up_leaves_nothing_that_can_be_settled_or_failed():
    reads = PendingReads()

    async def wait_for_a_read():
        future = reads.register()
        reads.start(mark=3, read_index=9)
        await future

    caller = asyncio.create_task(wait_for_a_read())
    await let_other_tasks_run()
    caller.cancel()
    await let_other_tasks_run()
    assert caller.cancelled()

    # Both must skip the cancelled future instead of raising InvalidStateError.
    reads.settle(always, last_applied=9)
    reads.register().cancel()
    reads.fail_all(NotLeaderError, "stepped down")

    assert len(reads) == 0
