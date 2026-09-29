"""Tier 1 tests for PendingProposals: waiting for a proposed command's entry to be applied.

CLIENT-4, CLIENT-5 (a proposal ends only once its entry is applied), DD-12 (and only if the
entry there is the one it appended), DD-33 (a wait can also end in failure).
"""

import asyncio

import pytest

from raftkv.consensus import LogPosition
from raftkv.runtime import LeadershipLostError, NodeStoppedError
from raftkv.runtime.pending_proposals import PendingProposals
from tests.support.store_doubles import let_other_tasks_run
from tests.support.waiting import within_bound


def at(index, term=1):
    return LogPosition(term=term, index=index)


async def test_a_proposal_is_answered_with_the_result_when_its_index_is_applied_in_its_term():
    proposals = PendingProposals()
    future = proposals.register(at(5, term=2))

    proposals.answer(index=5, term=2, result="the result")

    assert await within_bound(future) == "the result"
    assert len(proposals) == 0


async def test_a_proposal_waits_while_other_indexes_are_applied():
    proposals = PendingProposals()
    future = proposals.register(at(5))

    proposals.answer(index=4, term=1, result="another entry")
    proposals.answer(index=6, term=1, result="a later entry")

    assert not future.done()
    assert len(proposals) == 1


async def test_a_proposal_fails_when_its_index_is_applied_in_another_term():
    # A later Leader overwrote the entry, so what was applied there is not the command.
    proposals = PendingProposals()
    future = proposals.register(at(5, term=2))

    proposals.answer(index=5, term=3, result="someone else's result")

    with pytest.raises(LeadershipLostError, match="the command was not applied"):
        await within_bound(future)
    assert len(proposals) == 0


async def test_an_applied_entry_that_nobody_proposed_is_ignored():
    proposals = PendingProposals()

    proposals.answer(index=5, term=1, result="a Follower's entry")

    assert len(proposals) == 0


async def test_an_index_is_answered_once_and_then_free_to_wait_on_again():
    proposals = PendingProposals()
    first = proposals.register(at(5, term=1))
    proposals.answer(index=5, term=1, result="first")
    proposals.answer(index=5, term=1, result="applied again")  # changes nothing

    second = proposals.register(at(5, term=2))
    proposals.answer(index=5, term=2, result="second")

    assert (await within_bound(first), await within_bound(second)) == ("first", "second")


async def test_a_second_proposal_on_a_waiting_index_is_refused():
    proposals = PendingProposals()
    proposals.register(at(5))

    with pytest.raises(ValueError, match="already waits on index 5"):
        proposals.register(at(5, term=2))


async def test_a_discarded_proposal_is_forgotten_and_its_result_dropped():
    proposals = PendingProposals()
    future = proposals.register(at(5))

    proposals.discard(at(5))
    proposals.answer(index=5, term=1, result="too late")

    assert len(proposals) == 0
    assert not future.done()


async def test_discarding_a_proposal_leaves_a_later_terms_proposal_on_the_same_index():
    # The first caller wakes after its wait failed, and by then this node leads again and
    # another proposal waits on the same index: the late discard must not take that one.
    proposals = PendingProposals()
    failed = proposals.register(at(5, term=1))
    proposals.fail_all(LeadershipLostError, "stepped down")
    later = proposals.register(at(5, term=3))

    proposals.discard(at(5, term=1))

    assert len(proposals) == 1
    proposals.answer(index=5, term=3, result="the later result")
    assert await within_bound(later) == "the later result"
    with pytest.raises(LeadershipLostError):
        await within_bound(failed)


async def test_discarding_an_answered_proposal_does_nothing():
    proposals = PendingProposals()
    future = proposals.register(at(5))
    proposals.answer(index=5, term=1, result="done")

    proposals.discard(at(5))

    assert await within_bound(future) == "done"


async def test_failing_all_gives_every_waiting_caller_its_own_exception():
    proposals = PendingProposals()
    first, second = proposals.register(at(5)), proposals.register(at(6))

    proposals.fail_all(NodeStoppedError, "node 1 stopped")

    errors = []
    for future in (first, second):
        with pytest.raises(NodeStoppedError, match="node 1 stopped") as raised:
            await within_bound(future)
        errors.append(raised.value)
    assert errors[0] is not errors[1]
    assert len(proposals) == 0


async def test_failing_all_with_nobody_waiting_does_nothing():
    PendingProposals().fail_all(LeadershipLostError, "no one to tell")


async def test_a_caller_that_gave_up_leaves_nothing_that_can_be_answered_or_failed():
    proposals = PendingProposals()

    async def wait_for(position):
        await proposals.register(position)

    caller = asyncio.create_task(wait_for(at(5)))
    await let_other_tasks_run()
    caller.cancel()
    await let_other_tasks_run()
    assert caller.cancelled()

    # Both must skip the cancelled future instead of raising InvalidStateError.
    proposals.answer(index=5, term=1, result="too late")
    proposals.register(at(6)).cancel()
    proposals.fail_all(LeadershipLostError, "stepped down")

    assert len(proposals) == 0
