"""The proposals a Leader is waiting on: commands appended to its log, not yet applied (DD-33)."""

import asyncio
from dataclasses import dataclass

from raftkv.consensus import LogPosition
from raftkv.runtime.errors import LeadershipLostError


@dataclass(frozen=True)
class _Proposal:
    """One caller waiting for its command's entry to be applied.

    Attributes:
        term: The term the entry was appended in.
        future: Ends with the state machine's result, or with the error that ended the wait.
    """

    term: int
    future: asyncio.Future[object]


class PendingProposals:
    """The proposals waiting for their entries to be applied; each ends once.

    A proposal waits under its entry's index and remembers the term the entry was appended
    in. `answer` ends it with the state machine's result when that index is applied with that
    term, and with `LeadershipLostError` when it is applied with another, since a later Leader
    put a different entry there (DD-12). `fail_all` ends every proposal with the error it is
    given.
    """

    def __init__(self) -> None:
        """Create a registry with no proposal waiting."""
        self._waiting: dict[int, _Proposal] = {}

    def __len__(self) -> int:
        return len(self._waiting)

    def register(self, position: LogPosition) -> asyncio.Future[object]:
        """Register a proposal waiting for the entry at `position` to be applied.

        Args:
            position: The term and index the entry was appended at.

        Returns:
            A future that ends with the state machine's result, or with the error that
            ended the wait.

        Raises:
            ValueError: If a proposal already waits on that index.
        """
        if position.index in self._waiting:
            raise ValueError(f"a proposal already waits on index {position.index}")
        future = asyncio.get_running_loop().create_future()
        self._waiting[position.index] = _Proposal(position.term, future)
        return future

    def answer(self, index: int, term: int, result: object) -> None:
        """End the proposal waiting on `index`, if any, now that the entry there is applied.

        A proposal appended in another term than `term` fails with `LeadershipLostError`:
        a later Leader put a different entry at its index, so its command was not applied.

        Args:
            index: The index just applied.
            term: The term of the entry applied there.
            result: What the state machine returned for it.
        """
        proposal = self._waiting.pop(index, None)
        # NOTE: a caller that gave up has cancelled its future, which cannot take an answer.
        if proposal is None or proposal.future.done():
            return
        # NOTE: a backstop. A node that stops leading fails every proposal first, before such
        # an entry can be applied; the term check keeps an answer from resting on the index alone.
        if proposal.term == term:
            proposal.future.set_result(result)
        else:
            proposal.future.set_exception(
                LeadershipLostError(
                    f"index {index} holds an entry of term {term}, not the term-{proposal.term} "
                    "entry proposed there: the command was not applied"
                )
            )

    def discard(self, position: LogPosition) -> None:
        """Stop waiting for the entry at `position`; its result, once applied, is dropped.

        Does nothing if that proposal has ended already, or if the one now waiting on the
        index was appended in another term.

        Args:
            position: The term and index the entry was appended at.
        """
        proposal = self._waiting.get(position.index)
        if proposal is not None and proposal.term == position.term:
            del self._waiting[position.index]

    def fail_all(self, error_type: type[Exception], message: str) -> None:
        """End every wait with `error_type(message)`, a new exception for each caller.

        Args:
            error_type: The error every waiting caller gets.
            message: Its message.
        """
        waiting, self._waiting = self._waiting, {}
        for proposal in waiting.values():
            if not proposal.future.done():
                proposal.future.set_exception(error_type(message))
