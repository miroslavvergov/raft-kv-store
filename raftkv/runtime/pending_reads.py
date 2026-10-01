"""The reads a Leader holds: each waits for a majority's confirmation and for applying (DD-34)."""

import asyncio
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class _Read:
    """One caller waiting to read.

    Attributes:
        mark: The number of requests built when the read started; None until it starts.
            Only requests built after it confirm the read.
        read_index: The commit index the read started at; set with `mark`.
    """

    mark: int | None = None
    read_index: int = 0


class PendingReads:
    """Reads waiting to be served, oldest first, each ended once with its read index or an error.

    A read is registered when it arrives and starts once the Leader has committed an
    entry of its own term (CLIENT-10): its read index and mark are taken then. It is
    confirmed when a majority has answered a request built after its mark (CLIENT-8),
    and it ends when the state machine has applied through its read index (CLIENT-9).
    Reads start in arrival order, so neither the mark nor the read index ever falls
    along the queue: a read not yet confirmed, or not yet applied through, holds back
    every later one, and only the oldest and the newest ever need testing. `fail_all`
    ends every read with the error it is given.
    """

    def __init__(self) -> None:
        """Create a registry with no read waiting."""
        self._reads: OrderedDict[asyncio.Future[int], _Read] = OrderedDict()

    def __len__(self) -> int:
        return len(self._reads)

    def register(self) -> asyncio.Future[int]:
        """Register a read that has arrived and has not started.

        Returns:
            A future that ends with the read index, or with the error that ended the
            wait.
        """
        future = asyncio.get_running_loop().create_future()
        self._reads[future] = _Read()
        return future

    def has_unstarted(self) -> bool:
        """Whether a read has arrived and not started."""
        # NOTE: the unstarted reads are the newest, since a start takes all of them.
        return bool(self._reads) and next(reversed(self._reads.values())).mark is None

    def start(self, mark: int, read_index: int) -> None:
        """Start every read that has not started, at `read_index`, to be confirmed after `mark`.

        Args:
            mark: What confirms them: a request built after it.
            read_index: The commit index they start at.
        """
        for read in reversed(self._reads.values()):
            if read.mark is not None:
                return
            read.mark, read.read_index = mark, read_index

    def settle(self, is_confirmed: Callable[[int], bool], last_applied: int) -> None:
        """End each read, oldest first, that a majority has confirmed and the state applied through.

        The first read that is neither ends the pass: every later one has a mark and a
        read index at least as high, so it is neither either.

        Args:
            is_confirmed: Whether a majority has answered a request built after a mark.
            last_applied: The highest index the state machine has applied.
        """
        while self._reads:
            future, read = next(iter(self._reads.items()))
            if read.mark is None or read.read_index > last_applied or not is_confirmed(read.mark):
                return
            del self._reads[future]
            # NOTE: a caller that gave up has cancelled its future, which cannot take a result.
            if not future.done():
                future.set_result(read.read_index)

    def wants_round(self, answered_request: int, is_confirmed: Callable[[int], bool]) -> bool:
        """Whether a started read still needs a Follower to answer a request built after it began.

        The newest read has the highest mark, so if it does not need the answer, no
        earlier read does.

        Args:
            answered_request: The number of the newest request that Follower has
                answered. A read whose mark is below it has that Follower's answer.
            is_confirmed: Whether a majority has answered a request built after a mark.
        """
        newest = next(reversed(self._reads.values()), None)
        return (
            newest is not None
            and newest.mark is not None
            and newest.mark >= answered_request
            and not is_confirmed(newest.mark)
        )

    def discard(self, future: asyncio.Future[int]) -> None:
        """Stop waiting for a read; whatever it later reaches is dropped.

        Args:
            future: What `register` returned for it.
        """
        self._reads.pop(future, None)

    def fail_all(self, error_type: type[Exception], message: str) -> None:
        """End every read with `error_type(message)`, a new exception for each caller.

        Args:
            error_type: The error every waiting caller gets.
            message: Its message.
        """
        waiting, self._reads = self._reads, OrderedDict()
        for future in waiting:
            if not future.done():
                future.set_exception(error_type(message))
