"""The client sessions a state machine keeps, and their expiry once idle (DD-15)."""

import dataclasses
from collections import OrderedDict
from dataclasses import dataclass

from raftkv.kvstore.results import PutApplied

# NOTE: one hour of cluster time at the default 100 ms tick: far longer than a client keeps
# retrying one request, so only clients that have gone away are forgotten.
DEFAULT_SESSION_TIMEOUT = 36_000


@dataclass(frozen=True)
class Session:
    """One open session: its latest request, that request's result, and its last activity.

    Attributes:
        last_seq: The number of the latest request applied; 0 before the first.
        last_result: That request's result, which a retry of it gets again; None
            before the first.
        last_active: The cluster time at which the session opened, a request of
            it took effect, or its latest request was repeated; whichever came last.
    """

    last_seq: int
    last_result: PutApplied | None
    last_active: int


class SessionTable:
    """The open client sessions, most idle first, each with its latest request and result (DD-15).

    Every `now` is cluster time from the entry being applied (DD-32), never a
    clock, so every replica, and every replay of the log, opens, updates and
    expires the same sessions at the same entries. `expire_idle` forgets every
    session idle for more than `timeout`.

    Attributes:
        timeout: How long a session may stay idle, in cluster time, before it
            expires.
    """

    def __init__(self, timeout: int = DEFAULT_SESSION_TIMEOUT) -> None:
        """Create a table with no sessions.

        Raises:
            ValueError: If `timeout` is below 1.
        """
        if timeout < 1:
            raise ValueError(f"a session timeout must be at least 1, got {timeout}")
        self._timeout = timeout
        # NOTE: kept in order of last activity. Cluster time never falls along the log, so
        # moving a session to the end whenever it is active leaves the most idle at the front.
        self._sessions: OrderedDict[int, Session] = OrderedDict()

    @property
    def timeout(self) -> int:
        return self._timeout

    def __len__(self) -> int:
        return len(self._sessions)

    def __contains__(self, client_id: object) -> bool:
        return client_id in self._sessions

    def get(self, client_id: int) -> Session | None:
        """Return the open session `client_id`, or None if it is not open."""
        return self._sessions.get(client_id)

    def open(self, client_id: int, now: int) -> None:
        """Open session `client_id` with no requests yet, active at `now`."""
        self._sessions[client_id] = Session(last_seq=0, last_result=None, last_active=now)
        self._sessions.move_to_end(client_id)

    def record(self, client_id: int, seq: int, result: PutApplied, now: int) -> None:
        """Record that request `seq` of an open session took effect at `now`, with `result`.

        Raises:
            KeyError: If session `client_id` is not open.
        """
        if client_id not in self._sessions:
            raise KeyError(client_id)
        self._sessions[client_id] = Session(last_seq=seq, last_result=result, last_active=now)
        self._sessions.move_to_end(client_id)

    def touch(self, client_id: int, now: int) -> None:
        """Mark an open session active at `now`, changing nothing else.

        Raises:
            KeyError: If session `client_id` is not open.
        """
        session = self._sessions[client_id]
        self._sessions[client_id] = dataclasses.replace(session, last_active=now)
        self._sessions.move_to_end(client_id)

    def expire_idle(self, now: int) -> list[int]:
        """Forget every session idle for more than `timeout` at `now`.

        Returns:
            The IDs of the sessions forgotten, most idle first.
        """
        expired = []
        while self._sessions:
            client_id, session = next(iter(self._sessions.items()))
            if now - session.last_active <= self._timeout:
                break
            del self._sessions[client_id]
            expired.append(client_id)
        return expired

    def as_dict(self) -> dict[int, Session]:
        """Return a copy of every open session, by client ID, most idle first."""
        return dict(self._sessions)
