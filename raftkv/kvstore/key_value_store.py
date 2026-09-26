"""The KV Store layer's state machine: a key-value map guarded by client sessions."""

from raftkv.kvstore.commands import OpenSession, Put, decode_command
from raftkv.kvstore.results import (
    CommandResult,
    PutApplied,
    SessionExpired,
    SessionOpened,
    StaleRequest,
)
from raftkv.kvstore.session_table import DEFAULT_SESSION_TIMEOUT, Session, SessionTable


class KeyValueStore:
    """A key-value map and its client sessions, built only by applying committed commands.

    The Raft layer replicates each command as an opaque string and never decodes
    it (DD-21); this layer builds and reads them. `apply` depends on nothing but
    the current state and its arguments, so every replica that applies the same
    commands, at the same indexes and cluster times, holds the same map and the
    same sessions (APPLY-6, DD-12).

    Every put carries its session and request number (FAIL-4, DD-15). A put that
    repeats its session's latest request is answered with that request's first
    result and changes no key, so a retried put takes effect once (FAIL-6); a
    put whose session is not open is refused.

    Attributes:
        keys: The keys currently stored, in sorted order.
        sessions: A copy of every open session, by client ID, most idle first.
    """

    def __init__(self, session_timeout: int = DEFAULT_SESSION_TIMEOUT) -> None:
        """Create an empty store, as a node holds before it applies anything.

        Args:
            session_timeout: How long a session may stay idle, in cluster time,
                before it expires.

        Raises:
            ValueError: If `session_timeout` is below 1.
        """
        self._values: dict[str, str] = {}
        self._sessions = SessionTable(session_timeout)

    @property
    def keys(self) -> list[str]:
        return sorted(self._values)

    @property
    def sessions(self) -> dict[int, Session]:
        return self._sessions.as_dict()

    def apply(self, index: int, cluster_time: int, command: str) -> CommandResult:
        """Apply one committed command (APPLY-6, APPLY-7, FAIL-6).

        Sessions idle for more than the timeout at `cluster_time` are forgotten
        first. An OpenSession opens session `index`. A Put takes effect unless its
        session is not open (SessionExpired), it repeats the session's latest
        request (that request's result again), or it is older than that
        (StaleRequest).

        Args:
            index: The command's log index; a session it opens takes it as its
                client ID.
            cluster_time: The cluster time on the command's entry (DD-32).
            command: A command built by this layer.

        Returns:
            The command's result.

        Raises:
            ValueError: If `command` is not one this layer builds. Nothing is
                applied; a replica that refused a command every other replica
                accepted would diverge, so this signals a bug, not a bad request.
        """
        # NOTE: decoded before anything expires, so a command that fails to decode changes
        # nothing at all.
        request = decode_command(command)
        self._sessions.expire_idle(cluster_time)
        match request:
            case OpenSession():
                self._sessions.open(index, cluster_time)
                return SessionOpened(client_id=index)
            case Put():
                return self._apply_put(request, cluster_time)

    def get(self, key: str) -> str | None:
        """Return the value stored under `key`, or None if there is none."""
        return self._values.get(key)

    def session(self, client_id: int) -> Session | None:
        """Return the open session `client_id`, or None if it is not open."""
        return self._sessions.get(client_id)

    def as_dict(self) -> dict[str, str]:
        """Return a copy of the whole map, for comparing replicas."""
        return dict(self._values)

    def _apply_put(self, put: Put, cluster_time: int) -> CommandResult:
        session = self._sessions.get(put.client_id)
        if session is None:
            return SessionExpired()
        if put.seq < session.last_seq:
            return StaleRequest()
        if put.seq == session.last_seq:
            # NOTE: the stored result, not a new one: the key may have changed since, and a
            # retry must learn what its request did, not what a second put would do.
            self._sessions.touch(put.client_id, cluster_time)
            return session.last_result
        result = PutApplied(previous_value=self._values.get(put.key))
        self._values[put.key] = put.value
        self._sessions.record(put.client_id, put.seq, result, cluster_time)
        return result
