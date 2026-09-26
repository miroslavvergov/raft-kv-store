"""What applying a command returns: the KV Store layer's command results (DD-12)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SessionOpened:
    """A client session was opened (DD-15).

    Attributes:
        client_id: The new session's ID: the log index of its OpenSession command.
    """

    client_id: int


@dataclass(frozen=True)
class PutApplied:
    """A put took effect; a retry of it gets this same result again (FAIL-6).

    Attributes:
        previous_value: The value the key held just before the put took effect;
            None if it held none.
    """

    previous_value: str | None


@dataclass(frozen=True)
class SessionExpired:
    """The command's session is not open, so nothing was applied (DD-15).

    It expired, or never existed: once an expired session is forgotten, the two
    cannot be told apart. The client cannot learn whether an earlier attempt of
    the request took effect.
    """


@dataclass(frozen=True)
class StaleRequest:
    """The request is older than its session's latest, so nothing was applied (FAIL-6).

    A client has one request outstanding at a time, so this is a late copy of a
    request its client has already had answered.
    """


# Every result applying a command can return.
CommandResult = SessionOpened | PutApplied | SessionExpired | StaleRequest
