"""The commands the KV Store layer replicates, each serialized once into the string Raft carries."""

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class OpenSession:
    """A request to open a client session (DD-15).

    The session's client ID is the log index at which this command is applied,
    so it is unique without any replica choosing one. A client may use it only
    once this command has committed at that index, in the term it was appended
    in, and never after the client itself restarts.
    """

    def encode(self) -> str:
        """Return the command's one serialized form (DD-21)."""
        return json.dumps({"op": "open_session"}, sort_keys=True)


@dataclass(frozen=True)
class Put:
    """A request to store `value` under `key`, numbered within its client's session (FAIL-4).

    A client numbers its requests 1, 2, 3, and so on, never reusing a number for
    a different request, and resends a request with the same number, which is
    how every replica recognizes a retry (FAIL-6).

    Attributes:
        client_id: The client's session ID.
        seq: The request's number within the session, from 1.
        key: The key to store under.
        value: The value to store.

    Raises:
        TypeError: If `client_id` or `seq` is not an int, or `key` or `value` not
            a str.
        ValueError: If `client_id` or `seq` is below 1.
    """

    client_id: int
    seq: int
    key: str
    value: str

    def __post_init__(self) -> None:
        for name in ("client_id", "seq"):
            number = getattr(self, name)
            # NOTE: bool is an int subclass, so True would pass as 1 without this check.
            if isinstance(number, bool) or not isinstance(number, int):
                raise TypeError(f"Put.{name} must be an int, got {type(number).__name__}")
            if number < 1:
                raise ValueError(f"Put.{name} must be at least 1, got {number}")
        if not isinstance(self.key, str) or not isinstance(self.value, str):
            raise TypeError(
                f"Put.key and Put.value must be str, got {type(self.key).__name__} "
                f"and {type(self.value).__name__}"
            )

    def encode(self) -> str:
        """Return the command's one serialized form (DD-21)."""
        return json.dumps(
            {
                "op": "put",
                "client_id": self.client_id,
                "seq": self.seq,
                "key": self.key,
                "value": self.value,
            },
            sort_keys=True,
        )


# Every command this layer builds.
Command = OpenSession | Put


def decode_command(encoded: str) -> Command:
    """Return the Command that `encoded` serializes.

    Raises:
        ValueError: If `encoded` is not a command this layer builds.
    """
    try:
        fields = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise ValueError(f"not a command: {encoded!r}") from error
    if not isinstance(fields, dict):
        raise ValueError(f"not a command: {encoded!r}")
    op = fields.pop("op", None)
    try:
        if op == "open_session" and not fields:
            return OpenSession()
        if op == "put" and set(fields) == {"client_id", "seq", "key", "value"}:
            return Put(**fields)
    except (TypeError, ValueError) as error:
        raise ValueError(f"malformed {op}: {encoded!r}") from error
    raise ValueError(f"unknown command: {encoded!r}")
