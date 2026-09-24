"""The KV Store layer's state machine: the commands it builds, and what applying one does."""

import json
from typing import Any


class KeyValueStore:
    """A key-value map built only by applying committed commands (DD-12).

    The Raft layer replicates a command as an opaque string and never decodes it
    (DD-21); this class is the only place a command is written or read. `apply`
    depends on nothing but the current map and the command, so every replica that
    applies the same commands in the same order holds the same map (APPLY-6).

    Attributes:
        keys: The keys currently stored, in sorted order.
    """

    def __init__(self) -> None:
        """Create an empty store, as a node holds before it applies anything."""
        self._values: dict[str, str] = {}

    @property
    def keys(self) -> list[str]:
        return sorted(self._values)

    @staticmethod
    def put_command(key: str, value: str) -> str:
        """Return the command string that stores `value` under `key`.

        Serialized once here and replicated verbatim (DD-21), so the same call
        always produces the same string and two replicas cannot disagree on what a
        log entry says.

        Args:
            key: The key to store under.
            value: The value to store.

        Raises:
            TypeError: If either is not a str.
        """
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError("a key and a value must both be str")
        return json.dumps({"op": "put", "key": key, "value": value}, sort_keys=True)

    def apply(self, command: str) -> None:
        """Apply one committed command to the map (APPLY-6, APPLY-7).

        Args:
            command: A command string built by `put_command`.

        Raises:
            ValueError: If `command` is not one this store understands. Nothing is
                applied; a replica that rejected a command every other replica
                accepted would diverge, so this signals a bug, not a bad request.
        """
        try:
            decoded = json.loads(command)
        except json.JSONDecodeError as error:
            raise ValueError(f"not a command: {command!r}") from error
        if not isinstance(decoded, dict) or decoded.get("op") != "put":
            raise ValueError(f"unknown command: {command!r}")
        key, value = decoded.get("key"), decoded.get("value")
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"a put needs a str key and a str value: {command!r}")
        self._values[key] = value

    def get(self, key: str) -> str | None:
        """Return the value stored under `key`, or None if there is none."""
        return self._values.get(key)

    def as_dict(self) -> dict[str, Any]:
        """Return a copy of the whole map, for comparing replicas."""
        return dict(self._values)
