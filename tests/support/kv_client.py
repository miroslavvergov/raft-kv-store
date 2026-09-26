"""A test client of one KV session: it numbers its requests, and repeats a number only to retry."""

from raftkv.kvstore import Put


class KvClient:
    """Builds one session's puts, numbering each new request and reusing the number for a retry.

    Attributes:
        client_id: The session's client ID.
        last: The latest request built, which `retry` repeats; None before the first.
        last_seq: The number of that request; 0 before the first.
    """

    def __init__(self, client_id):
        self.client_id = client_id
        self.last = None
        self._seq = 0

    @property
    def last_seq(self):
        return self._seq

    def put(self, key, value):
        """Return a new request storing `value` under `key`."""
        self._seq += 1
        self.last = Put(self.client_id, self._seq, key, value).encode()
        return self.last

    def retry(self):
        """Return the latest request again, with its number unchanged."""
        return self.last
