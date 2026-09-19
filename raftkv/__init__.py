"""raft-kv-store: a key-value store replicated with Raft.

The `raftkv` logger gets a `NullHandler`, the standard setup for a library:
nothing is printed or written unless the application running a node (or a
test run with `--trace-elections`) attaches handlers of its own.
"""

import logging

logging.getLogger("raftkv").addHandler(logging.NullHandler())
