"""raft-kv-store: a key-value store replicated with Raft.

The `raftkv` logger has only a `NullHandler`: nothing is emitted unless the
application configures logging.
"""

import logging

logging.getLogger("raftkv").addHandler(logging.NullHandler())
