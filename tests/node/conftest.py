"""Node-test fixtures: a lock that never blocks, for negative controls."""

import pytest

from raftkv.node import DurableNodeState
from tests.support.store_doubles import NoLock


@pytest.fixture
def without_the_lock(monkeypatch):
    """Replace every new DurableNodeState's DD-8 lock with one that never blocks.

    For negative controls only: the one place tests touch a private attribute.
    """
    original_init = DurableNodeState.__init__

    def init_without_lock(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._lock = NoLock()

    # NOTE: patching the class, not one instance, strips the lock from every node a helper or
    # a restart builds later, since `load` builds them out of the test's reach.
    monkeypatch.setattr(DurableNodeState, "__init__", init_without_lock)
