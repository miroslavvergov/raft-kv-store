"""The decorator that reports a node's method calls without the method knowing."""

import functools
import inspect
import logging
from collections.abc import Callable
from typing import Any

from raftkv.tracing.node_snapshot import NodeSnapshot
from raftkv.tracing.node_tracer import NodeTracer

_failures = logging.getLogger(__name__)


def traced(report: Callable[..., None]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Report each call of the decorated `DurableNodeState` method to the node's tracer.

    Takes a `NodeSnapshot` just before and just after the call and passes both,
    with the call's arguments and its result or exception, to `report`. Apply it
    beneath the node's lock decorator, so both snapshots are taken under the
    lock. The "after" snapshot is taken even when the method raises: a failed
    write shows no change, and a cancelled caller's installed change shows.
    With tracing off, the call goes straight through and nothing is
    snapshotted or reported.

    Args:
        report: The `NodeTracer.report_*` method for this kind of call. For a
            coroutine method it is called as `report(tracer, before, after,
            *args, result=..., error=...)`, where `args` are the call's
            arguments for the method's positional parameters, defaults filled
            in, passed positionally however the caller passed them. For
            `__init__` it is called as `report(tracer, after)` once the
            constructor returns.

    Returns:
        The decorator.

    Raises:
        TypeError: If the decorator is applied to a synchronous method other
            than `__init__`.
    """

    def decorate(method: Callable[..., Any]) -> Callable[..., Any]:
        if not inspect.iscoroutinefunction(method):
            if method.__name__ != "__init__":
                raise TypeError(
                    f"traced supports coroutine methods and __init__, not {method.__qualname__}"
                )

            @functools.wraps(method)
            def traced_init(node: Any, *args: Any, **kwargs: Any) -> None:
                method(node, *args, **kwargs)
                tracer = NodeTracer(node.node_id)
                if tracer.enabled:
                    _report_safely(report, tracer, NodeSnapshot.of(node))

            return traced_init

        signature = inspect.signature(method)

        @functools.wraps(method)
        async def traced_method(node: Any, *args: Any, **kwargs: Any) -> Any:
            tracer = NodeTracer(node.node_id)
            if not tracer.enabled:
                return await method(node, *args, **kwargs)
            # NOTE: `report` sees `bound.args[1:]` with defaults filled in, so a call reads the
            # same however it was spelled.
            bound = signature.bind(node, *args, **kwargs)
            bound.apply_defaults()
            positional = bound.args[1:]
            before = NodeSnapshot.of(node)
            try:
                result = await method(node, *positional, **bound.kwargs)
            # NOTE: BaseException, so a cancelled call is reported too, with any change it
            # installed before the cancellation was raised.
            except BaseException as error:
                _report_safely(
                    report,
                    tracer,
                    before,
                    NodeSnapshot.of(node),
                    *positional,
                    result=None,
                    error=error,
                )
                raise
            _report_safely(
                report,
                tracer,
                before,
                NodeSnapshot.of(node),
                *positional,
                result=result,
                error=None,
            )
            return result

        return traced_method

    return decorate


def _report_safely(report: Callable[..., None], *args: Any, **kwargs: Any) -> None:
    """Call `report`; if it raises, log that and leave the traced call unaffected."""
    try:
        report(*args, **kwargs)
    except Exception:
        _failures.exception("reporting with %s failed", report.__qualname__)
