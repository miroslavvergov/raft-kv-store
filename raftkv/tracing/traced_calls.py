"""The decorator that reports a node's method calls without the method knowing."""

import functools
import inspect
from typing import Any, Callable

from raftkv.tracing.node_snapshot import NodeSnapshot
from raftkv.tracing.node_tracer import NodeTracer


def traced(report: Callable[..., None]) -> Callable:
    """Report every call of the decorated `DurableNodeState` method to the node's tracer.

    The method itself contains no tracing: the decorator takes a
    `NodeSnapshot` of the node just before the call and another just after
    it, and passes both — with the call's arguments, and its result or the
    exception it raised — to `report`, one of `NodeTracer`'s `report_*`
    methods, which works out what happened and reports it in etcd's
    format.

    Placed inside `_holding_the_lock`, both snapshots are taken with the
    node's lock held, so no other call can change the state between them
    and the method. The "after" snapshot is taken even when the method
    raises: a write that failed left the state as it was, so nothing is
    reported as changed; a caller cancelled while its write was in flight
    had the change installed before the cancellation was raised, so it is
    reported.

    With tracing off, the call goes straight through: no snapshot is
    taken and `report` is not called.

    A constructor is reported once it has finished, with no "before"
    snapshot.

    Args:
        report: The `NodeTracer` method that reports this kind of call. It
            is called as `report(tracer, before, after, *args, result=...,
            error=...)`.

    Returns:
        The decorator.
    """

    def decorate(method: Callable) -> Callable:
        if not inspect.iscoroutinefunction(method):

            @functools.wraps(method)
            def constructed(node: Any, *args: Any) -> None:
                method(node, *args)
                tracer = NodeTracer(node.node_id)
                if tracer.enabled:
                    report(tracer, None, NodeSnapshot.of(node), *args, result=None, error=None)

            return constructed

        @functools.wraps(method)
        async def called(node: Any, *args: Any) -> Any:
            tracer = NodeTracer(node.node_id)
            if not tracer.enabled:
                return await method(node, *args)
            before = NodeSnapshot.of(node)
            try:
                result = await method(node, *args)
            except BaseException as error:
                report(tracer, before, NodeSnapshot.of(node), *args, result=None, error=error)
                raise
            report(tracer, before, NodeSnapshot.of(node), *args, result=result, error=None)
            return result

        return called

    return decorate
