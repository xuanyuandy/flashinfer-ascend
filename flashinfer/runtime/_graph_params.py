"""Descriptor-driven task parameter updates for caller-owned ACL graphs.

This module never creates, captures or replays a graph. The caller surrounds
its forward with torch.npu.graph and recording(), then calls update() before
each replay on the current stream. Keep this object alive with that graph:
records deliberately own their tensor arguments, task handles and events.

Only fixed-layout host parameters are supported here. The public NPU extension
is flashinfer.npu; importing either module needs no optional device packages.
"""

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType

from .schema import GraphCapability, SchemaError


_ACTIVE_UPDATES = ContextVar("flashinfer_task_parameter_recording", default=None)


def _npu():
    import torch_npu

    return torch_npu.npu


def _snapshot(value):
    # Copy host containers; tensor storage must stay shared with the graph.
    if isinstance(value, Mapping):
        return {name: _snapshot(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_snapshot(item) for item in value)
    return value


def _host_layout(value):
    if type(value) in (bool, int, float, str, type(None)):
        return type(value)
    if isinstance(value, (tuple, list)):
        return (type(value), tuple(_host_layout(item) for item in value))
    raise SchemaError("graph parameter updates require host scalars or fixed-length sequences")


def _tensor_binding(tensor):
    return (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()),
            str(tensor.dtype), str(tensor.device))


def _tensor_bindings(value):
    if callable(getattr(value, "data_ptr", None)):
        yield value, _tensor_binding(value)
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _tensor_bindings(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensor_bindings(item)


def _validate_graph_declaration(descriptor):
    if descriptor.graph_capability != GraphCapability.NOT_CAPTURABLE and descriptor.graph_replay is None:
        raise SchemaError("%s: capturable declaration has no graph replay validation evidence" %
                          descriptor.kernel_id)


def _validate_graph_arguments(descriptor, arguments):
    """Validate the declared finite scope before creating or updating nodes."""
    spec = descriptor.graph_replay
    for name in spec.required_arguments:
        if arguments.get(name) is None:
            raise SchemaError("%s: graph replay requires argument %s" % (descriptor.kernel_id, name))
    for name, value in spec.fixed_arguments:
        if name not in arguments or type(arguments[name]) is not type(value) or arguments[name] != value:
            raise SchemaError("%s: graph replay requires %s=%r" % (descriptor.kernel_id, name, value))
    for name, shape, dtypes in spec.tensor_metadata:
        tensor = arguments.get(name)
        if (tuple(getattr(tensor, "shape", ())) != shape
                or str(getattr(tensor, "dtype", "")).removeprefix("torch.") not in dtypes
                or str(getattr(tensor, "device", "")).split(":", 1)[0] != "npu"):
            raise SchemaError("%s: %s is outside the verified graph tensor metadata" %
                              (descriptor.kernel_id, name))
    for name, minimum, maximum in spec.parameter_ranges:
        value = arguments.get(name)
        values = value if isinstance(value, (tuple, list)) else (value,)
        if not values or any(type(v) is not int or not minimum <= v <= maximum for v in values):
            raise SchemaError("%s: %s must stay within verified graph range [%d, %d]" %
                              (descriptor.kernel_id, name, minimum, maximum))


@dataclass(frozen=True)
class _ParameterLayout:
    names: tuple
    updates: tuple
    host_layouts: tuple

    @classmethod
    def from_arguments(cls, arguments, updates):
        updates = tuple(updates)
        if (not updates or len(set(updates)) != len(updates)
                or any(name not in arguments for name in updates)):
            raise SchemaError("graph update names must be unique, non-empty and bound to arguments")
        return cls(tuple(arguments), updates,
                   tuple(_host_layout(arguments[name]) for name in updates))

    def bind(self, arguments, changes):
        if not isinstance(changes, Mapping) or set(changes) != set(self.updates):
            raise SchemaError("graph update must supply exactly these parameters: %s" %
                              ", ".join(self.updates))
        values = _snapshot(changes)
        if tuple(_host_layout(values[name]) for name in self.updates) != self.host_layouts:
            raise SchemaError("graph update host parameter layout changed")
        return {**arguments, **values}


@dataclass(frozen=True)
class _RecordedTask:
    operation: object
    arguments: object
    layout: _ParameterLayout
    bindings: tuple
    handle: object
    event: object
    descriptor: object = None

    @property
    def kernel_id(self):
        return None if self.descriptor is None else self.descriptor.kernel_id

    def bind(self, changes):
        for tensor, binding in self.bindings:
            if _tensor_binding(tensor) != binding:
                raise SchemaError("graph task tensor address/shape/stride/dtype/device changed")
        if self.descriptor is not None:
            sources = self.descriptor.graph_replay.parameter_sources
            names = {source for _, source in sources}
            if not isinstance(changes, Mapping) or set(changes) != names:
                raise SchemaError("graph update must supply exactly these logical parameters: %s" %
                                  ", ".join(sorted(names)))
            changes = {name: changes[source] for name, source in sources}
        arguments = self.layout.bind(self.arguments, changes)
        if self.descriptor is not None:
            _validate_graph_arguments(self.descriptor, arguments)
        return arguments


class _ReplayParameterUpdates:
    """One recording scope per caller-owned graph; named updates per task.

    update() orders its stream after work already on the caller's current
    (replay) stream. Each captured ExternalEvent wait/reset orders the next
    task after its parameter update. The caller must serialize access to this
    object and enqueue replay on that same current stream after update().
    A failed device update invalidates the records: discard the graph too.
    """

    def __init__(self):
        self._tasks = []
        self._state = "new"
        self._capture_stream = None

    @property
    def tasks(self):
        return tuple(self._tasks)

    @contextmanager
    def recording(self):
        if self._state != "new" or _ACTIVE_UPDATES.get() is not None:
            raise SchemaError("graph parameter recording scopes cannot be nested or reused")
        npu = _npu()
        if not npu.is_current_stream_capturing():
            raise SchemaError("recording requires an active caller-owned NPU graph capture")
        self._capture_stream = npu.current_stream()
        self._state = "recording"
        token = _ACTIVE_UPDATES.set(self)
        try:
            yield self
            if self._state != "recording":
                raise SchemaError("graph parameter recording failed; discard the caller's graph")
            if not self._tasks:
                raise SchemaError("no updatable graph tasks were recorded")
        except BaseException:
            self._state = "failed"
            self._tasks.clear()
            raise
        else:
            self._state = "ready"
        finally:
            _ACTIVE_UPDATES.reset(token)

    def _record(self, operation, arguments, updates, descriptor=None):
        if self._state != "recording":
            raise SchemaError("graph parameter recording failed; discard the caller's graph")
        npu = _npu()
        stream = npu.current_stream()
        if not npu.is_current_stream_capturing() or stream != self._capture_stream:
            raise SchemaError("graph task recording must stay on its capture stream")
        arguments = _snapshot(arguments)
        layout = _ParameterLayout.from_arguments(arguments, updates)
        bindings = tuple(_tensor_bindings(arguments))
        event = npu.ExternalEvent()
        # These become wait/reset nodes in the caller's graph. Before every
        # replay, record the event AFTER graph_task_update_end, never before.
        try:
            event.wait(stream)
            event.reset(stream)
            npu.graph_task_group_begin(stream)
            try:
                result = operation(**arguments)
            finally:
                handle = npu.graph_task_group_end(stream)
        except BaseException:
            # Even if forward catches the operator's exception, a partial
            # capture must never be published as an updatable recording.
            self._state = "failed"
            raise
        self._tasks.append(_RecordedTask(operation, MappingProxyType(arguments),
                                         layout, bindings, handle, event, descriptor))
        return result

    def update(self, parameter_updates, *, stream):
        if self._state != "ready":
            raise SchemaError("graph parameter updates require a completed recording")
        npu = _npu()
        if npu.is_current_stream_capturing():
            raise SchemaError("graph parameters must be updated outside capture")
        replay_stream = npu.current_stream()
        if (stream.device != self._capture_stream.device
                or replay_stream.device != self._capture_stream.device):
            raise SchemaError("graph update and replay streams must use the capture device")
        parameter_updates = tuple(parameter_updates)
        if len(parameter_updates) != len(self._tasks):
            raise SchemaError("graph update count must match the recorded task count")
        # Validate ALL tasks before touching any graph node or recording events.
        arguments = tuple(task.bind(changes) for task, changes in
                          zip(self._tasks, parameter_updates))
        try:
            if stream != replay_stream:
                stream.wait_stream(replay_stream)
            with npu.stream(stream):
                for task, values in zip(self._tasks, arguments):
                    npu.graph_task_update_begin(stream, task.handle)
                    try:
                        task.operation(**values)
                    finally:
                        npu.graph_task_update_end(stream)
                    task.event.record(stream)
        except BaseException:
            self._state = "failed"
            raise


def _is_recording():
    return _ACTIVE_UPDATES.get() is not None


def _record_call(operation, arguments, *, updates):
    active = _ACTIVE_UPDATES.get()
    if active is None:
        return operation(**arguments)
    return active._record(operation, arguments, updates)


def _record_kernel_call(descriptor, operation, arguments):
    """Bind a descriptor's named recipe to the actual low-level operation.

    The adapter supplies the operation and its ordinary keyword arguments.
    It supplies no parameter update list or operator-specific replay code.
    """
    active = _ACTIVE_UPDATES.get()
    if active is None:
        return operation(**arguments)
    try:
        if descriptor.graph_capability == GraphCapability.NOT_CAPTURABLE:
            raise SchemaError("%s: graph_capability=not_capturable" % descriptor.kernel_id)
        _validate_graph_declaration(descriptor)
        _validate_graph_arguments(descriptor, arguments)
        if descriptor.graph_capability == GraphCapability.CAPTURE_SAFE:
            return operation(**arguments)
        return active._record(operation, arguments, descriptor.graph_param_updates, descriptor)
    except BaseException:
        active._state = "failed"
        raise
