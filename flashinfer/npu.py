"""NPU-only extensions; these APIs have no upstream FlashInfer counterpart.

The caller owns graph capture/replay, streams and fixed buffers. Query the
selected kernel ID before capture. NEEDS_PARAM_UPDATE additionally requires a
ReplayParameterUpdates recording scope inside the caller's capture, and a
before_replay call on the replay stream before EVERY replay. See the selected
descriptor's graph_replay for its verified argument scope and test evidence.
"""

from contextlib import contextmanager

from .runtime._graph_params import _ReplayParameterUpdates, _validate_graph_declaration
from .runtime.registry import get_global_registry
from .runtime.schema import GraphCapability, GraphReplaySpec


__all__ = ["GraphCapability", "GraphReplaySpec", "ReplayParameterUpdates",
           "graph_requirements", "maximum_workspace_size"]


def graph_requirements(kernel_id, *, registry=None):
    """Return (GraphCapability, update parameter names), without device probes.

    Non-default declarations must carry kernel-bound validation evidence.
    This is a capability query, not a promise about untested argument shapes;
    recording enforces the descriptor's graph_replay scope.
    """
    registry = get_global_registry() if registry is None else registry
    descriptor = registry.get(kernel_id)
    _validate_graph_declaration(descriptor)
    return descriptor.graph_capability, descriptor.graph_param_updates


class ReplayParameterUpdates:
    """Updates for one caller-owned graph, in capture task order.

    Keep this object alive with the graph. Use recording() inside the caller's
    capture. Each before_replay input is a mapping per recorded task (including
    repeated calls of the same kernel); task.kernel_id identifies the binding.
    Mapping keys are logical sources from descriptor.graph_replay.parameter_sources,
    e.g. kv_seq_lengths, not CANN argument names or positional ABI slots.
    The caller serializes access and enqueues replay on the same current
    stream. A failed device update invalidates this object and its graph.
    This object never creates, captures, or replays a graph.
    """

    def __init__(self):
        self._updates = _ReplayParameterUpdates()

    @property
    def tasks(self):
        """Captured tasks in forward order, with their bound kernel_id."""
        return self._updates.tasks

    @contextmanager
    def recording(self):
        """Record operator tasks inside an already active caller capture."""
        with self._updates.recording():
            yield self

    def before_replay(self, parameter_updates, *, stream):
        """Validate every task, patch parameters, then signal captured waits."""
        self._updates.update(parameter_updates, stream=stream)


def maximum_workspace_size(wrapper, *plan_args, **plan_kwargs):
    """T10 read-only maximum byte query, using wrapper.workspace_size inputs.

    Returns (float_workspace_bytes, int_workspace_bytes). This does not enable
    graph mode, set use_max_workspace, publish a plan, or allocate persistent
    storage. The existing use_max_workspace=False default remains unchanged.
    """
    return wrapper.workspace_size(*plan_args, **plan_kwargs)
