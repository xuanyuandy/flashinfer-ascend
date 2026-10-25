"""Public fixed-resource single decode for the validated CANN graph scope.

Preparation is deliberately separate from capture. No dispatch, workspace
query, output inference or tensor allocation occurs in run().
"""

import math
import weakref
from contextlib import contextmanager

from ..runtime import build_signature, current_hardware, dispatch, get_global_registry
from ..runtime._graph_params import (
    _record_kernel_call, _tensor_binding, _validate_graph_arguments,
    _validate_graph_declaration,
)
from ..runtime.schema import GraphCapability, SchemaError


_BOUND_GRAPHS = weakref.WeakSet()


class SingleDecodeWithKVCacheWrapper:
    """Plan/run resources for the verified single-decode aclgraph route.

    Import from ``flashinfer.decode``. Supports contiguous NHD Q=(8,64),
    K/V=(1024,2,64), matching FP16/BF16, unmasked attention, scale=0.125.
    These are descriptor evidence limits, not general FIA capabilities.
    Q/K/V contents may change in place; their storage and metadata may not.

    ``plan`` holds the caller's Q/K/V and optional workspace/output/LSE,
    allocating omitted resources outside capture. ``warmup`` is mandatory.
    The caller creates an NPUGraph and enters torch.npu.graph with
    ``auto_dispatch_capture=False``; inside it use
    ``with wrapper.recording(graph): wrapper.run()`` exactly once. Then call
    ``before_replay(length, stream=update_stream)`` and ``replay()`` for EVERY
    replay, on the same current stream. Access must be serialized. Keep the
    wrapper, graph, streams and buffers alive until device work completes.
    One graph can bind only one such wrapper; multiple independently updated
    wrappers would need a graph-wide replay coordinator, which is not provided.

    Native ``graph.replay()`` bypasses the missing-update diagnostic and is
    unsupported for this wrapper (it can wait indefinitely for an event).
    This wrapper does not implement model graph partitioning or bucketing.
    """

    def __init__(self, workspace_buffer=None, kv_layout="NHD"):
        if kv_layout != "NHD":
            raise SchemaError("single-decode graph supports only the verified NHD layout")
        self._workspace = workspace_buffer
        self._state = "new"
        self._warmed = False
        self._pending = False

    @property
    def workspace_buffer(self):
        """Fixed workspace, available after plan (or supplied at construction)."""
        return self._workspace

    @property
    def output(self):
        """Fixed output [8,64]; overwritten by run/replay."""
        self._require_plan()
        return self._out

    @property
    def lse(self):
        """Fixed FP32 log2 LSE [8]; overwritten by run/replay."""
        self._require_plan()
        return self._lse

    @property
    def kernel_id(self):
        self._require_plan()
        return self._descriptor.kernel_id

    @property
    def kv_capacity(self):
        self._require_plan()
        return self._capacity

    def _require_plan(self):
        if self._state == "new":
            raise SchemaError("single-decode graph requires plan() outside capture first")
        if self._state == "failed":
            raise SchemaError("single-decode graph is invalid; create a new wrapper and graph")

    @staticmethod
    def _outside_capture(action):
        import torch_npu

        if torch_npu.npu.is_current_stream_capturing():
            raise SchemaError(f"{action} must run outside capture")

    @staticmethod
    def _length(value, capacity):
        if type(value) is not int or not 1 <= value <= capacity:
            raise SchemaError(
                f"kv_seq_len must be an integer in [1, {capacity}]; "
                f"plan frozen KV capacity is {capacity}"
            )
        return value

    @staticmethod
    def _tensor(tensor, name, *, device, shape=None, dtype=None):
        import torch

        if (not isinstance(tensor, torch.Tensor) or tensor.device != device
                or tensor.layout != torch.strided or not tensor.is_contiguous()
                or tensor.requires_grad
                or (shape is not None and tuple(tensor.shape) != shape)
                or (dtype is not None and tensor.dtype != dtype)):
            raise SchemaError(
                f"{name} shape/dtype/device/contiguity differs from the frozen "
                "single-decode identity; use a new wrapper with supported tensors"
            )

    @staticmethod
    def _no_output_alias(inputs, outputs):
        # All resources are contiguous. Reject overlapping byte intervals,
        # including views with different data_ptr values in the same storage.
        def overlaps(a, b):
            return (a.data_ptr() < b.data_ptr() + b.numel() * b.element_size()
                    and b.data_ptr() < a.data_ptr() + a.numel() * a.element_size())

        previous = list(inputs)
        for name, tensor in outputs:
            if any(overlaps(tensor, other) for other in previous):
                raise SchemaError(f"{name} must not alias graph inputs or writable resources")
            previous.append(tensor)

    def _check_bindings(self, q=None, k=None, v=None):
        self._require_plan()
        for name, tensor, binding in self._bindings:
            if _tensor_binding(tensor) != binding:
                raise SchemaError(
                    f"{name} frozen address/shape/stride/dtype/device changed; use a new wrapper"
                )
        if any(tensor is not None for tensor in (q, k, v)):
            for name, tensor, original in zip(("q", "k", "v"), (q, k, v), self._inputs):
                if tensor is None or _tensor_binding(tensor) != _tensor_binding(original):
                    raise SchemaError(
                        f"{name} frozen address/shape/batch/dtype/device changed; use a new wrapper"
                    )

    def plan(self, q, k, v, *, kv_seq_len=None, out=None, lse=None):
        """Bind/freeze tensors, select CANN, and prepare maximum workspace.

        The first successful plan freezes the entire execution identity.
        Before capture, repeating plan may change only the logical KV length;
        after capture use before_replay. Replacing resources needs a new wrapper.
        Workspace capacity is measured in bytes, regardless of its tensor dtype.
        """
        import torch
        import torch_npu

        self._outside_capture("plan()")
        if self._state != "new":
            self._check_bindings(q, k, v)
            for name, supplied, held in (("out", out, self._out), ("lse", lse, self._lse)):
                if supplied is not None and _tensor_binding(supplied) != _tensor_binding(held):
                    raise SchemaError(f"{name} frozen identity changed; use a new wrapper")
            if self._state != "planned":
                raise SchemaError("plan is frozen after capture; use before_replay or a new wrapper")
            length = self._length(self._capacity if kv_seq_len is None else kv_seq_len, self._capacity)
            self._arguments["actual_seq_lengths_kv"] = [length]
            self._warmed = False
            return

        if (not isinstance(q, torch.Tensor) or q.device.type != "npu"
                or q.dtype not in (torch.float16, torch.bfloat16)):
            raise SchemaError("single-decode graph requires NPU FP16/BF16 tensors")
        for name, tensor, shape in (("q", q, (8, 64)), ("k", k, (1024, 2, 64)),
                                     ("v", v, (1024, 2, 64))):
            self._tensor(tensor, name, device=q.device, shape=shape, dtype=q.dtype)
        capacity = k.shape[0]
        length = self._length(capacity if kv_seq_len is None else kv_seq_len, capacity)
        signature = build_signature(
            "attention.single_decode", (q, k, v), layouts=("NHD",) * 3,
            specializations={"head_dim": q.shape[-1]},
        )
        decision = dispatch(signature, current_hardware(), backend="cann")
        descriptor = get_global_registry().get(decision.kernel_id)
        _validate_graph_declaration(descriptor)
        if (descriptor.kernel_id != "cann/attention/single_decode_fia_v1"
                or descriptor.graph_capability != GraphCapability.NEEDS_PARAM_UPDATE):
            raise SchemaError("selected kernel has no verified single-decode graph route")
        query, key, value = q.view(1, 1, 8, 64), k.view(1, 1024, 2, 64), v.view(1, 1024, 2, 64)
        options = dict(atten_mask=None, num_heads=8, num_key_value_heads=2,
                       input_layout="BSND", scale=.125, softmax_lse_flag=True)
        maximum = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
            query, key, value, **options,
        )
        workspace = maximum if self._workspace is None else self._workspace
        self._tensor(workspace, "workspace", device=q.device)
        required_bytes = maximum.numel() * maximum.element_size()
        if workspace.numel() * workspace.element_size() < required_bytes:
            raise SchemaError(f"workspace capacity is insufficient: requires {required_bytes} bytes")
        raw_out, natural_lse = torch_npu._npu_fused_infer_attention_score_infer_output(
            query, value, input_layout="BSND", num_heads=8,
            num_key_value_heads=2, softmax_lse_flag=True,
        )
        output = raw_out.view(8, 64) if out is None else out
        log2_lse = torch.empty((8,), dtype=torch.float32, device=q.device) if lse is None else lse
        self._tensor(output, "out", device=q.device, shape=(8, 64), dtype=q.dtype)
        self._tensor(log2_lse, "lse", device=q.device, shape=(8,), dtype=torch.float32)
        self._no_output_alias((q, k, v), (("workspace", workspace), ("out", output), ("lse", log2_lse)))
        arguments = dict(query=query, key=key, value=value, workspace=workspace,
                         out=[output.view(1, 1, 8, 64), natural_lse],
                         actual_seq_lengths_kv=[length], **options)
        _validate_graph_arguments(descriptor, arguments)
        natural_lse_flat = natural_lse.view(8)
        resources = (("q", q), ("k", k), ("v", v), ("workspace", workspace),
                     ("out", output), ("lse", log2_lse), ("query", query),
                     ("key", key), ("value", value), ("raw_out", arguments["out"][0]),
                     ("natural_lse", natural_lse), ("natural_lse_flat", natural_lse_flat))
        bindings = tuple((name, tensor, _tensor_binding(tensor)) for name, tensor in resources)
        # Publish only after the entire candidate plan has passed validation.
        self._descriptor, self._capacity = descriptor, capacity
        self._inputs, self._arguments, self._bindings = (q, k, v), arguments, bindings
        self._workspace, self._out, self._lse = workspace, output, log2_lse
        self._natural_lse = natural_lse_flat
        self._operation = torch_npu.npu_fused_infer_attention_score.out
        self._state = "planned"

    def warmup(self, iterations=3):
        """Execute and synchronize outside capture; enable recording on success."""
        import torch_npu

        self._require_plan()
        self._outside_capture("warmup()")
        if self._state != "planned":
            raise SchemaError("warmup requires an uncaptured plan; use a new wrapper")
        if type(iterations) is not int or iterations < 1:
            raise SchemaError("warmup iterations must be a positive integer")
        self._warmed = False
        for _ in range(iterations):
            self.run()
        torch_npu.npu.synchronize(self._out.device)
        self._warmed = True

    def run(self, q=None, k=None, v=None, *, return_lse=False):
        """Execute with prepared resources; capture exactly one call in recording()."""
        import torch
        import torch_npu

        self._check_bindings(q, k, v)
        capturing = torch_npu.npu.is_current_stream_capturing()
        if capturing and self._state != "recording":
            raise SchemaError("capture run requires warmup() and wrapper.recording(graph)")
        if self._state not in ("planned", "recording"):
            raise SchemaError("captured wrapper requires before_replay() and replay(); use a new wrapper for run")
        if self._state == "recording" and self._recorded_runs:
            raise SchemaError("recording requires exactly one run per wrapper; use a new wrapper")
        _record_kernel_call(self._descriptor, self._operation, self._arguments)
        torch.div(self._natural_lse, math.log(2), out=self._lse)
        if self._state == "recording":
            self._recorded_runs += 1
        return (self._out, self._lse) if return_lse else self._out

    @contextmanager
    def recording(self, graph):
        """Bind one caller graph inside its capture; record one prepared run."""
        from ..npu import ReplayParameterUpdates
        import torch_npu

        self._check_bindings()
        if self._state != "planned" or not self._warmed:
            raise SchemaError("recording requires plan() and warmup() outside capture; otherwise use a new wrapper")
        if not isinstance(graph, torch_npu.npu.NPUGraph):
            raise SchemaError("recording requires the caller's NPUGraph")
        if not torch_npu.npu.is_current_stream_capturing():
            raise SchemaError("recording requires an active caller-owned NPU graph capture")
        if graph.auto_dispatch_capture:
            raise SchemaError("recording requires auto_dispatch_capture=False for explicit task updates")
        if graph in _BOUND_GRAPHS:
            raise SchemaError("one single-decode wrapper per graph is supported; create a new graph")
        if torch_npu.npu.current_stream().device != self._out.device:
            raise SchemaError("recording stream must use the planned tensor device")
        _BOUND_GRAPHS.add(graph)
        updates = ReplayParameterUpdates()
        self._state, self._recorded_runs = "recording", 0
        try:
            with updates.recording():
                yield self
                if self._recorded_runs != 1:
                    raise SchemaError("recording requires exactly one run")
        except BaseException:
            self._state = "failed"
            raise
        self._updates, self._graph = updates, graph
        self._state = "captured"

    def before_replay(self, kv_seq_len, *, stream):
        """Patch KV length and order task updates before the next guarded replay."""
        import torch_npu

        self._check_bindings()
        self._outside_capture("before_replay()")
        if self._state != "captured":
            raise SchemaError("before_replay requires completed capture")
        if self._pending:
            self._pending, self._state = False, "failed"
            raise SchemaError("unconsumed before_replay update; discard graph and use a new wrapper")
        length = self._length(kv_seq_len, self._capacity)
        try:
            self._updates.before_replay(({"kv_seq_lengths": [length]},), stream=stream)
        except BaseException:
            # A device update may have partially patched nodes. Fail closed.
            self._state = "failed"
            raise
        self._replay_stream = torch_npu.npu.current_stream()
        self._pending = True

    def replay(self):
        """Replay the bound graph once, consuming a fresh before_replay update."""
        import torch_npu

        self._check_bindings()
        self._outside_capture("replay()")
        if self._state != "captured" or not self._pending:
            raise SchemaError("every replay requires a fresh before_replay(kv_seq_len, stream=...) call")
        if torch_npu.npu.current_stream() != self._replay_stream:
            raise SchemaError("replay must use the same current stream as before_replay")
        self._pending = False
        try:
            self._graph.replay()
        except BaseException:
            self._state = "failed"
            raise
        return self._out
