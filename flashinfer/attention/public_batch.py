"""Public Attention lifecycle shared by paged and ragged frontends.

ReferenceTensor and installed package adapters retain their existing contracts.
Real NPU tensors use the main kernel registry and an eager FIA composite.
"""

from copy import copy
from dataclasses import replace
import math
import warnings

from flashinfer.runtime import InputSignature, current_hardware, dispatch, get_global_registry
from flashinfer.runtime.schema import SchemaError

from .batch import HostBatchReferenceWrapper, require_no_host_plan_extensions, validate_graph_buffer
from .execution_identity import AttentionGraphResourceContract, AttentionPersistentBufferSpec
from .frontend import (
    adapt_batch_custom_mask, adapt_head_scale, adapt_paged_kv_data,
    adapt_ragged_kv_data, canonicalize_dtype_name, canonicalize_kv_dtype,
    finite_scalar, framework_index_values, parse_kv_layout,
    parse_pos_encoding_mode, validate_framework_workspace_buffer,
)
from .planner import AttentionFrameworkSession, AttentionStateError
from .reference import ReferenceBuffer, ReferenceTensor
from .schema import (
    AttentionMode, AttentionPlanSpec, CustomMaskSpec, KVLayout,
    PagedKVMetadata, PagedPrefillMetadata, RaggedKVMetadata,
)
from .workspace import AttentionWorkspaceContract


def _real_tensor(value):
    return callable(getattr(value, "data_ptr", None))


class PublicBatchWrapper(HostBatchReferenceWrapper):
    def _init_facade(self, mode, values):
        self._mode = mode
        self._kv_layout = parse_kv_layout(values["kv_layout"])
        self._backend = values["backend"]
        self._use_tensor_cores = bool(values.get("use_tensor_cores", False))
        self._native_backend = _real_tensor(values["float_workspace_buffer"]) and self._backend != "reference"
        self._max_total_num_rows = values.get("max_total_num_rows")
        if values.get("variant_owns_mask") or values.get("jit_args") is not None or values.get("jit_kwargs") is not None:
            raise NotImplementedError("custom JIT variants are not implemented by the batch facade")
        graph = bool(values["use_cuda_graph"])
        self._use_max_workspace = bool(values.get("use_max_workspace", False))
        self._native_workspace_query_only = False
        self._native_max_workspace_key = None
        if mode == AttentionMode.BATCH_PREFILL_PAGED:
            names = ("qo_indptr_buf", "paged_kv_indptr_buf", "paged_kv_indices_buf", "paged_kv_last_page_len_buf")
            fixed = values.get("qo_indptr_buf")
        elif mode == AttentionMode.BATCH_PREFILL_RAGGED:
            names = ("qo_indptr_buf", "kv_indptr_buf")
            fixed = values.get("qo_indptr_buf")
        else:
            names = ("paged_kv_indptr_buffer", "paged_kv_indices_buffer", "paged_kv_last_page_len_buffer")
            fixed = values.get("paged_kv_indptr_buffer")
        buffers = tuple(values.get(name) for name in names)
        fixed_batch_size = None
        if graph and not self._native_backend and self._backend != "reference" and not isinstance(values["float_workspace_buffer"], ReferenceTensor):
            raise NotImplementedError("provider graph resources are not bound to public batch wrappers")
        if graph:
            if any(value is None for value in buffers):
                raise SchemaError("graph mode requires all fixed metadata buffers")
            for name, value in zip(names, buffers):
                validate_graph_buffer(value, name, dtype="int32")
                if str(value.device) != str(values["float_workspace_buffer"].device):
                    raise SchemaError("fixed graph buffers and workspace must share one device")
                if self._native_backend and not _real_tensor(value):
                    raise TypeError("CANN graph buffers must be writable tensors")
            fixed_batch_size = fixed.shape[0] - 1
            if fixed_batch_size <= 0:
                raise SchemaError("fixed metadata buffers must describe a positive batch size")
            if mode == AttentionMode.BATCH_PREFILL_PAGED:
                validate_graph_buffer(buffers[1], names[1], dtype="int32", length=fixed_batch_size + 1)
                validate_graph_buffer(buffers[3], names[3], dtype="int32", length=fixed_batch_size)
            elif mode == AttentionMode.BATCH_PREFILL_RAGGED:
                validate_graph_buffer(buffers[1], names[1], dtype="int32", length=fixed_batch_size + 1)
            else:
                validate_graph_buffer(buffers[2], names[2], dtype="int32", length=fixed_batch_size)
            custom, maskptr = values.get("custom_mask_buf"), values.get("mask_indptr_buf")
            if (custom is None) != (maskptr is None):
                raise SchemaError("custom_mask_buf and mask_indptr_buf must be supplied together")
            if custom is not None:
                validate_graph_buffer(custom, "custom_mask_buf", dtype="uint8")
                validate_graph_buffer(maskptr, "mask_indptr_buf", dtype="int32", length=fixed_batch_size + 1)
                if self._native_backend and not all(_real_tensor(value) for value in (custom, maskptr)):
                    raise TypeError("CANN graph buffers must be writable tensors")
                if any(str(value.device) != str(fixed.device) for value in (custom, maskptr)):
                    raise SchemaError("fixed graph buffers and workspace must share one device")
                buffers += (custom, maskptr)
        self._fixed_graph_buffers = buffers if graph else ()
        self._graph_plan_state = None
        self._fixed_block_tables = None
        self._fixed_block_table_binding = None
        self._graph_buffer_bindings = tuple(self._graph_buffer_binding(value) for value in self._fixed_graph_buffers)
        if self._native_backend:
            workspace = validate_framework_workspace_buffer(values["float_workspace_buffer"], "float_workspace_buffer")
            if str(workspace.device).split(":", 1)[0] != "npu":
                raise SchemaError("CANN workspace must be on NPU")
            if self._backend not in ("auto", "cann"):
                raise NotImplementedError("batch tensor execution currently supports backend='auto' or 'cann'")
            import torch

            if graph:
                for i, value in enumerate(buffers):
                    if any(torch._C._overlaps(value, other) for other in (workspace,) + buffers[:i]):
                        raise SchemaError("fixed graph buffers and workspace cannot alias")
            self._operator_runtime = None
            self._session = AttentionFrameworkSession(mode, graph_enabled=graph, fixed_batch_size=fixed_batch_size)
            self._float_workspace_buffer = workspace
            self._int_workspace_buffer = torch.empty(0, dtype=torch.uint8, device=workspace.device)
            self._graph_workspace_bindings = tuple(self._graph_buffer_binding(value) for value in
                                                   (workspace, self._int_workspace_buffer))
            self._workspace_contract = AttentionWorkspaceContract(
                "cann", str(workspace.device), workspace.numel(), 0, graph_enabled=graph,
            )
            self._graph_resources = (
                AttentionGraphResourceContract(mode, True, fixed_batch_size, tuple(
                    AttentionPersistentBufferSpec(name.removesuffix("_buffer").removesuffix("_buf"),
                                                  canonicalize_dtype_name(value.dtype), value.numel(), str(value.device))
                    for name, value in zip(names + ("custom_mask_buf", "mask_indptr_buf"), buffers)
                )) if graph else AttentionGraphResourceContract.disabled(mode)
            )
            self._capture_record = None
            self._capture_generation = 0
            self._custom_mask_data = None
            self._native_decision = None
            return
        if mode == AttentionMode.BATCH_DECODE_PAGED and self._use_tensor_cores and self._backend != "reference":
            raise NotImplementedError("provider matrix-core preference binding is not implemented")
        self._init_host_wrapper(
            mode=mode, float_workspace_buffer=values["float_workspace_buffer"],
            backend=self._backend, graph_enabled=graph,
            fixed_batch_size=fixed_batch_size, graph_buffers=buffers if graph else (),
            provider_runtime_enabled=True,
        )

    @property
    def is_planning_done(self):
        return self._operator_runtime.is_planned if self._operator_runtime is not None else self._session.is_planned

    @property
    def resolved_backend(self):
        _ = self.plan_state
        return "cann" if self._native_backend else self.plan_selection.backend

    @property
    def plan_selection(self):
        if not self._native_backend:
            return super().plan_selection
        from .plan_selection import AttentionPlanSelection

        plan = self.plan_state
        return AttentionPlanSelection(
            plan.spec.mode, "builtin", "cann", plan.generation, plan.fingerprint,
            kernel_id=self._native_decision.kernel_id,
            registry_generation=get_global_registry().revision,
        )

    @staticmethod
    def _graph_buffer_binding(value):
        if isinstance(value, ReferenceTensor):
            return (id(value), value.shape, value.dtype, value.device)
        return (value.data_ptr(), tuple(value.shape), tuple(value.stride()),
                canonicalize_dtype_name(value.dtype), str(value.device))

    def _validate_graph_inputs(self, inputs):
        bindings = tuple(self._graph_buffer_binding(value) for value in self._fixed_graph_buffers)
        if bindings != self._graph_buffer_bindings:
            raise AttentionStateError("fixed graph buffer address/shape/dtype/device changed")
        if len(inputs) != len(self._fixed_graph_buffers):
            raise SchemaError("graph custom masks require fixed custom_mask_buf and mask_indptr_buf")
        for buffer, source in zip(self._fixed_graph_buffers, inputs):
            size = source.numel() if _real_tensor(source) else len(source.data) if isinstance(source, ReferenceTensor) else len(source)
            if size > buffer.shape[0]:
                raise SchemaError("graph buffer capacity is too small")

    def _validate_block_tables(self, table, metadata):
        """Bind a caller-owned dense table only when it agrees with the CSR plan."""

        if table is None:
            return
        if not self._native_backend or not self.is_graph_enabled or self._mode == AttentionMode.BATCH_PREFILL_RAGGED:
            raise NotImplementedError("Host reference plan does not implement: block_tables")
        if not _real_tensor(table) or len(table.shape) != 2 or canonicalize_dtype_name(table.dtype) != "int32":
            raise SchemaError("block_tables must be a rank-2 int32 tensor")
        if str(table.device) != self._workspace_contract.device or not table.is_contiguous():
            raise SchemaError("block_tables must be contiguous on the wrapper device")
        pages = metadata.paged_kv if isinstance(metadata, PagedPrefillMetadata) else metadata
        if table.shape[0] != pages.batch_size:
            raise SchemaError("block_tables batch size does not match paged metadata")
        rows = table.tolist()
        for r, (start, end) in enumerate(zip(pages.indptr, pages.indptr[1:])):
            if table.shape[1] < end - start:
                raise SchemaError("block_tables width cannot hold every planned sequence")
            if tuple(rows[r][:end - start]) != pages.indices[start:end]:
                raise SchemaError("block_tables does not match paged KV indices")
        import torch

        if any(torch._C._overlaps(table, other) for other in (self._float_workspace_buffer,) + self._fixed_graph_buffers):
            raise SchemaError("block_tables cannot alias fixed metadata or workspace")

    def _copy_graph_inputs(self, inputs, *, non_blocking):
        # Immutable ReferenceTensor buffers model structural contracts only.
        # Physical address guarantees apply to caller-owned framework tensors.
        if not self._native_backend:
            return
        import torch

        for buffer, source in zip(self._fixed_graph_buffers, inputs):
            if not _real_tensor(source):
                source = torch.as_tensor(source.data if isinstance(source, ReferenceTensor) else source,
                                         dtype=buffer.dtype, device="cpu")
            buffer[:source.numel()].view(source.shape).copy_(source, non_blocking=non_blocking)

    def _graph_inputs(self, metadata, sources, mask_data, *, packed=False):
        if not self.is_graph_enabled:
            return ()
        inputs = tuple(sources)
        if len(self._fixed_graph_buffers) > len(inputs):
            if mask_data is None:
                return inputs + ((), (0,) * (metadata.batch_size + 1))
            offsets = [0]
            lengths = metadata.paged_kv.sequence_lengths if isinstance(metadata, PagedPrefillMetadata) else metadata.kv_lengths
            for q_len, kv_len in zip(metadata.qo_lengths, lengths):
                size = q_len * kv_len
                offsets.append(offsets[-1] + ((size + 7) // 8 if packed else size))
            return inputs + (mask_data, tuple(offsets))
        if mask_data is not None:
            raise SchemaError("graph custom masks require fixed custom_mask_buf and mask_indptr_buf")
        return inputs

    def _plan_prefill(self, values):
        values = dict(values)
        values.pop("self", None)
        reader = framework_index_values
        qo = reader(values["qo_indptr"], "qo_indptr")
        if self._mode == AttentionMode.BATCH_PREFILL_PAGED:
            paged = PagedKVMetadata(
                reader(values["paged_kv_indptr"], "paged_kv_indptr"),
                reader(values["paged_kv_indices"], "paged_kv_indices"),
                reader(values["paged_kv_last_page_len"], "paged_kv_last_page_len"),
                values["page_size"],
            )
            metadata = PagedPrefillMetadata(qo, paged)
            lengths = paged.sequence_lengths
        else:
            metadata = RaggedKVMetadata(qo, reader(values["kv_indptr"], "kv_indptr"))
            lengths = metadata.kv_lengths
        for name, actual in (("qo_indptr_cpu", qo), ("kv_indptr_cpu", getattr(metadata, "kv_indptr", None))):
            hint = values.get(name)
            if hint is not None and reader(hint, name) != actual:
                raise SchemaError(name + " does not match device metadata")
        uniform = values.get("uniform_q_len")
        if uniform is not None and any(n != uniform for n in metadata.qo_lengths):
            raise SchemaError("uniform_q_len does not match query lengths")
        if self._max_total_num_rows is not None and metadata.total_qo_tokens > self._max_total_num_rows:
            raise SchemaError("query rows exceed max_total_num_rows")
        extensions = {key: values.get(key) for key in (
            "prefix_len_ptr", "token_pos_in_items_ptr", "token_pos_in_items_len",
            "max_item_len_ptr", "seq_lens", "seq_lens_q", "block_tables",
            "max_token_per_sequence", "max_sequence_kv", "v_indptr", "o_indptr",
        )}
        if values.get("fixed_split_size") is not None or values.get("disable_split_kv"):
            raise NotImplementedError("split-K planning controls are not implemented")
        block_tables = extensions.pop("block_tables")
        require_no_host_plan_extensions(**extensions)
        self._validate_block_tables(block_tables, metadata)
        custom, packed = values.get("custom_mask"), values.get("packed_custom_mask")
        if self._operator_runtime is not None and (custom is not None or packed is not None):
            raise NotImplementedError("provider custom-mask plan binding is not implemented")
        if self._native_backend:
            mask_data = packed if packed is not None else custom
            mask = None
            if mask_data is not None:
                import torch

                expected = sum((q * k + 7) // 8 if packed is not None else q * k
                               for q, k in zip(metadata.qo_lengths, lengths))
                dtype = torch.uint8 if packed is not None else torch.bool
                if mask_data.numel() != expected or mask_data.dtype != dtype or mask_data.device != self._float_workspace_buffer.device:
                    raise SchemaError("custom mask shape/dtype/device mismatch")
                mask = CustomMaskSpec(expected, packed=packed is not None)
        else:
            mask, mask_data = adapt_batch_custom_mask(
                custom, packed,
                segment_sizes=tuple(q * k for q, k in zip(metadata.qo_lengths, lengths)),
                device=self._workspace_contract.device,
            )
        q_dtype = canonicalize_dtype_name(values["q_data_type"])
        kv_dtype, quant = canonicalize_kv_dtype(values.get("kv_data_type"), q_dtype)
        spec = AttentionPlanSpec(
            self._mode, values["num_qo_heads"], values["num_kv_heads"], values["head_dim_qk"],
            head_dim_vo=values.get("head_dim_vo"), kv_layout=self._kv_layout,
            q_dtype=q_dtype, kv_dtype=kv_dtype, kv_quant_spec=quant,
            o_dtype=q_dtype if values.get("o_data_type") is None else canonicalize_dtype_name(values["o_data_type"]),
            causal=bool(values["causal"]), custom_mask=mask,
            pos_encoding_mode=parse_pos_encoding_mode(values["pos_encoding_mode"]),
            sm_scale=None if values["sm_scale"] is None else finite_scalar(values["sm_scale"], "sm_scale"),
            window_left=values["window_left"], logits_soft_cap=values["logits_soft_cap"],
            rope_scale=values["rope_scale"], rope_theta=values["rope_theta"],
            use_fp16_qk_reduction=values["use_fp16_qk_reduction"],
        )
        sources = ((values["qo_indptr"], values["paged_kv_indptr"], values["paged_kv_indices"], values["paged_kv_last_page_len"])
                   if self._mode == AttentionMode.BATCH_PREFILL_PAGED else (values["qo_indptr"], values["kv_indptr"]))
        self._publish_plan(spec, metadata, mask_data,
                           graph_inputs=self._graph_inputs(metadata, sources, mask_data, packed=mask is not None and mask.packed),
                           block_tables=block_tables, non_blocking=values["non_blocking"])

    def _plan_decode(self, values):
        legacy_names = (
            "pos_encoding_mode", "window_left", "logits_soft_cap", "q_data_type",
            "kv_data_type", "o_data_type", "data_type", "sm_scale", "rope_scale",
            "rope_theta", "non_blocking", "block_tables", "seq_lens",
            "fixed_split_size", "disable_split_kv", "q_len_per_req",
        )
        options = dict(
            pos_encoding_mode="NONE", window_left=-1, window_right=0,
            logits_soft_cap=None, q_data_type="float16", kv_data_type=None,
            o_data_type=None, data_type=None, sm_scale=None, rope_scale=None,
            rope_theta=None, non_blocking=True, block_tables=None, seq_lens=None,
            fixed_split_size=None, disable_split_kv=False, q_len_per_req=1,
            is_causal=None,
        )
        given = dict(values.get("kwargs", {}))
        positional = values.get("deprecated_positional_args", ())
        if len(positional) > len(legacy_names):
            raise TypeError("too many deprecated plan arguments")
        if positional:
            warnings.warn("plan options after page_size must be passed by keyword", DeprecationWarning, stacklevel=3)
            for name, item in zip(legacy_names, positional):
                if name in given:
                    raise TypeError("multiple values for " + name)
                given[name] = item
        unknown = set(given) - set(options)
        if unknown:
            raise TypeError("unexpected plan keyword argument: " + sorted(unknown)[0])
        options.update(given)
        q_len = options["q_len_per_req"]
        if not isinstance(q_len, int) or q_len < 1:
            raise ValueError("q_len_per_req must be a positive integer")
        if q_len > 1 and not self._use_tensor_cores:
            raise ValueError("q_len_per_req > 1 requires use_tensor_cores=True")
        if options["is_causal"] not in (None, q_len > 1):
            raise ValueError("is_causal must match q_len_per_req > 1")
        if options["window_right"] != 0:
            raise ValueError("window_right != 0 requires a backend that supports it")
        if options["fixed_split_size"] is not None or options["disable_split_kv"]:
            raise NotImplementedError("split-K planning controls are not implemented")
        require_no_host_plan_extensions(seq_lens=options["seq_lens"])
        metadata = PagedKVMetadata(
            framework_index_values(values["indptr"], "indptr"),
            framework_index_values(values["indices"], "indices"),
            framework_index_values(values["last_page_len"], "last_page_len"),
            values["page_size"],
        )
        self._validate_block_tables(options["block_tables"], metadata)
        if options["data_type"] is not None:
            options["q_data_type"] = options["q_data_type"] or options["data_type"]
            options["kv_data_type"] = options["kv_data_type"] or options["data_type"]
        q_dtype = canonicalize_dtype_name(options["q_data_type"])
        kv_dtype, quant = canonicalize_kv_dtype(options["kv_data_type"], q_dtype)
        spec = AttentionPlanSpec(
            self._mode, values["num_qo_heads"], values["num_kv_heads"], values["head_dim"],
            kv_layout=self._kv_layout, q_dtype=q_dtype, kv_dtype=kv_dtype,
            o_dtype=q_dtype if options["o_data_type"] is None else canonicalize_dtype_name(options["o_data_type"]),
            kv_quant_spec=quant, q_len_per_req=q_len, causal=q_len > 1,
            pos_encoding_mode=parse_pos_encoding_mode(options["pos_encoding_mode"]),
            sm_scale=None if options["sm_scale"] is None else finite_scalar(options["sm_scale"], "sm_scale"),
            window_left=options["window_left"], logits_soft_cap=options["logits_soft_cap"],
            rope_scale=options["rope_scale"], rope_theta=options["rope_theta"],
        )
        self._publish_plan(spec, metadata, None,
                           graph_inputs=self._graph_inputs(metadata, (values["indptr"], values["indices"], values["last_page_len"]), None),
                           block_tables=options["block_tables"], non_blocking=options["non_blocking"])

    def _publish_plan(self, spec, metadata, mask_data, *, graph_inputs=(), block_tables=None, non_blocking=True):
        if self._native_backend:
            if spec.pos_encoding_mode.value != "NONE" or spec.logits_soft_cap != 0 or spec.use_fp16_qk_reduction or spec.kv_quant_spec is not None:
                raise NotImplementedError("CANN facade supports dense KV, NONE position mode, no soft-cap or FP16 QK reduction")
            signature = InputSignature(
                "attention." + spec.mode.value, (spec.q_dtype, spec.kv_dtype),
                layouts=(spec.kv_layout.value,), specializations=(("head_dim", str(spec.head_dim_qk)),),
            )
            candidate = self._session.prepare_plan(spec, metadata)
            decision = dispatch(signature, current_hardware(), backend="cann")
            descriptor = get_global_registry().get(decision.kernel_id)
            graph_state = self._graph_resources._validate_plan(
                candidate, resolved_backend=decision.backend.value,
                kernel_identity=(decision.kernel_id, descriptor.fingerprint),
                block_table_width=None if block_tables is None else block_tables.shape[1],
                frozen=self._graph_plan_state,
            )
            if self.is_graph_enabled:
                self._validate_graph_inputs(graph_inputs)
                if self._fixed_block_tables is not None and self._graph_buffer_binding(self._fixed_block_tables) != self._fixed_block_table_binding:
                    raise AttentionStateError("fixed block_tables address/shape/dtype/device changed")
            workspace, workspace_key = self._prepare_native_workspace(candidate)
            if self._native_workspace_query_only:
                self._workspace_contract = workspace
                return
            if self.is_graph_enabled:
                self._copy_graph_inputs(graph_inputs, non_blocking=non_blocking)
                if block_tables is not None and self._fixed_block_tables is not None:
                    self._fixed_block_tables.copy_(block_tables, non_blocking=non_blocking)
            self._session.commit_prepared_plan(candidate)
            self._native_decision = decision
            self._workspace_contract = workspace
            self._native_max_workspace_key = workspace_key
            self._graph_plan_state = graph_state
            if self.is_graph_enabled and block_tables is not None and self._fixed_block_tables is None:
                self._fixed_block_tables = block_tables
                self._fixed_block_table_binding = self._graph_buffer_binding(block_tables)
                self._graph_resources = replace(self._graph_resources, persistent_buffers=self._graph_resources.persistent_buffers + (
                    AttentionPersistentBufferSpec("block_tables", "int32", block_tables.numel(), str(block_tables.device)),
                ))
            if self.is_graph_enabled and mask_data is not None:
                fixed_mask = self._fixed_graph_buffers[-2][:mask_data.numel()]
                mask_data = fixed_mask if spec.custom_mask.packed else fixed_mask.view(mask_data.dtype)
            self._custom_mask_data = mask_data
            return
        if self.is_graph_enabled:
            candidate = self._session.prepare_plan(spec, metadata)
            graph_state = self._graph_resources._validate_plan(candidate, resolved_backend="reference", frozen=self._graph_plan_state)
            self._validate_graph_inputs(graph_inputs)
        self._commit_plan(spec, metadata, mask_data)
        if self.is_graph_enabled:
            self._graph_plan_state = graph_state

    def _validate_native_workspace_binding(self):
        bindings = tuple(self._graph_buffer_binding(value) for value in
                         (self._float_workspace_buffer, self._int_workspace_buffer))
        if bindings != self._graph_workspace_bindings:
            raise AttentionStateError("fixed graph workspace address/shape/dtype/device cannot change")

    def _prepare_native_workspace(self, candidate):
        """Bind requirements through T9's existing caller-owned workspace contract."""
        if not self._native_workspace_query_only and not (self.is_graph_enabled and self._use_max_workspace):
            return self._workspace_contract.bind_plan(candidate.generation), None
        if not self._native_workspace_query_only:
            self._validate_native_workspace_binding()
            # torch_npu's .out kernel passes storage().data(), not data_ptr().
            # A sliced workspace would therefore write before the bound view.
            if not self._float_workspace_buffer.is_contiguous() or self._float_workspace_buffer.storage_offset() != 0:
                raise SchemaError("CANN maximum workspace must be contiguous with storage_offset=0")
        from flashinfer.cann.attention import maximum_workspace_bytes

        spec, metadata = candidate.spec, candidate.metadata
        if self._mode == AttentionMode.BATCH_DECODE_PAGED:
            query_lengths = (spec.q_len_per_req,) * metadata.batch_size
            kv_lengths = metadata.sequence_lengths
        else:
            query_lengths = metadata.qo_lengths
            kv_lengths = (metadata.paged_kv.sequence_lengths if isinstance(metadata, PagedPrefillMetadata)
                          else metadata.kv_lengths)
        # The vendor maximum covers changing KV sequence lengths. Re-query on
        # other changes allowed by T9, without freezing any additional dimension
        # or replacing storage. LSE on/off and all requests share one byte buffer.
        key = (spec.q_dtype, spec.kv_dtype, spec.num_qo_heads, spec.num_kv_heads,
               spec.head_dim_qk, spec.head_dim_vo, spec.sm_scale,
               spec.effective_causal or spec.window_left >= 0 or spec.custom_mask is not None,
               tuple(sorted(set(query_lengths))))
        required = (self._workspace_contract.required_float_bytes if key == self._native_max_workspace_key
                    else maximum_workspace_bytes(spec, query_lengths, kv_lengths, device=self._workspace_contract.device))
        return self._workspace_contract.bind_requirements(
            required_float_bytes=required, required_int_bytes=0, plan_generation=candidate.generation,
        ), key

    def _run_facade(self, q, kv, values):
        values = dict(values)
        spec = self.plan_state.spec
        # CUDA PDL is a hardware launch hint; upstream disables it on NPU.
        # Preserve the public argument while explicitly discarding the hint.
        values.pop("enable_pdl", None)
        if values.get("q_len_per_req") not in (None, spec.q_len_per_req):
            raise SchemaError("run q_len_per_req must match the plan")
        if values.get("lse_base", "log2") not in ("log2", "ln") or values.get("lse_layout", "NH") not in ("NH", "HN"):
            raise ValueError("lse_base must be log2/ln and lse_layout must be NH/HN")
        if self._operator_runtime is not None:
            for label, enabled in (
                ("query-scale", values.get("q_scale") is not None),
                ("o_scale", values.get("o_scale") is not None),
                ("attention-sink", values.get("sinks") is not None),
                ("LSE-buffer", values.get("lse") is not None),
                ("SP-compression", values.get("uses_spcompress") is True),
                ("FP16 softmax", values.get("use_fp16_softmax") is True),
                ("skip-softmax", values.get("skip_softmax_threshold_scale_factor") is not None),
                ("custom JIT", bool(values.get("args"))),
            ):
                if enabled:
                    raise NotImplementedError("provider " + label + " binding is not implemented")
            if values.get("window_left") not in (None, spec.window_left):
                raise SchemaError("run window_left must match the plan")
            if self._mode == AttentionMode.BATCH_PREFILL_RAGGED and spec.kv_quant_spec is not None:
                from .operator_quantization import combine_attention_operator_quantized_kv_input

                kv = combine_attention_operator_quantized_kv_input(*kv)
            return self._operator_runtime.run(
                q, kv, return_lse=values.get("return_lse", False),
                out=values.get("out"), k_scale=values.get("k_scale"),
                v_scale=values.get("v_scale"), kv_cache_sf=values.get("kv_cache_sf"),
            )
        if values.get("o_scale") is not None:
            raise NotImplementedError("o_scale quantized output is not implemented")
        wants_lse = bool(values.get("return_lse")) or values.get("lse") is not None
        if self._native_backend:
            for name in ("sinks", "kv_cache_sf", "skip_softmax_threshold_scale_factor"):
                if values.get(name) is not None:
                    raise NotImplementedError("CANN " + name + " is not implemented")
            if values.get("args") or values.get("use_fp16_softmax") or values.get("uses_spcompress"):
                raise NotImplementedError("CANN JIT/FP16-softmax/SP-compression controls are not implemented")
            if values.get("window_left") not in (None, spec.window_left):
                raise SchemaError("run window_left must match the plan")
            result = get_global_registry().get(self._native_decision.kernel_id).runner(
                self, q, kv, return_lse=wants_lse,
                q_scale=values.get("q_scale"), k_scale=values.get("k_scale"), v_scale=values.get("v_scale"),
            )
            output, lse = result if wants_lse else (result, None)
            if values.get("out") is not None:
                self._copy_tensor_result(values["out"], output, "out")
                output = values["out"]
            natural = False
        else:
            if self._mode == AttentionMode.BATCH_PREFILL_RAGGED:
                kv_data = adapt_ragged_kv_data(
                    *kv, total_kv_tokens=self.plan_state.metadata.total_kv_tokens,
                    num_kv_heads=spec.num_kv_heads, head_dim_qk=spec.head_dim_qk,
                    head_dim_vo=spec.head_dim_vo, dtype=spec.kv_dtype, layout=spec.kv_layout,
                    quant_spec=spec.kv_quant_spec,
                )
            else:
                metadata = self.plan_state.metadata
                pages = metadata.paged_kv if isinstance(metadata, PagedPrefillMetadata) else metadata
                kv_data = adapt_paged_kv_data(
                    kv, page_size=pages.page_size, num_kv_heads=spec.num_kv_heads,
                    head_dim_qk=spec.head_dim_qk, head_dim_vo=spec.head_dim_vo,
                    dtype=spec.kv_dtype, layout=spec.kv_layout, quant_spec=spec.kv_quant_spec,
                )
            scales = [adapt_head_scale(values.get(name), name=name, num_heads=heads, device=q.device)
                      for name, heads in (("q_scale", spec.num_qo_heads), ("k_scale", spec.num_kv_heads), ("v_scale", spec.num_kv_heads))]
            if self.is_graph_enabled and values.get("lse_layout", "NH") == "HN":
                raise NotImplementedError("HN LSE graph resources are not bound yet (T9)")
            result = self._execute_reference(
                q, kv_data, args=values.get("args", ()),
                q_scale=scales[0], k_scale=scales[1], v_scale=scales[2],
                out=values.get("out"),
                lse=values.get("lse") if values.get("lse_layout", "NH") == "NH" else None,
                return_lse=wants_lse, capture_return_lse=bool(values.get("return_lse")),
                enable_pdl=values.get("enable_pdl"), window_left=values.get("window_left"),
                sinks=values.get("sinks"), kv_cache_sf=values.get("kv_cache_sf"),
                skip_softmax_threshold_scale_factor=values.get("skip_softmax_threshold_scale_factor"),
                use_fp16_softmax=values.get("use_fp16_softmax"), uses_spcompress=values.get("uses_spcompress"),
            )
            output, lse = result if wants_lse else (result, None)
            natural = True
        if wants_lse:
            lse = self._finish_lse(lse, natural, values.get("lse_base", "log2"), values.get("lse_layout", "NH"), values.get("lse"))
        return (output, lse) if values.get("return_lse") else output

    @staticmethod
    def _copy_tensor_result(out, value, name):
        if out.shape != value.shape or out.dtype != value.dtype or out.device != value.device:
            raise SchemaError(name + " shape/dtype/device mismatch")
        out.copy_(value)

    @staticmethod
    def _finish_lse(value, natural, base, layout, out):
        factor = (1 / math.log(2) if base == "log2" else 1) if natural else (math.log(2) if base == "ln" else 1)
        if isinstance(value, (ReferenceTensor, ReferenceBuffer)):
            data = tuple(x * factor for x in value.data)
            shape = value.shape
            if layout == "HN":
                n, h = shape
                data = tuple(data[i * h + j] for j in range(h) for i in range(n))
                shape = (h, n)
            result = ReferenceTensor(shape, data, value.dtype, value.device)
            if out is not None:
                if not isinstance(out, ReferenceBuffer):
                    raise TypeError("lse must be a ReferenceBuffer")
                out.copy_from(result)
                result = out
        else:
            result = value * factor
            if layout == "HN":
                result = result.t().contiguous()
            if out is not None:
                PublicBatchWrapper._copy_tensor_result(out, result, "lse")
                result = out
        return result

    def _run_native(self, q, kv, *, return_lse, q_scale, k_scale, v_scale):
        import torch
        from flashinfer.cann.attention import dense_attention

        plan = self.plan_state
        spec, metadata = plan.spec, plan.metadata
        self._workspace_contract.validate_run(device=str(q.device), plan_generation=plan.generation)
        workspace_options = {}
        if self.is_graph_enabled and self._use_max_workspace:
            self._validate_native_workspace_binding()
            workspace_options["workspace"] = self._float_workspace_buffer
        if tuple(q.shape) != plan._expected_query_shape() or str(q.dtype).removeprefix("torch.") != spec.q_dtype:
            raise SchemaError("q shape/dtype does not match the plan")
        if self._mode == AttentionMode.BATCH_PREFILL_RAGGED:
            keys, vals = kv
            if self._kv_layout == KVLayout.HND:
                keys, vals = keys.transpose(0, 1), vals.transpose(0, 1)
            if keys.shape != (metadata.total_kv_tokens, spec.num_kv_heads, spec.head_dim_qk) or vals.shape != (metadata.total_kv_tokens, spec.num_kv_heads, spec.head_dim_vo):
                raise SchemaError("ragged k/v shapes do not match the plan")
            kv_lengths = metadata.kv_lengths
            qo = metadata.qo_indptr
        else:
            pages = metadata.paged_kv if isinstance(metadata, PagedPrefillMetadata) else metadata
            if isinstance(kv, torch.Tensor):
                if kv.ndim != 5 or kv.shape[1] != 2:
                    raise SchemaError("packed paged KV must have shape [pages, 2, ...]")
                keys, vals = kv[:, 0], kv[:, 1]
            elif isinstance(kv, (tuple, list)) and len(kv) == 2:
                keys, vals = kv
            else:
                raise TypeError("paged KV must be a packed tensor or a K/V pair")
            if self._kv_layout == KVLayout.HND:
                keys, vals = keys.transpose(1, 2), vals.transpose(1, 2)
            expected = (pages.page_size, spec.num_kv_heads)
            if keys.ndim != 4 or vals.ndim != 4 or keys.shape[1:3] != expected or vals.shape[1:3] != expected or keys.shape[-1] != spec.head_dim_qk or vals.shape[-1] != spec.head_dim_vo or keys.shape[0] != vals.shape[0]:
                raise SchemaError("paged K/V shapes do not match the plan")
            if pages.max_page_index >= keys.shape[0]:
                raise SchemaError("page index exceeds cache capacity")
            if self.is_graph_enabled:
                index_slot = 2 if self._mode == AttentionMode.BATCH_PREFILL_PAGED else 1
                page_ids = self._fixed_graph_buffers[index_slot]
            else:
                page_ids = torch.tensor(pages.indices, dtype=torch.int64, device=q.device)
            kv_lengths = pages.sequence_lengths
            qo = metadata.qo_indptr if isinstance(metadata, PagedPrefillMetadata) else tuple(i * spec.q_len_per_req for i in range(pages.batch_size + 1))
        if keys.device != q.device or vals.device != q.device or str(keys.dtype).removeprefix("torch.") != spec.kv_dtype or vals.dtype != keys.dtype:
            raise SchemaError("K/V dtype/device does not match the plan")
        outputs, lses = [], []
        mask_offset = 0
        for r, (start, end) in enumerate(zip(qo, qo[1:])):
            if self._mode == AttentionMode.BATCH_PREFILL_RAGGED:
                ks, ke = metadata.kv_indptr[r:r + 2]
                key, val = keys[ks:ke], vals[ks:ke]
            else:
                ps, pe = pages.indptr[r:r + 2]
                request_pages = (self._fixed_block_tables[r, :pe - ps]
                                 if self.is_graph_enabled and self._fixed_block_tables is not None else page_ids[ps:pe])
                key = keys.index_select(0, request_pages).reshape(-1, spec.num_kv_heads, spec.head_dim_qk)[:kv_lengths[r]]
                val = vals.index_select(0, request_pages).reshape(-1, spec.num_kv_heads, spec.head_dim_vo)[:kv_lengths[r]]
            mask_options = {}
            if spec.custom_mask is not None:
                count = (end - start) * kv_lengths[r]
                if spec.custom_mask.packed:
                    count = (count + 7) // 8
                    mask_options["packed_custom_mask"] = self._custom_mask_data.flatten()[mask_offset:mask_offset + count]
                else:
                    mask_options["custom_mask"] = self._custom_mask_data.flatten()[mask_offset:mask_offset + count].reshape(end - start, kv_lengths[r])
                mask_offset += count
            result = dense_attention(
                q[start:end], key, val, causal=spec.effective_causal,
                scale_q=q_scale, k_scale=k_scale, v_scale=v_scale,
                o_dtype=spec.o_dtype, sm_scale=spec.sm_scale, window_left=spec.window_left,
                return_lse=return_lse, **mask_options, **workspace_options,
            )
            if return_lse:
                output, lse = result
                lses.append(lse)
            else:
                output = result
            outputs.append(output)
        output = torch.cat(outputs) if outputs else torch.empty(plan._expected_output_shape(), dtype=getattr(torch, spec.o_dtype), device=q.device)
        if return_lse:
            lse = torch.cat(lses) if lses else torch.empty(plan._expected_lse_shape(), device=q.device)
            return output, lse
        return output

    def _workspace_query(self, values):
        if self._native_backend:
            # Explicit queries return the native maximum even for an eager
            # wrapper. No fixed metadata, active plan, or freeze state is written.
            probe = copy(self)
            probe._session = copy(self._session)
            probe._native_workspace_query_only = True
            probe._native_max_workspace_key = None
            probe._workspace_contract = replace(self._workspace_contract, float_capacity_bytes=(1 << 63) - 1)
            values = {k: v for k, v in values.items() if k != "self"}
            values.setdefault("non_blocking", True)
            probe.plan(**values)
            return probe.workspace_contract.required_sizes
        values = {k: v for k, v in values.items() if k != "self"}
        if self._operator_runtime is not None:
            probe = self._provider_workspace_query_probe()
        else:
            probe = copy(self)
            probe._session = copy(self._session)
        probe.plan(**values)
        return probe.workspace_contract.required_sizes

    def _deprecated_forward(self, q, kv, values):
        warnings.warn("forward() is deprecated; use plan()/run()", DeprecationWarning, stacklevel=3)
        spec = self.plan_state.spec
        for name in ("causal", "pos_encoding_mode", "use_fp16_qk_reduction", "window_left", "logits_soft_cap", "sm_scale", "rope_scale", "rope_theta"):
            value = values.get(name)
            if value is not None and value != getattr(spec, name):
                raise SchemaError(name + " must match the plan")
        return self._run_facade(q, kv, {k: values.get(k) for k in ("q_scale", "k_scale", "v_scale")})

    def reset_workspace_buffer(self, float_workspace_buffer, int_workspace_buffer):
        if not self._native_backend:
            return super().reset_workspace_buffer(float_workspace_buffer, int_workspace_buffer)
        first = validate_framework_workspace_buffer(float_workspace_buffer, "float_workspace_buffer")
        second = validate_framework_workspace_buffer(int_workspace_buffer, "int_workspace_buffer", device=str(first.device))
        import torch

        if torch._C._overlaps(first, second):
            raise SchemaError("float and int workspace buffers cannot alias")
        if self.is_graph_enabled and (
            self._graph_buffer_binding(first) != self._graph_buffer_binding(self._float_workspace_buffer)
            or self._graph_buffer_binding(second) != self._graph_buffer_binding(self._int_workspace_buffer)
        ):
            raise AttentionStateError("fixed graph workspace address/shape/dtype/device cannot change")
        contract = self._workspace_contract.rebind(
            device=str(first.device), float_capacity_bytes=first.numel(), int_capacity_bytes=second.numel(), allow_device_change=False,
        )
        self._float_workspace_buffer, self._int_workspace_buffer = first, second
        self._workspace_contract = contract
