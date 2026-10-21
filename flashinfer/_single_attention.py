"""Canonical single-request Attention adapters and explicit execution routes."""

from .attention.frontend import (
    adapt_framework_single_qkv, adapt_head_scale, adapt_single_custom_mask,
    adapt_single_qkv, canonicalize_dtype_name, finite_scalar,
    multiply_scale, optional_finite_scalar, parse_pos_encoding_mode,
    require_reference_backend,
)
from .attention.planner import AttentionFrameworkSession
from .attention.reference import ReferenceAttentionExecutor, ReferenceTensor
from .attention.schema import AttentionMode, AttentionPlanSpec
from .runtime import build_signature, current_hardware, dispatch, get_global_registry
from .runtime.schema import SchemaError


def is_framework_tensor(value):
    return callable(getattr(value, "data_ptr", None))


def single_attention(mode, q, k, v, *, backend="auto", kv_layout="NHD",
                     pos_encoding_mode="NONE", causal=False,
                     scale_q=None, scale_k=None, scale_v=None,
                     k_scale=None, v_scale=None, o_dtype=None, custom_mask=None,
                     packed_custom_mask=None, sm_scale=None, window_left=-1,
                     logits_soft_cap=None, rope_scale=None, rope_theta=None,
                     use_fp16_qk_reduction=False, use_tensor_cores=False,
                     return_lse=False, kv_cache_sf=None):
    mode = AttentionMode(mode)
    if kv_cache_sf is not None:
        raise NotImplementedError("NVFP4 kv_cache_sf is not implemented")
    position = parse_pos_encoding_mode(pos_encoding_mode)
    sm_scale = None if sm_scale is None else finite_scalar(sm_scale, "sm_scale")
    if isinstance(q, ReferenceTensor):
        if mode != AttentionMode.SINGLE_DECODE:
            require_reference_backend(backend)
        adapted = adapt_single_qkv(q, k, v, mode=mode, kv_layout=kv_layout)
        mask, mask_data = adapt_single_custom_mask(
            custom_mask, packed_custom_mask, qo_len=adapted.metadata.qo_len,
            kv_len=adapted.metadata.kv_len, device=q.device,
        )
        spec = AttentionPlanSpec(
            mode, adapted.num_qo_heads, adapted.num_kv_heads, adapted.head_dim_qk,
            head_dim_vo=adapted.head_dim_vo, kv_layout=adapted.layout,
            q_dtype=q.dtype, kv_dtype=k.dtype,
            kv_quant_spec=adapted.kv_data.spec.quant_spec,
            o_dtype=q.dtype if o_dtype is None else canonicalize_dtype_name(o_dtype),
            causal=bool(causal), custom_mask=mask, pos_encoding_mode=position,
            sm_scale=sm_scale, window_left=window_left,
            logits_soft_cap=logits_soft_cap, rope_scale=rope_scale,
            rope_theta=rope_theta, use_fp16_qk_reduction=use_fp16_qk_reduction,
        )
        plan = AttentionFrameworkSession(mode).plan(spec, adapted.metadata)
        scales = []
        for name, value, count, legacy in (
            ("q_scale", scale_q, adapted.num_qo_heads, None),
            ("k_scale", scale_k, adapted.num_kv_heads, k_scale),
            ("v_scale", scale_v, adapted.num_kv_heads, v_scale),
        ):
            scales.append(multiply_scale(
                adapt_head_scale(value, name=name, num_heads=count, device=q.device),
                optional_finite_scalar(legacy, name),
            ))
        result = ReferenceAttentionExecutor().execute(
            plan, q, adapted.kv_data, return_lse=return_lse,
            custom_mask_data=mask_data,
            q_scale=scales[0], k_scale=scales[1], v_scale=scales[2],
        )
        if not return_lse:
            return result.output
        import math

        # Upstream state.cuh::get_lse emits m + log2(d), including single APIs.
        lse = ReferenceTensor(result.lse.shape,
                              tuple(x / math.log(2) for x in result.lse.data),
                              result.lse.dtype, result.lse.device)
        return result.output, lse
    if not is_framework_tensor(q):
        # Preserve the existing external-package route for tensor-like adapters.
        from .attention.holistic import attention_operator_runtime_registry_snapshot
        from .attention.operator_resolver import AttentionOperatorRuntime

        for label, enabled in (
            ("Q/K/V scale (query-scale)" if mode == AttentionMode.SINGLE_DECODE else "per-head scale",
             any(x is not None for x in (scale_q, scale_k, scale_v))),
            ("custom-mask", custom_mask is not None or packed_custom_mask is not None),
            ("KV scale", k_scale is not None or v_scale is not None),
            ("matrix-core", use_tensor_cores),
        ):
            if enabled:
                raise NotImplementedError("provider " + label + " binding is not implemented")
        adapted = adapt_framework_single_qkv(q, k, v, mode=mode, kv_layout=kv_layout)
        spec = AttentionPlanSpec(
            mode, adapted.num_qo_heads, adapted.num_kv_heads, adapted.head_dim_qk,
            head_dim_vo=adapted.head_dim_vo, kv_layout=adapted.layout,
            q_dtype=adapted.q_dtype, kv_dtype=adapted.kv_dtype,
            kv_quant_spec=adapted.kv_quant_spec,
            o_dtype=adapted.q_dtype if o_dtype is None else canonicalize_dtype_name(o_dtype),
            causal=bool(causal), pos_encoding_mode=position, sm_scale=sm_scale,
            window_left=window_left, logits_soft_cap=logits_soft_cap,
            rope_scale=rope_scale, rope_theta=rope_theta,
            use_fp16_qk_reduction=use_fp16_qk_reduction,
        )
        snapshot = attention_operator_runtime_registry_snapshot()
        runtime = AttentionOperatorRuntime(adapted.device, snapshot.registry,
                                           snapshot.operation_catalog, mode=mode)
        runtime.plan(spec, adapted.metadata)
        kv = (k, v)
        if adapted.kv_quant_spec is not None:
            from .attention.operator_quantization import combine_attention_operator_quantized_kv_input

            kv = combine_attention_operator_quantized_kv_input(k, v)
        return runtime.run(q, kv, return_lse=return_lse)
    # CUDA's Tensor Core hint selects an implementation of the same formula;
    # real NPU tensors use the registered CANN implementation in either case.
    del use_tensor_cores
    options = dict(
        decode=mode == AttentionMode.SINGLE_DECODE, kv_layout=kv_layout,
        causal=bool(causal), pos_encoding_mode=pos_encoding_mode,
        scale_q=scale_q, scale_k=scale_k, scale_v=scale_v,
        k_scale=k_scale, v_scale=v_scale, o_dtype=o_dtype,
        custom_mask=custom_mask, packed_custom_mask=packed_custom_mask,
        sm_scale=sm_scale, window_left=window_left, logits_soft_cap=logits_soft_cap,
        rope_scale=rope_scale, rope_theta=rope_theta,
        use_fp16_qk_reduction=use_fp16_qk_reduction, return_lse=return_lse,
    )
    from . import config

    policy = config.default_backend() if backend == "auto" else backend
    if policy == "reference":
        from .cann.attention import dense_attention

        return dense_attention(q, k, v, native=False, **options)
    signature = build_signature(
        "attention." + mode.value, (q, k, v), layouts=(kv_layout,) * 3,
        specializations={"head_dim": q.shape[-1]},
    )
    decision = dispatch(signature, current_hardware(), backend=policy)
    return get_global_registry().get(decision.kernel_id).runner(q, k, v, **options)


def run_single_jit(mode, jit_module, q, k, v, args, *, kv_layout="NHD",
                   mask_mode=0, window_left=-1, return_lse=False):
    from .attention.jit_protocol import (
        ATTENTION_SINGLE_JIT_TEMP_BYTES, freeze_jit_buffer, make_single_jit_buffers,
        record_single_jit_protocol_call, require_jit_run,
        upstream_kv_layout_code, validate_upstream_mask_mode,
    )
    from .attention.frontend import parse_kv_layout

    run = require_jit_run(jit_module)
    layout = parse_kv_layout(kv_layout)
    validate_upstream_mask_mode(mask_mode)
    if window_left < -1:
        raise SchemaError("window_left must be -1 or nonnegative")
    decode = mode == AttentionMode.SINGLE_DECODE
    shape = tuple(q.shape[:-1]) + (v.shape[-1],)
    lse_shape = tuple(q.shape[:-1])
    if isinstance(q, ReferenceTensor):
        adapt_single_qkv(q, k, v, mode=mode, kv_layout=kv_layout)
        tmp, output, lse = make_single_jit_buffers(
            q, output_shape=shape, lse_shape=lse_shape, return_lse=return_lse,
        )
    else:
        import torch

        tmp = torch.empty(ATTENTION_SINGLE_JIT_TEMP_BYTES, dtype=torch.uint8, device=q.device)
        output = torch.empty(shape, dtype=q.dtype, device=q.device)
        lse = torch.empty(lse_shape, dtype=torch.float32, device=q.device) if return_lse else None
    tail = (upstream_kv_layout_code(layout), window_left, *args)
    call = (q, k, v, tmp, output, lse, *tail) if decode else (
        q, k, v, tmp, output, lse, mask_mode, *tail,
    )
    from contextlib import nullcontext

    recording = record_single_jit_protocol_call(
        mode=mode.value, jit_module=jit_module, q=q, k=k, v=v,
        tmp=tmp, output=output, lse=lse,
        layout_code=upstream_kv_layout_code(layout),
        mask_code=None if decode else mask_mode,
        window_left=window_left, return_lse=return_lse, extra_args=args,
    ) if isinstance(q, ReferenceTensor) else nullcontext()
    with recording:
        run(*call)
        if isinstance(q, ReferenceTensor):
            output = freeze_jit_buffer(output, "output")
            if lse is not None:
                lse = freeze_jit_buffer(lse, "lse")
    # The pinned upstream JIT decode entry returns only output even with an LSE buffer.
    return (output, lse) if return_lse and not decode else output
