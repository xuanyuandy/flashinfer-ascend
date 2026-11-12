"""CANN FIA Attention and explicit tensor reference for numerical comparison."""

from functools import partial

from flashinfer.runtime.schema import Backend, CapabilitySpec, GraphCapability, GraphReplaySpec, KernelDescriptor, SchemaError

_DECODE_ID = "cann/attention/single_decode_fia_v1"

_DECODE_REPLAY = GraphReplaySpec(
    validated_kernel_id=_DECODE_ID,
    validation_tests=("tests/test_replay_param_npu.py::test_fia_full_capture_replay_needs_parameter_update",),
    validated_environment="Ascend950PR_9579; torch 2.10.0+cpu; torch_npu 2.10.0.post7; CANN 9.2",
    parameter_sources=(("actual_seq_lengths_kv", "kv_seq_lengths"),),
    required_arguments=("workspace", "out"),
    fixed_arguments=(("input_layout", "BSND"), ("atten_mask", None),
                     ("num_heads", 8), ("num_key_value_heads", 2),
                     ("scale", .125), ("softmax_lse_flag", True)),
    tensor_metadata=tuple((name, shape, ("float16", "bfloat16")) for name, shape in (
        ("query", (1, 1, 8, 64)), ("key", (1, 1024, 2, 64)), ("value", (1, 1024, 2, 64)),
    )),
    parameter_ranges=(("actual_seq_lengths_kv", 1, 1024),),
)

def _scale(value, heads, tensor, name):
    import torch

    if value is None:
        return 1.0
    if isinstance(value, torch.Tensor):
        if value.device != tensor.device or value.shape != (heads,):
            raise SchemaError(f"{name} must be a per-head tensor on the input device")
        if not bool(torch.isfinite(value).all()):
            raise SchemaError(f"{name} must be finite")
        return value.float().reshape(1, heads, 1)
    import math

    result = float(value)
    if not math.isfinite(result):
        raise SchemaError(f"{name} must be finite")
    return result

def dense_attention(q, k, v, *, native=True, decode=False, kv_layout="NHD",
                    causal=False, pos_encoding_mode="NONE", scale_q=None,
                    scale_k=None, scale_v=None, k_scale=None, v_scale=None,
                    o_dtype=None, custom_mask=None, packed_custom_mask=None,
                    sm_scale=None, window_left=-1, logits_soft_cap=None,
                    rope_scale=None, rope_theta=None,
                    use_fp16_qk_reduction=False, return_lse=False,
                    workspace=None):
    import math
    import torch

    if kv_layout not in ("NHD", "HND"):
        raise SchemaError("kv_layout must be NHD or HND")
    # Upstream selects CUSTOM before CAUSAL when either mask is supplied.
    if custom_mask is not None or packed_custom_mask is not None:
        causal = False
    if q.ndim != (2 if decode else 3) or k.ndim != 3 or v.ndim != 3:
        raise SchemaError("single Attention expects unbatched q/k/v shapes")
    if k.device != q.device or v.device != q.device:
        raise SchemaError("q, k and v must be on the same device")
    if k.dtype != v.dtype:
        raise SchemaError("k/v dtypes must match")
    if kv_layout == "HND":
        k, v = k.transpose(0, 1), v.transpose(0, 1)
    if decode:
        q = q.unsqueeze(0)
    sq, h, d = q.shape
    sk, hk, dk = k.shape
    if v.shape[:2] != k.shape[:2] or dk != d or hk == 0 or h % hk:
        raise SchemaError("incompatible q/k/v lengths, heads or dimensions")
    if decode and v.shape[-1] != d:
        raise SchemaError("single decode requires equal q/k and v dimensions")
    if window_left < -1:
        raise SchemaError("window_left must be -1 or nonnegative")
    if pos_encoding_mode != "NONE":
        raise NotImplementedError("CANN dense Attention currently supports pos_encoding_mode='NONE'")
    if use_fp16_qk_reduction:
        raise NotImplementedError("FP16 QK reduction is not implemented by this FIA adapter")
    if native and logits_soft_cap not in (None, 0, 0.0):
        raise NotImplementedError("FIA does not expose the upstream logits_soft_cap transform")
    if native and q.device.type != "npu":
        raise SchemaError("CANN Attention requires NPU tensors")
    scale = 1 / math.sqrt(d) if sm_scale is None else float(sm_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise SchemaError("sm_scale must be positive and finite")
    out_dtype = q.dtype if o_dtype is None else (
        getattr(torch, o_dtype.removeprefix("torch.")) if isinstance(o_dtype, str) else o_dtype
    )
    qs = _scale(scale_q, h, q, "q_scale")
    ks = _scale(scale_k, hk, k, "k_scale") * _scale(k_scale, hk, k, "k_scale")
    vs = _scale(scale_v, hk, v, "v_scale") * _scale(v_scale, hk, v, "v_scale")
    q1, k1, v1 = q.float() * qs, k.float() * ks, v.float() * vs
    if sq == 0 or sk == 0:
        output = torch.zeros((sq, h, v.shape[-1]), dtype=out_dtype, device=q.device)
        lse = torch.full((sq, h), -math.inf, dtype=torch.float32, device=q.device)
    else:
        allowed = None
        if causal or window_left >= 0 or custom_mask is not None or packed_custom_mask is not None:
            rows = torch.arange(sq, device=q.device).unsqueeze(1) + sk - sq
            cols = torch.arange(sk, device=q.device).unsqueeze(0)
            allowed = torch.ones((sq, sk), dtype=torch.bool, device=q.device)
            if causal:
                allowed &= cols <= rows
            if window_left >= 0:
                allowed &= cols >= rows - window_left
            if packed_custom_mask is not None:
                if packed_custom_mask.device != q.device or packed_custom_mask.dtype != torch.uint8 or packed_custom_mask.numel() != (sq * sk + 7) // 8:
                    raise SchemaError("packed_custom_mask shape/dtype/device mismatch")
                bits = torch.arange(sq * sk, device=q.device)
                unpacked = ((packed_custom_mask.flatten()[bits // 8].int() >> (bits % 8)) & 1).bool()
                allowed &= unpacked.reshape(sq, sk)
            elif custom_mask is not None:
                if custom_mask.device != q.device or custom_mask.dtype != torch.bool or custom_mask.shape != (sq, sk):
                    raise SchemaError("custom_mask shape/dtype/device mismatch")
                allowed &= custom_mask
        if native:
            import torch_npu
            from flashinfer.runtime._graph_params import _is_recording, _record_kernel_call

            if _is_recording() and workspace is None:
                raise SchemaError("updatable FIA capture requires caller-owned fixed workspace")
            if _is_recording() and allowed is not None:
                # These masks encode capture-time KV positions. Updating only
                # FIA's host length would leave that preprocessing stale.
                raise NotImplementedError("FIA parameter recording requires unmasked attention (T11A)")

            # A5/CANN 9.2 BNSD masked GQA writes incomplete LSE rows.
            # BSND is verified by probe_attention_lse.py and keeps KV shared.
            query = q1.to(q.dtype).unsqueeze(0).contiguous()
            key = k1.to(k.dtype).unsqueeze(0).contiguous()
            value = v1.to(v.dtype).unsqueeze(0).contiguous()
            options = dict(
                atten_mask=None if allowed is None else (~allowed)[None, None].contiguous(),
                num_heads=h, num_key_value_heads=hk, input_layout="BSND",
                scale=scale, softmax_lse_flag=bool(return_lse),
            )
            if workspace is None:
                result, natural_lse = torch_npu.npu_fused_infer_attention_score(query, key, value, **options)
            else:
                outputs = torch_npu._npu_fused_infer_attention_score_infer_output(
                    query, value, input_layout="BSND", num_heads=h,
                    num_key_value_heads=hk, softmax_lse_flag=bool(return_lse),
                )
                from flashinfer.runtime.registry import get_global_registry

                kernel_id = _DECODE_ID if decode else "cann/attention/single_prefill_fia_v1"
                result, natural_lse = _record_kernel_call(
                    get_global_registry().get(kernel_id),
                    torch_npu.npu_fused_infer_attention_score.out,
                    dict(query=query, key=key, value=value, workspace=workspace,
                         out=list(outputs), actual_seq_lengths_kv=[sk], **options),
                )
            output = result.squeeze(0).to(out_dtype)
            lse = natural_lse.reshape(h, sq).transpose(0, 1) / math.log(2) if return_lse else None
        else:
            k2 = k1.repeat_interleave(h // hk, dim=1)
            v2 = v1.repeat_interleave(h // hk, dim=1)
            scores = torch.einsum("qhd,khd->hqk", q1, k2) * scale
            if logits_soft_cap not in (None, 0, 0.0):
                scores = float(logits_soft_cap) * torch.tanh(scores / float(logits_soft_cap))
            if allowed is not None:
                scores = scores.masked_fill(~allowed.unsqueeze(0), -math.inf)
            lse = torch.logsumexp(scores, -1).transpose(0, 1) / math.log(2)
            probs = torch.nan_to_num(scores.softmax(-1), nan=0.0)
            output = torch.einsum("hqk,khd->qhd", probs, v2).to(out_dtype)
        if allowed is not None:
            valid = allowed.any(-1)
            output = torch.where(valid[:, None, None], output, 0)
            if return_lse:
                lse = lse.masked_fill(~valid[:, None], -math.inf)
    if decode:
        output = output.squeeze(0)
        if return_lse:
            lse = lse.squeeze(0)
    return (output, lse.contiguous()) if return_lse else output

def maximum_workspace_bytes(spec, query_lengths, kv_lengths, *, device):
    """Query the real v1 BSND composite, outside capture, including both LSE modes.

    torch_npu returns an allocated tensor in query.dtype, NOT a byte count.
    Only its byte capacity is retained: the caller-owned workspace already
    bound by the public wrapper supplies the persistent storage. Prototypes
    use real contiguous shapes; the query performs CANN tiling/stride checks.
    """
    import torch
    import torch_npu

    required = 0
    # Requests execute sequentially, so their workspace is shared (max, not sum).
    shapes = {(max(1, q), max(1, k)) for q, k in zip(query_lengths, kv_lengths)}
    for sq, sk in sorted(shapes):
        query = torch.empty((1, sq, spec.num_qo_heads, spec.head_dim_qk),
                            dtype=getattr(torch, spec.q_dtype), device=device)
        key = torch.empty((1, sk, spec.num_kv_heads, spec.head_dim_qk),
                          dtype=getattr(torch, spec.kv_dtype), device=device)
        value = torch.empty((1, sk, spec.num_kv_heads, spec.head_dim_vo),
                            dtype=key.dtype, device=device)
        masked = spec.effective_causal or spec.window_left >= 0 or spec.custom_mask is not None
        mask = torch.empty((1, 1, sq, sk), dtype=torch.bool, device=device) if masked else None
        for return_lse in (False, True):
            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                query, key, value, atten_mask=mask,
                num_heads=spec.num_qo_heads, num_key_value_heads=spec.num_kv_heads,
                input_layout="BSND", scale=spec.sm_scale, softmax_lse_flag=return_lse,
            )
            required = max(required, workspace.numel() * workspace.element_size())
            del workspace
    return required
