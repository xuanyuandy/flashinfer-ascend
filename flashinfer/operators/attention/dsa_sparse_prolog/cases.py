"""CPU reference and the enabled DsaSparseProlog cases from ops-transformer pytest."""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


class DsaSparsePrologConfig:
    """DSA sparse prolog dimensions. Defaults are the source real shape."""

    def __init__(
        self,
        hidden_size: int = 8192,
        num_attention_heads: int = 64,
        q_lora_rank: int = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        mla_scale_q_lora: bool = True,
        mla_scale_kv_lora: bool = True,
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.mla_scale_q_lora = mla_scale_q_lora
        self.mla_scale_kv_lora = mla_scale_kv_lora

    @property
    def q_scale(self) -> float:
        if self.mla_scale_q_lora:
            return (self.hidden_size / self.q_lora_rank) ** 0.5
        return 1.0

    @property
    def kv_scale(self) -> float:
        if self.mla_scale_kv_lora:
            return (self.hidden_size / self.kv_lora_rank) ** 0.5
        return 1.0


def build_rope_cos_sin(seq_len, dim, base=10000.0, dtype=torch.float32, device=torch.device("cpu")):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    freqs = torch.einsum("i,j->ij", torch.arange(seq_len, device=device, dtype=inv_freq.dtype), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_mla(t, cos, sin):
    *batch, h, d = t.shape
    t = t.view(*batch, h, d // 2, 2).transpose(-1, -2).reshape(*batch, h, d)
    cos = cos.reshape(*cos.shape[:-1], 1, d)
    sin = sin.reshape(*sin.shape[:-1], 1, d)
    return (t * cos) + (_rotate_half(t) * sin)


def rand_weights_fp32(cfg, seed=0):
    generator = torch.Generator().manual_seed(seed)
    he, hcq, hckv = cfg.hidden_size, cfg.q_lora_rank, cfg.kv_lora_rank
    d, dr, n = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.num_attention_heads
    scale = 0.02
    return {
        "weight_dq": torch.randn(he, hcq, generator=generator) * scale,
        "weight_uq_qr": torch.randn(hcq, n * (d + dr), generator=generator) * scale,
        "weight_dkv_kr": torch.randn(he, hckv + dr, generator=generator) * scale,
        "gamma_cq": torch.randn(hcq, generator=generator),
        "gamma_ckv": torch.randn(hckv, generator=generator),
    }


def dsa_sparse_prolog_golden(cfg, token_x, weights, rope_cos, rope_sin):
    batch_shape = token_x.shape[:-1]
    hckv = cfg.kv_lora_rank
    d, dr, n = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.num_attention_heads
    q_m = token_x @ weights["weight_dq"]
    q_rstd = torch.rsqrt(q_m.float().square().mean(-1, keepdim=True) + cfg.rms_norm_eps)
    qr = F.rms_norm(q_m, (cfg.q_lora_rank,), weights["gamma_cq"], cfg.rms_norm_eps)
    q_states = (qr @ weights["weight_uq_qr"]).view(*batch_shape, n, d + dr)
    q_nope, q_pe = torch.split(q_states, [d, dr], dim=-1)
    q_nope = q_nope * cfg.q_scale
    q_pe = q_pe * cfg.q_scale
    compressed_kv = token_x @ weights["weight_dkv_kr"]
    ckv_m, k_pe = torch.split(compressed_kv, [hckv, dr], dim=-1)
    ckv_rstd = torch.rsqrt(ckv_m.float().square().mean(-1, keepdim=True) + cfg.rms_norm_eps)
    ct_kv = F.rms_norm(ckv_m, (hckv,), weights["gamma_ckv"], cfg.rms_norm_eps)
    ct_kv = ct_kv * cfg.kv_scale
    k_pe = k_pe.view(*batch_shape, 1, dr)
    q_pe = apply_rope_mla(q_pe, rope_cos, rope_sin)
    k_pe = apply_rope_mla(k_pe, rope_cos, rope_sin)
    return q_nope, q_pe, ct_kv, k_pe, qr, q_m, q_rstd, ckv_m, ckv_rstd


def dsa_sparse_prolog_golden_fp64(cfg, token_x, weights, rope_cos, rope_sin):
    weights64 = {key: value.detach().double().cpu() for key, value in weights.items()}
    return dsa_sparse_prolog_golden(
        cfg, token_x.detach().double().cpu(), weights64,
        rope_cos.detach().double().cpu(), rope_sin.detach().double().cpu(),
    )


def rel_err(left, right):
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    return (torch.linalg.vector_norm(left - right) / torch.linalg.vector_norm(right).clamp_min(1e-30)).item()


def check_precision_row(row, demote):
    """Source criterion: op vs fp64 is no worse than 1.5x the torch baseline."""
    if row["op_vs_fp64"] > max(row["torch_vs_fp64"] * 1.5, 1e-12):
        return False
    if demote and row["op_vs_torch"] > 5e-3 and row["op_vs_fp64"] > row["torch_vs_fp64"]:
        return False
    return True


class _ScaleOverride(DsaSparsePrologConfig):
    def __init__(self, base, q_scale, kv_scale):
        self.__dict__.update(base.__dict__)
        self._q_scale, self._kv_scale = q_scale, kv_scale

    @property
    def q_scale(self):
        return self._q_scale

    @property
    def kv_scale(self):
        return self._kv_scale


OUT_NAMES = ("q_nope", "q_pe", "ct_kv", "k_pe", "qr", "q_m", "q_rstd", "ckv_m", "ckv_rstd")

# The source pytest enables these six ND cases. NZ stays in the kernel and is
# not part of the enabled runner matrix.
CASE_PARAMETERS = (
    dict(batch_shape=(2, 64), dtype="bfloat16", demote=True, fusion=False, scale_q=0.0, scale_kv=0.0, weight_layout="nd_kn", case_name="tnd_B2S64_bf16_demote"),
    dict(batch_shape=(512,), dtype="bfloat16", demote=True, fusion=False, scale_q=0.0, scale_kv=0.0, weight_layout="nd_kn", case_name="tnd_T512_bf16_demote"),
    dict(batch_shape=(1, 128), dtype="float16", demote=True, fusion=False, scale_q=0.0, scale_kv=0.0, weight_layout="nd_kn", case_name="tnd_B1S128_fp16_demote"),
    dict(batch_shape=(2, 64), dtype="bfloat16", demote=True, fusion=False, scale_q=2.0, scale_kv=3.0, weight_layout="nd_kn", case_name="tnd_B2S64_bf16_scale"),
    dict(batch_shape=(2, 64), dtype="bfloat16", demote=True, fusion=False, scale_q=0.0, scale_kv=0.0, weight_layout="nd_kn", case_name="tnd_B2S64_bf16_nd_kn"),
    dict(batch_shape=(2, 64), dtype="bfloat16", demote=True, fusion=False, scale_q=0.0, scale_kv=0.0, weight_layout="nd_nk", case_name="tnd_B2S64_bf16_nd_nk"),
)


def _inputs(configuration):
    cfg = DsaSparsePrologConfig()
    batch_shape = configuration["batch_shape"]
    generator = torch.Generator().manual_seed(1)
    token_x = torch.randn(*batch_shape, cfg.hidden_size, generator=generator)
    weights = rand_weights_fp32(cfg, seed=0)
    length = batch_shape[-1] if len(batch_shape) > 1 else batch_shape[0]
    cos, sin = build_rope_cos_sin(length, cfg.qk_rope_head_dim, cfg.rope_theta)
    if len(batch_shape) > 1:
        cos = cos.unsqueeze(0).expand(batch_shape[0], -1, -1).contiguous()
        sin = sin.unsqueeze(0).expand(batch_shape[0], -1, -1).contiguous()
    golden_cfg = cfg
    if configuration["scale_q"] > 0 and configuration["scale_kv"] > 0:
        golden_cfg = _ScaleOverride(cfg, configuration["scale_q"], configuration["scale_kv"])
    return cfg, golden_cfg, token_x, weights, cos, sin


def sample(torch_module, torch_npu, **configuration):
    """Build device arguments. nd_nk is transposed before launch, as in the source runner."""
    del torch_module
    cfg, golden_cfg, token_x, weights, cos, sin = _inputs(configuration)
    dtype = getattr(torch, configuration["dtype"])
    baseline = dsa_sparse_prolog_golden(
        golden_cfg, token_x.to(dtype), {key: value.to(dtype) for key, value in weights.items()},
        cos.to(dtype), sin.to(dtype),
    )
    fp64 = dsa_sparse_prolog_golden_fp64(cfg, token_x, weights, cos, sin)

    def device(tensor):
        return torch_npu.npu_format_cast(tensor.to(dtype).npu().contiguous(), 2)

    names = ("weight_dq", "weight_uq_qr", "weight_dkv_kr")
    if configuration["weight_layout"] == "nd_kn":
        launched = {key: device(weights[key]) for key in names}
    elif configuration["weight_layout"] == "nd_nk":
        launched = {key: device(weights[key].t()) for key in names}
    else:
        raise ValueError("enabled cases are ND only, got " + configuration["weight_layout"])
    args = (
        device(token_x), launched["weight_dq"], launched["weight_uq_qr"], launched["weight_dkv_kr"],
        device(weights["gamma_cq"]), device(weights["gamma_ckv"]), device(sin), device(cos),
    )
    options = dict(
        rmsnorm_epsilon_cq=cfg.rms_norm_eps, rmsnorm_epsilon_ckv=cfg.rms_norm_eps,
        mla_scale_q_lora=float(configuration["scale_q"]), mla_scale_kv_lora=float(configuration["scale_kv"]),
        qk_nope_head_dim=cfg.qk_nope_head_dim, is_type_demotion=bool(configuration["demote"]),
        apply_rope_fusion=bool(configuration["fusion"]), backend="ascendc",
    )
    expected = dict(baseline=baseline, fp64=fp64, demote=bool(configuration["demote"]), case_name=configuration["case_name"])
    return args, options, expected


def compare(actual, expected):
    rows = []
    for name, op_out, base_out, fp64_out in zip(OUT_NAMES, actual, expected["baseline"], expected["fp64"]):
        rows.append(dict(
            out=name,
            op_vs_torch=rel_err(op_out.cpu(), base_out.cpu()),
            op_vs_fp64=rel_err(op_out.cpu(), fp64_out),
            torch_vs_fp64=rel_err(base_out.cpu(), fp64_out),
        ))
    return all(check_precision_row(row, expected["demote"]) for row in rows), rows
