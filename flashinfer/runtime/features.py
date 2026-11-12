"""Typed functional dimensions shared by capability matching and its catalogue.

Operator identity, tensor contracts, QuantSpec, ordered fusion/output contracts
and execution/artifact validation retain their existing owners. This catalogue
describes scalar functional choices, never M/N/K limits or backend priorities.
Adding a dimension extends data here, not the dispatcher or descriptor schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Tuple, Union


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    category: str
    value_type: str
    description: str
    choices: Tuple[Union[str, bool, int], ...] = ()

    def validate(self, value):
        if self.value_type == "bool":
            valid = type(value) is bool
        elif self.value_type == "positive_int":
            valid = type(value) is int and value > 0
        else:
            valid = isinstance(value, str) and bool(value.strip())
        if not valid or (self.choices and value not in self.choices):
            raise ValueError(f"feature {self.name}: invalid {value!r}; "
                             f"expected {self.value_type}, choices={self.choices!r}")


_DEFINITIONS = (
    FeatureDefinition("compute.dtype", "数值精度", "string", "实际计算精度"),
    FeatureDefinition("accumulator.dtype", "数值精度", "string", "实际累加精度"),
    FeatureDefinition("gemm.mode", "计算模式", "string", "矩阵乘计算形式",
                      ("dense", "batched", "grouped")),
    FeatureDefinition("gemm.transpose_a", "计算模式", "bool", "A 的逻辑转置语义"),
    FeatureDefinition("gemm.transpose_b", "计算模式", "bool", "B 的逻辑转置语义"),
    FeatureDefinition("attention.mode", "计算模式", "string", "Attention 调用阶段",
                      ("prefill", "decode", "chunked_prefill", "multi_token_decode")),
    FeatureDefinition("attention.variant", "计算模式", "string", "Attention 头组织或算法",
                      ("mha", "gqa", "mqa", "mla")),
    FeatureDefinition("attention.fp16_qk_reduction", "数值精度", "bool", "是否要求 FP16 QK reduction"),
    FeatureDefinition("kv_cache.kind", "数据组织", "string", "KV 的组织方式",
                      ("dense", "ragged", "paged")),
    FeatureDefinition("kv_cache.layout", "数据组织", "string", "KV 的逻辑轴顺序", ("NHD", "HND")),
    FeatureDefinition("mask.kind", "Mask 与位置", "string", "Mask 语义或表示形式",
                      ("none", "causal", "custom", "packed_custom")),
    FeatureDefinition("position.encoding", "Mask 与位置", "string", "位置编码语义",
                      ("NONE", "ROPE_LLAMA", "ALIBI")),
    FeatureDefinition("attention.causal", "Mask 与位置", "bool", "是否要求因果 Attention"),
    FeatureDefinition("attention.sliding_window", "Mask 与位置", "bool", "是否要求滑动窗口"),
    FeatureDefinition("sparse.format", "稀疏", "string", "稀疏计算或存储形式",
                      ("dense", "block_sparse", "structured_2_4", "csr", "bsr")),
    FeatureDefinition("moe.routing_method", "路由与并行", "string", "专家选择方法",
                      ("topk", "grouped_topk", "softmax_topk", "sigmoid_topk")),
    FeatureDefinition("moe.top_k", "路由与并行", "positive_int", "每 Token 选择的专家数"),
    FeatureDefinition("moe.expert_parallel_size", "路由与并行", "positive_int", "专家并行规模"),
    FeatureDefinition("sampling.method", "计算模式", "string", "采样或过滤方法",
                      ("categorical", "top_k", "top_p", "min_p", "top_k_top_p")),
    FeatureDefinition("sampling.filter_order", "计算模式", "string", "组合过滤的语义顺序",
                      ("top_k_first", "top_p_first", "joint")),
    FeatureDefinition("normalization.kind", "计算模式", "string", "归一化计算形式", ("rms", "layer")),
    FeatureDefinition("normalization.affine", "计算模式", "bool", "是否要求仿射变换"),
    FeatureDefinition("activation.kind", "计算模式", "string", "激活函数语义",
                      ("silu", "gelu", "gelu_tanh", "relu", "sigmoid")),
    FeatureDefinition("reduction.kind", "计算模式", "string", "归约的数学语义",
                      ("sum", "max", "min", "mean", "logsumexp")),
    FeatureDefinition("reduction.keepdim", "计算模式", "bool", "归约是否保留维度"),
)
_BY_NAME = MappingProxyType({item.name: item for item in _DEFINITIONS})


def functional_feature_catalog():
    """Return immutable definitions; enumeration is not kernel support evidence."""
    return _DEFINITIONS


def validate_feature(name, value):
    try:
        definition = _BY_NAME[name]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unknown functional feature {name!r}") from error
    definition.validate(value)


# Compatibility views for the previously introduced flat semantic fields.
# These aliases share one matcher and cannot contradict explicit features.
INFERENCE_FEATURE_ALIASES = (
    ("compute_dtype", "compute.dtype", "compute_dtypes"),
    ("accumulator_dtype", "accumulator.dtype", "accumulator_dtypes"),
    ("attention_mode", "attention.mode", "attention_modes"),
    ("attention_variant", "attention.variant", "attention_variants"),
    ("kv_cache_type", "kv_cache.kind", "kv_cache_types"),
    ("mask_type", "mask.kind", "mask_types"),
    ("pos_encoding", "position.encoding", "pos_encodings"),
    ("causal", "attention.causal", "causal_values"),
    ("sliding_window", "attention.sliding_window", "sliding_window_values"),
    ("sparse_format", "sparse.format", "sparse_formats"),
    ("routing_method", "moe.routing_method", "routing_methods"),
    ("moe_top_k", "moe.top_k", "moe_top_k_values"),
    ("expert_parallel_size", "moe.expert_parallel_size", "expert_parallel_sizes"),
)
