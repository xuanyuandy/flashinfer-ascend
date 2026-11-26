"""Ascend950 native C4A4 declaration and ordered ACLNN signature."""

from flashinfer.ascendc.definition import AscendCImplementation, BuildConfig, NativeBinding
from flashinfer.jit.aclnn_contract import AttributeParameter, TensorDtypeConversion
from flashinfer.jit.recipe import ShapeConstraint, TensorParameter
from flashinfer.operators.attention.quant_flash_attn.operator import E8M0, FP4, OPERATOR
from flashinfer.operators.author_api import Support
from flashinfer.runtime import GraphCapability
from flashinfer.runtime.schema import GraphReplaySpec
from .runner import admission, make_runner

KERNEL_ID = "ascendc/attention/quant_flash_attn/native/Ascend950"
SHARED_HEADER_DIR = "csrc/kernels/attention/common/op_kernel"
FP4_CONVERSION = TensorDtypeConversion(FP4, "float4_e2m1", 2)
E8M0_CONVERSION = TensorDtypeConversion(E8M0, "float8_e8m0")
STORAGE_TYPES = (FP4, E8M0, "int32", "int8", "bfloat16", "float32", "none")


def _input(name, dtype, shape, *, optional=False, alternative=(), conversion=None, allow_strided=False):
    return TensorParameter(name, "input", dtype, shape, optional=optional,
                           shape_alternatives=(alternative,) if alternative else (),
                           native_conversions=(conversion,) if conversion else (), allow_strided=allow_strided)


def _scale(name, shape, alternative, *, allow_strided=False):
    return _input(name, E8M0, shape, alternative=alternative, conversion=E8M0_CONVERSION,
                  allow_strided=allow_strided)


payload = ("Pages", "KvAxis1", "KvAxis2", "PackedD")
nz_payload = ("Pages", "KvHeads", "FractalD", "Page", 32)
TENSORS = (
    _input("q", FP4, ("Tokens", "Heads", "PackedD"), conversion=FP4_CONVERSION),
    _input("k", FP4, payload, alternative=nz_payload, conversion=FP4_CONVERSION, allow_strided=True),
    _input("v", FP4, payload, alternative=nz_payload, conversion=FP4_CONVERSION, allow_strided=True),
    _scale("q_descale", ("Tokens", "Heads", "ScaleD", 2), ("KvHeads", "Tokens", "Group", "ScaleD", 2)),
    _scale("k_descale", ("Pages", "KvAxis1", "KvAxis2", "ScaleD", 2),
           ("Pages", "KvHeads", "PageTiles", "ScaleD", 16, 2), allow_strided=True),
    _scale("v_descale", ("Pages", "VScaleAxis1", "VScaleAxis2", "D", 2),
           ("Pages", "KvHeads", "ChannelTiles", "PageScale", 16, 2), allow_strided=True),
    _input("block_table", "int32", ("Batch", "PageSlots")),
    _input("p_scale", "float32", (1,), optional=True),
    _input("cu_seqlens_q", "int32", ("BatchPlusOne",)),
    _input("cu_seqlens_kv", "int32", ("BatchPlusOne",), optional=True),
    _input("seqused_q", "int32", ("Batch",), optional=True),
    _input("seqused_kv", "int32", ("Batch",)),
    _input("sinks", "float32", ("Heads",), optional=True),
    _input("attn_mask", "int8", (2048, 2048), optional=True),
    _input("metadata", "int32", ("Schedule",), alternative=(2, "Schedule")),
    _input("v_tail", "bfloat16", ("Pages", "Page", "KvHeads", "D"), optional=True),
    _input("block_table_tail", "int32", ("Batch", "PageSlots"), optional=True),
    _input("seqused_v_tail", "int32", ("Batch",), optional=True),
    TensorParameter("attn_out", "output", "bfloat16", ("Tokens", "Heads", "D")),
    TensorParameter("softmax_lse", "output", "float32", (1,), shape_alternatives=(("Heads", "Tokens"),)),
)

ATTRIBUTES = (
    AttributeParameter("quant_compute_mode", "int64", 5, 5, 5),
    AttributeParameter("softmax_scale", "float64", 1., 0., 3.4028234663852886e38),
    AttributeParameter("mask_mode", "int64", 0, 0, 3),
    AttributeParameter("win_left", "int64", -1, -1, -1),
    AttributeParameter("win_right", "int64", -1, -1, -1),
    AttributeParameter("max_seqlen_q", "int64", -1, -1, 2**31 - 1),
    AttributeParameter("max_seqlen_kv", "int64", -1, -1, 2**31 - 1),
    AttributeParameter("layout_q", "string", "TND"),
    AttributeParameter("layout_q_descale", "string", "TND"),
    AttributeParameter("layout_kv", "string", "PA_BBND"),
    AttributeParameter("layout_out", "string", "TND"),
    AttributeParameter("return_softmax_lse", "bool", False),
    AttributeParameter("quant_schema", "int64", 2, 0, 2),
)
dimensions = sorted({axis for tensor in TENSORS for shape in (tensor.shape,) + tensor.shape_alternatives
                     for axis in shape if isinstance(axis, str)})

IMPLEMENTATION = AscendCImplementation(
    operator=OPERATOR, kernel_id=KERNEL_ID,
    support=Support(soc_models=("Ascend950",), dtypes=STORAGE_TYPES),
    binding=NativeBinding(tensors=TENSORS, attributes=ATTRIBUTES,
                          dimensions=tuple(ShapeConstraint(name, 1, 2**31 - 1) for name in dimensions)),
    build=BuildConfig(
        device="op_kernel/quant_flash_attn.cpp", host=("op_host/quant_flash_attn_def.cpp",),
        op_type="FiQuantFlashAttn", tiling="FiQuantFlashAttnTiling",
        device_entry="fi_quant_flash_attn", cmake="CMakeLists.txt",
        shared_header_dirs=(f"{SHARED_HEADER_DIR}/arch35", f"{SHARED_HEADER_DIR}/memcopy"),
        shared_headers=tuple(f"{SHARED_HEADER_DIR}/{name}" for name in (
            "arch_info.h", "attn_buffer.h", "attn_buffer_manager.h", "buffers_policy.h",
            "const_def.h", "util.h", "vector_common.h",
        )),
    ),
    workspace_bound=256 * 1024 * 1024,
    runner_factory=make_runner, admission=admission,
    graph_capability=GraphCapability.CAPTURE_SAFE,
    graph_replay=GraphReplaySpec(
        validated_kernel_id=KERNEL_ID,
        validation_tests=(
            "tests/operators/test_quant_flash_attn_graph.py::test_c4a4_public_fixed_output_replay",
        ),
        validated_environment="Ascend950DT_9572, CANN 9.2.1, torch_npu 2.10.0.post7",
        strategy="none",
        required_arguments=("out", "metadata"),
        fixed_arguments=(("quant_schema", 2),),
    ),
)
