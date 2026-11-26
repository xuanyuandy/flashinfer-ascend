"""Ascend950 AOT implementation of the C4A4 AICPU metadata operator."""

from flashinfer.ascendc.definition import AscendCImplementation, BuildConfig, NativeBinding
from flashinfer.jit.aclnn_contract import AttributeParameter
from flashinfer.jit.recipe import ShapeConstraint, TensorParameter
from flashinfer.operators.attention.quant_flash_attn_metadata.operator import OPERATOR
from flashinfer.operators.author_api import Support

KERNEL_ID = "ascendc/attention/quant_flash_attn_metadata/native/Ascend950"
SHARED_HEADER_DIR = "csrc/kernels/attention/common/op_kernel"

IMPLEMENTATION = AscendCImplementation(
    operator=OPERATOR,
    kernel_id=KERNEL_ID,
    support=Support(soc_models=("Ascend950",), dtypes=("int32", "none")),
    binding=NativeBinding(
        tensors=(
            TensorParameter("cu_seqlens_q", "input", "int32", ("BatchPlusOne",)),
            TensorParameter("cu_seqlens_kv", "input", "int32", ("BatchPlusOne",), optional=True),
            TensorParameter("seqused_q", "input", "int32", ("Batch",), optional=True),
            TensorParameter("seqused_kv", "input", "int32", ("Batch",)),
            TensorParameter("metadata", "output", "int32", (2, "Schedule")),
        ),
        attributes=(
            AttributeParameter("batch_size", "int64", 1, 1),
            AttributeParameter("max_seqlen_q", "int64", -1, -1),
            AttributeParameter("max_seqlen_kv", "int64", -1, -1),
            AttributeParameter("num_heads_q", "int64", 1, 1),
            AttributeParameter("num_heads_kv", "int64", 1, 1),
            AttributeParameter("head_dim", "int64", 128, 128, 256),
            AttributeParameter("head_dim_v", "int64", 128, 128, 256),
            AttributeParameter("quant_mode", "int64", 5, 5, 5),
            AttributeParameter("mask_mode", "int64", 0, 0, 3),
            AttributeParameter("win_left", "int64", -1, -1, -1),
            AttributeParameter("win_right", "int64", -1, -1, -1),
            AttributeParameter("layout_q", "string", "TND"),
            AttributeParameter("layout_q_descale", "string", "TND"),
            AttributeParameter("layout_kv", "string", "PA_BBND"),
            AttributeParameter("layout_out", "string", "TND"),
            AttributeParameter("is_grad_enabled", "bool", False),
        ),
        dimensions=tuple(ShapeConstraint(name, 1, 2**31 - 1)
                         for name in ("Batch", "BatchPlusOne", "Schedule")),
    ),
    build=BuildConfig(
        device="op_kernel_aicpu/quant_flash_attn_metadata_aicpu.cpp",
        host=("op_host/op_api/aclnn_quant_flash_attn_metadata.cpp",
              "op_host/op_api/l0_quant_flash_attn_metadata.cpp"),
        op_type="FiQuantFlashAttnMetadata", tiling=None, tiling_mode="none",
        device_entry="RunCpuKernel", task_type="aicpu", sync_policy="compiler_default",
        includes=("op_kernel_aicpu", "op_host"), cmake="CMakeLists.txt",
        shared_header_dirs=(f"{SHARED_HEADER_DIR}/load_balance",),
        shared_headers=(f"{SHARED_HEADER_DIR}/aicpu_common.h",),
    ),
    workspace_bound=256 * 1024 * 1024,
)
