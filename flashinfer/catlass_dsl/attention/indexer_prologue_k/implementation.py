"""Ascend950 上基于 CATLASS DSL JIT 的 indexer K 路径 prologue 实现。

kernel 与本声明同目录，位于 ``indexer_prologue_k.py``，由
``CatlassDslIndexerPrologueRunner`` 编译并发射。该算子原地更新 cache：
``latent[T, H]`` 与 ``wk[D, H]`` 相乘投影，随后做 RMSNorm、尾部 RoPE、
block-32 MXFP4 打包，再按 ``cache_index`` 散射进 paged ``k_cache``。
"""

from flashinfer.operators.author_api import CatlassDslImplementation, PythonKernel, Support
from flashinfer.operators.definition import OperatorDefinition

from .runner import CatlassDslIndexerPrologueRunner


OPERATOR = OperatorDefinition(
    name="indexer_prologue_k",
    domain="attention",
    op="attention.indexer_prologue_k",
    description=(
        "DSA indexer K 路径 prologue。storage_mode 0 把 cache_index 当作 "
        "k_cache[P,S,N,D/2] 的平铺 slot，MXFP4 scale 写入同形状的 "
        "k_scale_cache[P,S,N,D/32]；storage_mode 1 把 combined_block_size 行及其 "
        "scale 打包进同一行 k_cache。负索引跳过；同一 slot 被多行写入时只有最后 "
        "一行可见。D 需为 64 的倍数，(D - Dr) 也需为 64 的倍数。不可入图。"
    ),
)

IMPLEMENTATION = CatlassDslImplementation(
    operator=OPERATOR,
    kernel_id="catlass_dsl/attention/indexer_prologue_k/Ascend950",
    target_soc="Ascend950",
    support=Support(
        dtypes=("bfloat16", "float32", "uint8", "int64"),
        soc_models=("Ascend950",),
        physical_layouts=("ND",),
        specializations=(("tensor_contract", ("supported",)),),
        contiguous=True,
    ),
    source=PythonKernel(file="indexer_prologue_k.py", language="catlass_dsl"),
    runner_factory=CatlassDslIndexerPrologueRunner,
)
