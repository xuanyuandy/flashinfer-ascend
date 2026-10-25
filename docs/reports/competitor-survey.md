# 上游 FlashInfer（CUDA）架构与能力调研

> 调研基线：flashinfer-ori @ da992ed9919604c5eb6dc255c63f5e89a466452c，调研日期 2026-10-24
>
> 范围：用户指定 CUDA 仓库的本地静态调研；未运行 GPU kernel、未下载 artifact、未构建或安装 wheel。
> 文中以 `flashinfer-ori:` 标记的相对路径均指该基线中的真实文件或目录；运行期生成布局另行标注为模式示意。
> “存在/支持”表示有对应实现或明确入口，不自动代表所有架构、dtype、shape、后端组合均可用，也不等于稳定发布承诺。

## 1. 结论速览

1. **能力面已超出 attention 库。** 标准 attention、MLA、稀疏 attention、GEMM、MoE/EP、通信、采样、归一化、RoPE、量化、KV 管理之外，
   还有 GDN/KDA/Mamba、模型专用融合、训练辅助、trace、autotuner 和多种实验后端。
   直接目录与入口逐项见 §2；公开面和实验面需分别对标，不能只比较顶层导出函数数量。
2. **可组合性主要来自分层接口。** wrapper 负责参数、layout、workspace 与 plan 状态；generator 负责特化；
   JIT/AOT 负责准备 module；TVM-FFI typed export 使 native 函数成为 Python 属性。
   示例链路：`flashinfer-ori: flashinfer/decode.py` → `flashinfer-ori: flashinfer/jit/attention/modules.py`
   → `flashinfer-ori: csrc/batch_decode_jit_binding.cu`，详见 §3.1、§3.6。
3. **编译与分发是多路线体系。** `JitSpec` 已抽象共同生命周期，NVCC 和 CuTe-DSL 分别实现 `.so`/对象缓存；
   Cake、NVRTC、Triton 及部分独立运行时还有专门路径，不能宣称全部由一个 JIT key 或注册表治理。
   证据：`flashinfer-ori: flashinfer/jit/core.py`、`flashinfer-ori: flashinfer/jit/cute_dsl_core.py`、`flashinfer-ori: flashinfer/jit/cake_program.py`。
4. **分发至少有四个职责角色。** 主 Python 包、JIT-cache shim、架构 provider、cubin 包分别承担 API/源码、发现、host module、设备 artifact；
   provider 的 module manifest、版本与架构兼容性共同决定 AOT 命中，缺失再回落 JIT。
   证据：`flashinfer-ori: flashinfer/jit/env.py`、`flashinfer-ori: flashinfer-jit-cache`、`flashinfer-ori: flashinfer-jit-cache-provider`、`flashinfer-ori: flashinfer-cubin`。
5. **图支持依赖生命周期与地址契约。** plan/预热在捕获外准备，run 消费已持有的 workspace 和元数据；
   支持范围随 wrapper/backend 改变，不能由一个 `use_cuda_graph` 参数推导所有路径均可动态重放。
   证据与 wrapper 清单见 §3.4，重点是输入更新、容量、shape、重规划限制和预热要求。
6. **PyTorch 注册存在必须保留的例外。** 本基线通用 `register_custom_op/register_fake_op` helper 是 identity decorator；
   模块里写了 decorator 与 fake 函数不等于已经进入 dispatcher。直接 `torch.library` 路线需逐项确认。
   证据：`flashinfer-ori: flashinfer/utils.py`，`register_custom_op/register_fake_op`；边界见 §3.3。
7. **后端策略与性能调优是两层。** `auto` 通常先做架构/特征/shape 合法性与 heuristic；autotuner 再在可用候选/tactic 上计时及缓存。
   不存在适用于所有算子的全局 backend 排序。证据：`flashinfer-ori: flashinfer/utils.py`、`flashinfer-ori: flashinfer/autotuner`，详见 §3.5。
8. **下游与可扩展性要按证据层级验收。** 本仓可证明 API 使用示例、测试以及部分集成设计，不能证明下游所有发布版本的真实调用点。
   §4 提供可执行扩展序列；§5 用稳定机制 ID 作为 R2 的差分主键；未确认项统一保留在 §6。

## 2. 仓库结构与能力域全景

### 2.1 口径与边界

本节将“支持”限定为此 commit 中存在入口、实现或明确的 backend 路由；不是对官方发行版、所有 GPU 或所有输入组合的支持承诺。
公开入口以 `__init__.py` 的重导出、模块内非私有函数/类和函数装饰器为证据；内核内部符号与面向用户的 API 分开记录。
`flashinfer-ori: flashinfer/experimental/__init__.py` 明确：experimental backend 不承诺兼容性；实验 API 可位于 core，不能只靠目录判断稳定性。
`flashinfer-ori: flashinfer/api_logging.py` 的 `flashinfer_experimental_api` 标记实验 API；`experimental_auto_backends_allowed` 控制自动选择。
显式调用实验 API / 显式选择实验 backend 与 `backend="auto"` 的准入不同，详见 §3.5。
例如 `flashinfer-ori: flashinfer/__init__.py` 无条件导出 `gdn_fused_decode_step`，实际实现位于 `flashinfer-ori: flashinfer/gdn_kernels/experimental`；这说明路径名不能代替入口契约。
下表列代表性公开符号，按能力域归纳；后面的目录覆盖表负责完整点名，不能将每个实现目录都误读为独立产品 API。

### 2.2 计算与数据能力域

| ID | 能力域与对外符号 | 主要文件证据 | 底层来源与边界 |
|---|---|---|---|
| CAP-ATT-01 | 单请求 decode；`single_decode_with_kv_cache`；GQA/MQA、位置编码、window、soft cap 等特化参数 | `flashinfer-ori: flashinfer/decode.py`；`flashinfer-ori: csrc/single_decode.cu`；`flashinfer-ori: include/flashinfer/attention/decode.cuh` | FlashInfer CUDA 模板；Python 校验/选择后调用生成模块；可传自定义 JIT module 的入口另列在同文件。 |
| CAP-ATT-02 | 单请求 prefill；`single_prefill_with_kv_cache` | `flashinfer-ori: flashinfer/prefill.py`；`flashinfer-ori: csrc/single_prefill.cu`；`flashinfer-ori: csrc/single_prefill_sm90.cu` | FA2 CUDA 与 FA3/Hopper 路径；特化 dtype/head dimension/mask 等，并非任意参数均落同一 kernel。 |
| CAP-ATT-03 | 批量 paged/ragged prefill；`BatchPrefillWithPagedKVCacheWrapper`、`BatchPrefillWithRaggedKVCacheWrapper` | `flashinfer-ori: flashinfer/prefill.py`；`flashinfer-ori: csrc/batch_prefill.cu`；`flashinfer-ori: csrc/batch_prefill_sm90.cu` | CUDA/CUTLASS、TRT-LLM、cuDNN、CuTe-DSL、prims_ts 等由入口分支接入；plan/run 是主要复用边界。 |
| CAP-ATT-04 | 批量 decode 与图包装；`BatchDecodeWithPagedKVCacheWrapper`、`CUDAGraphBatchDecodeWithPagedKVCacheWrapper`、`fast_decode_plan` | `flashinfer-ori: flashinfer/decode.py`；`flashinfer-ori: csrc/batch_decode.cu` | paged KV metadata + workspace；FA2/FA3 路径及 TRT-LLM/XQA/Cake/CuTe 等特定路线，以各参数验证条件为准。 |
| CAP-ATT-05 | 统一批注意力、attention sink、causal bidirectional ranges；`BatchAttention`、`BatchAttentionWithAttentionSinkWrapper`、`BatchPrefillWithCausalBidirectionalRangesWrapper` | `flashinfer-ori: flashinfer/attention/__init__.py`；`flashinfer-ori: flashinfer/attention/_core.py`；`flashinfer-ori: csrc/batch_attention.cu` | 公共 Python facade 组合 attention module；具体 mask/sink 语义是入口契约，不能仅对标一个 dense attention kernel。 |
| CAP-ATT-06 | prefill/decode 同时执行；`PODWithPagedKVCacheWrapper`、`BatchPODWithPagedKVCacheWrapper` | `flashinfer-ori: flashinfer/pod.py`；`flashinfer-ori: csrc/pod.cu`；`flashinfer-ori: csrc/batch_pod.cu` | CUDA POD 模板及独立 plan/run；是调度与融合能力，区别于两个普通 API 顺序调用。 |
| CAP-ATT-07 | shared-prefix/cascade attention；`merge_state`、`merge_states`、`merge_state_in_place`、`MultiLevelCascadeAttentionWrapper` 与两种 shared-prefix wrapper | `flashinfer-ori: flashinfer/cascade.py`；`flashinfer-ori: csrc/cascade.cu`；`flashinfer-ori: flashinfer/triton/cascade.py` | 对输出与 LSE 做状态合并；多层级 wrapper 协调独立 attention 计算；CUDA 与 Triton 实现并存。 |
| CAP-ATT-08 | MLA 与 paged MLA；`BatchMLAPagedAttentionWrapper`、TRT-LLM/CuTe 路由 | `flashinfer-ori: flashinfer/mla/_core.py`；`flashinfer-ori: flashinfer/mla/_batch_mla`；`flashinfer-ori: csrc/batch_mla_run.cu`；`flashinfer-ori: csrc/cutlass_mla.cu` | 吸收后的 MLA 专用数据布局与 scheduler；CUTLASS、TRT-LLM cubin、CuTe-DSL 与特定 Cake 路线，不能等价于普通 MHA。 |
| CAP-ATT-09 | XQA/SM110 特化 decode；`xqa`、`xqa_mla`、`sm110_gqa_decode`、`prepare_sm110_gqa_decode`、`prepare`/`attention` | `flashinfer-ori: flashinfer/decode.py`；`flashinfer-ori: flashinfer/xqa.py`；`flashinfer-ori: flashinfer/sm110_xqa.py`；`flashinfer-ori: flashinfer/experimental/sm110_gqa_decode`；`flashinfer-ori: csrc/xqa` | TRT-LLM XQA 和生成 CUDA/cubin 路线；SM110 prepared launch 是专用优化，需单独核对设备/shape。 |
| CAP-ATT-10 | Cake FMHA/request-ordered/DCP；`cake_batch_context_with_kv_cache`、`cake_batch_decode_with_kv_cache`、`CakeFmhaRequestOrderedDecodePlan`、`run_dcp_spec_decode` | `flashinfer-ori: flashinfer/cake_fmha.py`；`flashinfer-ori: flashinfer/cake_fmha_request_ordered.py`；`flashinfer-ori: flashinfer/cake_dcp.py`；`flashinfer-ori: csrc/cake_fmha` | 生成 CUDA 程序、route/manifest、prepared 参数与工作区；保留请求顺序与 decode context parallel 的定制路径。 |
| CAP-SPA-01 | block sparse / variable block sparse；`BlockSparseAttentionWrapper`、`VariableBlockSparseAttentionWrapper` | `flashinfer-ori: flashinfer/sparse.py`；`flashinfer-ori: flashinfer/cute_dsl/sparse/__init__.py` | BSR mask 转换、paged/ragged attention 复用，以及 CuTe-DSL BSA；block size、mask 和 head geometry 是独立约束。 |
| CAP-SPA-02 | DSA indexer 与 sparse attention；`dsa_indexer_topk`、`dsa_sparse_attention`、`dsa_sparse_attention_nope` 及 varlen 版本 | `flashinfer-ori: flashinfer/dsa_indexer.py`；`flashinfer-ori: flashinfer/dsa_sparse_attention.py`；`flashinfer-ori: flashinfer/cute_dsl/attention/dsa` | 索引选择与稀疏 attention 分成两个阶段；DSA indexer 连接实验 Cake backend，稀疏 attention 存在 CuTe-DSL 实现。 |
| CAP-SPA-03 | paged attention scores；`fp8_paged_mqa_logits`、`fp4_paged_mqa_logits`、`compute_paged_mqa_logits_schedule`、`precompile_paged_mqa_logits` | `flashinfer-ori: flashinfer/attn_scores/attn_scores.py`；`flashinfer-ori: flashinfer/attn_scores/kernels` | CuTe-DSL：生成 schedule，构建/缓存 kernel；输出 logits 供后续选择，并非完整 attention。 |
| CAP-SPA-04 | dense/sparse/paged MQA 实验入口；`fp8_mqa_logits`、`prepare_dense_mqa_logits`、`prepare_sparse_mqa_logits`、`prepare_paged_mqa_logits` | `flashinfer-ori: flashinfer/dense_mqa.py`；`flashinfer-ori: flashinfer/sparse_mqa.py`；`flashinfer-ori: flashinfer/paged_mqa.py`；`flashinfer-ori: flashinfer/sm120_paged_mqa_logits.py` | DeepGEMM artifact/生成 kernel 路线；与 `attn_scores` 的同名函数不可只凭名称合并。 |
| CAP-SPA-05 | MSA/VSA、top-k 选择、ICP indexer；`msa_proxy_score`、`msa_sparse_attention`、`msa_sparse_decode_attention`、`msa_topk_select`、`plan_cake_vsa`、`run_cake_vsa` | `flashinfer-ori: flashinfer/msa_ops/__init__.py`；`flashinfer-ori: flashinfer/cake_vsa.py`；`flashinfer-ori: flashinfer/cake_vsa_sm90_backend/__init__.py` | scores/selection/attention 的流水线；包含 NVFP4 cache、CuTe-DSL 与 Cake CUDA，workspace 独立管理。 |
| CAP-SPA-06 | DeepSeek V4.1/NVFP4 MLA 专用路径；`deepseek_v41_decode`、`deepseek_v41_window_decode`、`deepseek_v41_quantize_cache`、`nvfp4_attention_sm120_fwd` | `flashinfer-ori: flashinfer/deepseek_v41.py`；`flashinfer-ori: flashinfer/nvfp4_attention_sm120.py`；`flashinfer-ori: flashinfer/mla/_sparse_mla_sm120` | 显式实验 API 与模型/架构特化 CUDA；不能将存在这些入口推导为所有 DeepSeek 版本完整模型支持。 |
| CAP-GEM-01 | dense/BMM；`mm_bf16`、`bmm_bf16`、`mm_fp8`、`bmm_fp8`、`mm_mxfp8`、`bmm_mxfp8`、`mm_fp4` | `flashinfer-ori: flashinfer/gemm/__init__.py`；`flashinfer-ori: flashinfer/gemm/gemm_base.py`；`flashinfer-ori: csrc/mm_bf16_cublaslt.cu` | cuDNN、CUTLASS、cuBLASLt、CuTe-DSL、cuTile、TRT-LLM、TGV 等因 API/设备不同；以 suitability checker + runner 选择。 |
| CAP-GEM-02 | segmented/grouped GEMM；`SegmentGEMMWrapper`、`group_gemm_fp8_nt_groupwise`、`group_gemm_mxfp4_nt_groupwise`、`group_gemm_nvfp4_nt_groupwise` | `flashinfer-ori: flashinfer/gemm/gemm_base.py`；`flashinfer-ori: csrc/group_gemm.cu`；`flashinfer-ori: csrc/group_gemm_sm90.cu` | CUTLASS 分段/分组/量化 GEMM；group boundaries、scale layout 与 shape 特化独立于 dense GEMM。 |
| CAP-GEM-03 | 统一 grouped_mm；`grouped_mm_bf16`、`grouped_mm_fp8`、`grouped_mm_mxfp8`、`grouped_mm_fp4` | `flashinfer-ori: flashinfer/grouped_mm/__init__.py`；`flashinfer-ori: flashinfer/grouped_mm/core.py`；`flashinfer-ori: flashinfer/grouped_mm/cudnn` | `grouped_mm_*` 主要接 cuDNN，BF16 另有 Cake；同子包另导出 SM120 CuTe `moe_gemm_*_nt_groupwise`。这两类入口不能混为一条 backend 选择链。 |
| CAP-GEM-04 | mixed low precision、SVDQuant、LoRA；`mm_bf16_fp4`、`prepare_bf16_fp4_weights`、`mm_nvfp4_svdquant`、`svdquant_linear`、`mm_fp8_lora`、`mm_nvfp4_lora` | `flashinfer-ori: flashinfer/gemm/gemm_bf16_fp4.py`；`flashinfer-ori: flashinfer/gemm/gemm_svdquant.py`；`flashinfer-ori: flashinfer/gemm/cake_quant_lora_gemm.py` | 权重预处理、量化 scale 布局、低秩分支融合；CUTLASS/CuTe/cuDNN/Cake；`PreparedNvFp4LoraAdapter` 含 adapter 生命周期契约。 |
| CAP-GEM-05 | 低延迟/router/模型固定形状 GEMM；`trtllm_low_latency_gemm`、`prepare_low_latency_gemm_weights`、`tinygemm_bf16`、`mm_M1_16_K7168_N256` | `flashinfer-ori: flashinfer/trtllm_low_latency_gemm.py`；`flashinfer-ori: flashinfer/gemm/routergemm.py`；`flashinfer-ori: flashinfer/dsv3_ops/__init__.py` | TRT-LLM artifact + CUDA glue，以及 router-specific 手写 CUDA；固定形状入口不等同通用 GEMM。 |
| CAP-GEM-06 | masked/ragged BMM 与 prims_ts 融合线性层；`masked_bmm`、`ragged_bmm`、`ragged_scaled_bmm`、`fp8_linear_swiglu`、`fp4_qkv_qknorm_rope` | `flashinfer-ori: flashinfer/gemm/gemm_base.py`；`flashinfer-ori: flashinfer/prims_ts/__init__.py`；`flashinfer-ori: flashinfer/prims_ts/gemm` | cuTile 的 masked/ragged kernel，以及 CuTe-DSL experimental/task-scheduled GEMM + activation/norm/RoPE 融合。 |
| CAP-MOE-01 | 传统 fused MoE；`cutlass_fused_moe`、`trtllm_bf16_moe`、`trtllm_fp8_block_scale_moe`、`trtllm_fp4_block_scale_moe` 及 routed 版本 | `flashinfer-ori: flashinfer/fused_moe/core.py`；`flashinfer-ori: flashinfer/fused_moe/__init__.py`；`flashinfer-ori: csrc/fused_moe` | routing→GEMM1→activation/quant→GEMM2→finalize；CUTLASS 和 TRT-LLM cubin；routed 入口允许外部提供路由。 |
| CAP-MOE-02 | typed MoE 层/runner 抽象；`MoELayer`、`MoEConfig`、`RoutingConfig`、`QuantConfig`、`MoEWeightPack`、各 `Runner` | `flashinfer-ori: flashinfer/fused_moe/layer.py`；`flashinfer-ori: flashinfer/fused_moe/api.py`；`flashinfer-ori: flashinfer/fused_moe/runners.py` | legacy flat API 之外的统一配置/权重/执行层；CUTLASS、TRT-LLM、CuTe、cuTile、cuDNN Frost/grouped GEMM、prims_ts、B12x/Cake 等。 |
| CAP-MOE-03 | 路由与专用融合；`fused_topk_deepseek`、`alphamoe_fused_router`、`hash_topk`、`trtllm_gen_routing`、`mono_moe` | `flashinfer-ori: flashinfer/fused_moe/fused_routing_dsv3.py`；`flashinfer-ori: flashinfer/fused_moe/alphamoe_fused_router.py`；`flashinfer-ori: flashinfer/fused_moe/monomoe.py`；`flashinfer-ori: flashinfer/fused_moe/hash_topk.py`；`flashinfer-ori: flashinfer/fused_moe/trtllm_gen_routing.py`；`flashinfer-ori: flashinfer/jit/fused_moe.py` | 独立 CUDA/NVRTC/生成路由与小 batch 特化；`mono_moe` 固定 E=256、N=512、K=2048、M≤8，公开 metadata/plan 与结果布局是组合边界。 |
| CAP-MOE-04 | MoE LoRA/BGMV；`bgmv_moe`、`bgmv_moe_shrink`、`bgmv_moe_expand`、`prepare_bgmv_moe`、`bgmv_moe_gemm1_lora_delta` | `flashinfer-ori: flashinfer/fused_moe/bgmv_moe.py`；`flashinfer-ori: flashinfer/fused_moe/moe_lora_delta.py`；`flashinfer-ori: csrc/bgmv_moe` | 专用 CUDA 与 Cake prepared plan，覆盖 MoE 低秩增量注入。 |
| CAP-MOE-05 | 单节点/多节点 expert parallel；`MoEEpLayer`、`MoEEpMegaLayer`、`MoEEpSplitLayer`、`Fleet`、`Handle`、`create_communication` | `flashinfer-ori: flashinfer/moe_ep/__init__.py`；`flashinfer-ori: flashinfer/moe_ep/core/comm/communication.py`；`flashinfer-ori: flashinfer/moe_ep/modes` | 通信/计算可分离 split 模式和融合 mega 模式；NCCL-EP、NIXL-EP、NVLink、Cake；不同 capability 需显式检查。 |
| CAP-MOE-06 | MegaMoE 专用路线；`MegaMoESymmBuffer`、`nvfp4_mega_moe`、`mxfp4_mega_moe`、`select_fused_plan`、`select_split_plan` | `flashinfer-ori: flashinfer/megamoe/__init__.py`；`flashinfer-ori: flashinfer/megamoe/heuristics.py`；`flashinfer-ori: flashinfer/mega_moe_v3.py` | Hopper FP4 megakernel、对称内存与 workspace heuristic；V3 是另一个实验 runtime，不能与 `moe_ep` 统一概括为单一实现。 |
| CAP-COM-01 | AllReduce / fusion；`allreduce_fusion`、`create_allreduce_fusion_workspace`、`trtllm_allreduce_fusion`、`vllm_all_reduce` | `flashinfer-ori: flashinfer/comm/__init__.py`；`flashinfer-ori: flashinfer/comm/allreduce.py`；`flashinfer-ori: flashinfer/comm/trtllm_ar.py`；`flashinfer-ori: flashinfer/comm/vllm_ar.py`；`flashinfer-ori: csrc/vllm_custom_all_reduce.cu` | TRT-LLM/vLLM CUDA 实现、CuTe MNNVL、workspace 抽象；含 RMSNorm/quant/finalize 融合，不只是 collective API。 |
| CAP-COM-02 | PCIe IPC、NVSHMEM、MNNVL 与拓扑；`PcieIpcAllReduceWorkspace`、`PcieIpcAllGatherWorkspace`、`PcieIpcReduceScatterWorkspace`、`Mapping` | `flashinfer-ori: flashinfer/comm/__init__.py`；`flashinfer-ori: flashinfer/comm/pcie_ipc_collectives`；`flashinfer-ori: flashinfer/comm/mnnvl.py`；`flashinfer-ori: flashinfer/comm/nvshmem.py` | CUDA IPC、对称内存、拓扑策略和设备通信；可选依赖与设备连接结构影响可用 backend。 |
| CAP-COM-03 | 计算/通信重叠；`all_gather_matmul`、`fp8_rowwise_gemm_allreduce`、`QkvNormRopeExchangeOverlap` | `flashinfer-ori: flashinfer/comm/all_gather_matmul/__init__.py`；`flashinfer-ori: flashinfer/comm/fp8_rowwise_gemm_allreduce/__init__.py`；`flashinfer-ori: flashinfer/comm/qkv_norm_rope_exchange_overlap.py` | prepare/执行分离，融合 CUDA kernel 与多 stream/通信工作区；低精度数据布局是接口的一部分。 |
| CAP-COM-04 | context parallel / Ulysses / ring；`UlyssesCommunicator`、`ulysses_a2a`、`ParallelAttention`、`decode_cp_a2a_alltoall`、`decode_cp_a2a_lse_reduce` | `flashinfer-ori: flashinfer/parallel_attention/parallel_attention.py`；`flashinfer-ori: flashinfer/comm/ulysses.py`；`flashinfer-ori: flashinfer/comm/dcp_alltoall.py`；`flashinfer-ori: flashinfer/comm/dcp_lse_reduce.py` | attention 与分布式分片/交换/结果归约组合；ring/Ulysses 由 parallel wrapper 调度，DCP 是 decode 专用。 |
| CAP-SMP-01 | softmax、随机采样、约束采样、speculative sampling；`sampling_from_logits`、`top_k_top_p_sampling_from_probs`、`min_p_sampling_from_probs`、`chain_speculative_sampling` | `flashinfer-ori: flashinfer/sampling.py`；`flashinfer-ori: csrc/sampling.cu`；`flashinfer-ori: include/flashinfer/sampling.cuh` | CUDA 采样/renorm/mask；seed+offset 与 generator 语义；Cake Blackwell 采样另有入口 `cake_sampling.py`。 |
| CAP-SMP-02 | logits pipeline；`LogitsPipe`、`Temperature`、`Softmax`、`TopK`、`TopP`、`MinP`、`Sample`、`compile_pipeline` | `flashinfer-ori: flashinfer/logits_processor/__init__.py`；`flashinfer-ori: flashinfer/logits_processor/pipeline.py`；`flashinfer-ori: flashinfer/logits_processor/compiler.py` | Python typed processor→legalization→fusion-rule 编译→底层 sampling ops；此处 compile 是 pipeline 合并，不能混同 NVCC 或 torch.compile。 |
| CAP-SEL-01 | TopK / page table transform / variable length TopK；`top_k`、`top_k_page_table_transform`、`top_k_ragged_transform`、`top_k_varlen` | `flashinfer-ori: flashinfer/topk.py`；`flashinfer-ori: flashinfer/topk_varlen/topk_varlen.py`；`flashinfer-ori: csrc/cub_topk.cu` | 手写 CUDA/CUB cluster/radix 与 CuTe GVR/GVR2；支持选择结果直接转 paged/ragged 索引，连接稀疏注意力。 |
| CAP-NRM-01 | RMSNorm/LayerNorm/Gemma/fused residual/quant；`rmsnorm`、`fused_add_rmsnorm`、`gemma_rmsnorm`、`layernorm`、`rmsnorm_quant` | `flashinfer-ori: flashinfer/norm/__init__.py`；`flashinfer-ori: csrc/norm.cu`；`flashinfer-ori: include/flashinfer/norm.cuh` | CUDA 与 CuTe-DSL 补充 kernel；norm + quant/SiLU/QK RoPE/DIT residual 是独立融合接口。 |
| CAP-ACT-01 | activation gating 与量化融合；`silu_and_mul`、`gelu_and_mul`、`gelu_tanh_and_mul`、`silu_and_mul_mxfp8_quantize` 及 backward | `flashinfer-ori: flashinfer/activation.py`；`flashinfer-ori: flashinfer/gated_act_mxfp8.py`；`flashinfer-ori: include/flashinfer/activation.cuh` | CUDA JIT 与低精度 fused activation；存在 backward API，不能将基线概括为纯推理前向。 |
| CAP-POS-01 | RoPE/Llama3.1/缓存 RoPE/量化入 KV；`apply_rope`、`apply_rope_pos_ids`、`apply_llama31_rope`、`rope_quantize_fp8_append_paged_kv_cache` | `flashinfer-ori: flashinfer/rope.py`；`flashinfer-ori: csrc/rope.cu`；`flashinfer-ori: include/flashinfer/pos_enc.cuh` | CUDA 与其他 backend；in-place/out-of-place、显式 position IDs、cos/sin cache、RoPE→quant→append 的融合接口。 |
| CAP-QNT-01 | FP4/MXFP4/NVFP4、FP8/MXFP8、packbits；`fp4_quantize`、`mxfp4_quantize`、`nvfp4_quantize`、`mxfp8_quantize`、`packbits`、`segment_packbits` | `flashinfer-ori: flashinfer/quantization/__init__.py`；`flashinfer-ori: flashinfer/quantization/fp4_quantization.py`；`flashinfer-ori: flashinfer/quantization/fp8_quantization.py`；`flashinfer-ori: flashinfer/quantization/packbits.py` | CUDA/CuTe/cuTile 等；包含 scale interleave/shuffle、grouped/per-token quant、host dequant、NVFP4 4over6 recipe；这些布局工具是 GEMM/KV 的共同契约。 |
| CAP-KVC-01 | paged KV append / 长度与索引 / MLA packing；`append_paged_kv_cache`、`append_paged_mla_kv_cache`、`get_seq_lens`、`get_batch_indices_positions` | `flashinfer-ori: flashinfer/page.py`；`flashinfer-ori: csrc/page.cu`；`flashinfer-ori: include/flashinfer/page.cuh`；`flashinfer-ori: flashinfer/mla_kv_pack.py` | CUDA kernel 处理已分配 page 的写入与索引；包含 NVFP4 quantize append、slot_mapping；没有证据表明这是完整服务端 KV page 分配器。 |
| CAP-KVC-02 | KV 转换与 QK 融合追加；`concat_mla_k`、`cake_fused_qk_rmsnorm_rope_append_paged_kv_cache`、FP8 对应融合 API | `flashinfer-ori: flashinfer/concat_ops.py`；`flashinfer-ori: flashinfer/cake_fused_qk_rope_append.py`；`flashinfer-ori: flashinfer/cake_fused_qk_rope_fp8_append.py` | MLA no-PE/PE 拼接及归一化→RoPE→量化→cache append 的生成 CUDA 路线；数据流融合是对标点。 |
| CAP-LIN-01 | GDN/GDN2/GDP；`chunk_gated_delta_rule`、`chunk_gated_delta_rule2`、`chunk_gated_delta_product`、`gated_delta_rule_decode`、`gated_delta_rule_mtp` | `flashinfer-ori: flashinfer/gdn_prefill.py`；`flashinfer-ori: flashinfer/gdn2_prefill.py`；`flashinfer-ori: flashinfer/gdp_prefill.py`；`flashinfer-ori: flashinfer/gdn_decode.py` | CuTe-DSL、Cake/生成 CUDA、cuDNN linear attention；pretranspose、MTP、state dtype、状态更新/回滚各有契约。 |
| CAP-LIN-02 | KDA 推理与训练；`recurrent_kda`、`RecurrentKDAPrefillWrapper`、`RecurrentKDAPrefillWorkspace`、`fused_kda_decode`、`packed_kda_decode`、`chunk_kda` | `flashinfer-ori: flashinfer/kda.py`；`flashinfer-ori: flashinfer/kda_prefill.py`；`flashinfer-ori: flashinfer/kda_decode.py`；`flashinfer-ori: flashinfer/kda_train.py` | Cake CUDA/NVRTC、CuTe、PTX、persistent、TIRx 与 cuDNN；`chunk_kda` 是显式实验训练入口。 |
| CAP-SSM-01 | Mamba/SSM；`selective_state_update`、`checkpointing_ssu`、`allocate_checkpointing_ssu_scratch`、`replayssm_materialize`、`SSDCombined`、`ssd_combined_fwd` | `flashinfer-ori: flashinfer/mamba/__init__.py`；`flashinfer-ori: flashinfer/mamba/ssd_combined.py`；`flashinfer-ori: csrc/selective_state_update.cu` | CUDA 状态更新 + checkpoint/replay；SSDCombined 组合 CUDA chunk cumsum、CuTe SSD kernel 或 Cake 路线，Triton 也存在于 chunk-state 计算。 |
| CAP-MOD-01 | mHC / gated residual；`mhc_post`、`mhc_pre_big_fuse`、`prepare_mega_mhc`、`prepare_mega_gate`、Cake mHC training | `flashinfer-ori: flashinfer/mhc.py`；`flashinfer-ori: flashinfer/mega_mhc.py`；`flashinfer-ori: flashinfer/mega_gate.py`；`flashinfer-ori: flashinfer/cake_mhc_train.py` | CUDA/CuTe/DeepGEMM 生成实现；前向 prepare/stats/finish 与反向 prepare/finish 是更大粒度融合，不应归入普通 RMSNorm 后忽略。 |
| CAP-MOD-02 | diffusion/MiniMaxH3；DIT norm、QKV quant/pack、attention、fc1 SwiGLU、out projection、MLP | `flashinfer-ori: flashinfer/diffusion_ops/__init__.py`；`flashinfer-ori: flashinfer/cake_minimax_h3.py`；`flashinfer-ori: flashinfer/experimental/minimax_h3_varlen_attention` | 模型阶段级生成 CUDA 优化；BF16、MXFP8、NVFP4 及 SM120 量化路线是分别暴露的入口。 |
| CAP-MOD-03 | 模型专用实验模块；`Qwen35BlockWrapper`、`prepare_mok_bf16`、Kimi-K3 vision/latent-MoE/attention-residual/TP12-tail、`chunked_lm_head_loss` | `flashinfer-ori: flashinfer/qwen_3_5_block.py`；`flashinfer-ori: flashinfer/mok.py`；`flashinfer-ori: flashinfer/kimi_k3_vision.py`；`flashinfer-ori: flashinfer/chunked_lm_head.py` | 由 core 薄入口转交 experimental；包括整块、多 kernel 链与训练损失；仅能确认所列算子/阶段，不等于端到端模型支持。 |
| CAP-TRN-01 | Cake 训练辅助；`CakeRMSNormFunction`、`cake_rmsnorm`、`CakeCausalConv1dFunction`、`cake_causal_conv1d` | `flashinfer-ori: flashinfer/cake_rmsnorm_train.py`；`flashinfer-ori: flashinfer/cake_causal_conv1d_train.py`；`flashinfer-ori: flashinfer/cake_mhc_train.py` | `torch.autograd.Function` 与显式 backward/workspace API + 生成 CUDA；说明调查基线含训练向融合能力。 |

### 2.3 全部直接子目录及平台层

实际扫描 `flashinfer` 得到 **42 个直接子目录、471 个各层目录**（不计 `__pycache__`）。
本表逐一列出 42 个直接子目录；§2.7 给出其余深层目录的完整路径覆盖。
“内部”表示实现或支撑目录，并不暗示其 Python 符号绝对不可导入。

| 目录（证据） | 能力/代表公开符号或内部职责 | 实现来源与交付位置 |
|---|---|---|
| `flashinfer-ori: flashinfer/attention` | `BatchAttention` 与 sink/ranges wrapper；prims_ts、CuTe 与 cuTile attention backend | public façade + backend adapters/kernels；见 CAP-ATT。 |
| `flashinfer-ori: flashinfer/attn_scores` | FP8/FP4 paged MQA logits、schedule、precompile | CuTe-DSL host 与 kernels；见 CAP-SPA-03。 |
| `flashinfer-ori: flashinfer/autotuner` | `AutoTuner`、`TunableRunner`、`TuningConfig`、`OptimizationProfile`、`DynamicDim`、`ValueProfileArena`、`autotune` | Python 候选 runner、分桶、profile、缓存；§3.5。 |
| `flashinfer-ori: flashinfer/cake_causal_conv1d_train_backend` | 前/反向、`CakeCausalConv1dFunction`、workspace 计算 | `cake_jit.py`、manifest、csrc 生成 CUDA；由同名 core façade 调用。 |
| `flashinfer-ori: flashinfer/cake_fused_qk_rope_append_backend` | `launch_plan`、QK RMSNorm/RoPE/append | arch/stage/route 元数据 + csrc CUDA；与 FP8 backend 分开。 |
| `flashinfer-ori: flashinfer/cake_fused_qk_rope_fp8_append_backend` | `launch_plan`、QK RMSNorm/RoPE/FP8 quant/append | arch/stage/route 元数据 + csrc CUDA。 |
| `flashinfer-ori: flashinfer/cake_mhc_train_backend` | `CakeMHCTrainPrepareFunction`、`CakeMHCTrainFinishFunction`、workspace/reset | 生成 CUDA 分阶段训练融合。 |
| `flashinfer-ori: flashinfer/cake_nvfp4_svdquant_gemm_backend` | `is_supported`、`run`、`select_record` | SVDQuant backend route + generated CUDA；由 GEMM façade 接入。 |
| `flashinfer-ori: flashinfer/cake_rmsnorm_train_backend` | `CakeRMSNormFunction`、forward/backward/workspace | 生成 CUDA 与 autograd 包装。 |
| `flashinfer-ori: flashinfer/cake_sage_block_sparse_attention_backend` | `bsa_attn_sm100_sage_fwd_cake`、`bsa_attn_sm120_blk64_sage_fwd_cake`、quant | Cake Sage BSA 的 SM100/SM120 kernel 集。 |
| `flashinfer-ori: flashinfer/cake_ulysses_backend` | `CakeUlyssesA2A`、`route_arch` | 生成 CUDA all-to-all、rank/peer table/signals；作为通信 backend。 |
| `flashinfer-ori: flashinfer/cake_vsa_sm90_backend` | `CakeVsaSm90Plan`、`create_plan`、`plan_vsa_sm90` | CUDA VSA kernel + host planner；small/persistent/split 策略。 |
| `flashinfer-ori: flashinfer/comm` | collective、IPC workspace、通信/计算融合与拓扑选择 | TRT-LLM/vLLM/Cake CUDA、CuTe、NVSHMEM/PyTorch symmetric memory；见 CAP-COM。 |
| `flashinfer-ori: flashinfer/cudnn` | `cudnn_batch_decode_with_kv_cache`、`cudnn_batch_prefill_with_kv_cache`、GDN/GDP/KDA linear attention | cuDNN Python/frontend 图构造与缓存，而不是 FlashInfer CUDA template 的别名。 |
| `flashinfer-ori: flashinfer/cute_dsl` | 可用性探测；attention 构件/variants、BSA、通用 DSL 工具 | Python CuTe-DSL 源码、JIT kernel 与可复用 scheduler/roles/fusion。 |
| `flashinfer-ori: flashinfer/cutile` | `is_cuda_tile_available` 与环境/版本共用逻辑 | cuda.tile 支撑；具体 kernels 分布于 attention/gemm/MoE/quantization。 |
| `flashinfer-ori: flashinfer/diffusion_ops` | MiniMaxH3/DIT 成组阶段 API | CUDA/CUTLASS 量化融合与专用 generated programs。 |
| `flashinfer-ori: flashinfer/dsv3_ops` | `mm_M1_16_K7168_N128`、`mm_M1_16_K7168_N256`、`fused_topk_deepseek`、`concat_mla_k` | 聚合重导出既有 GEMM/MoE/KV 操作；无独立 kernel 树。 |
| `flashinfer-ori: flashinfer/experimental` | 实验实现与 feature gates；DeepGEMM、Cake、b12x、cuDNN Frost、模型特化 | 无兼容性保证；core 入口可显式标实验；详见 §2.5。 |
| `flashinfer-ori: flashinfer/fused_moe` | flat API、`MoELayer`、typed config、runners、routing、LoRA | 多 backend MoE 实现/适配/权重预处理。 |
| `flashinfer-ori: flashinfer/gdn_kernels` | decode/MTP/prefill/prefix-materialize、fused decode step | CuTe-DSL/Cake 后端、状态布局与 kernel helper；并非单个 GDN kernel。 |
| `flashinfer-ori: flashinfer/gemm` | dense/BMM/grouped/LoRA/SVDQuant/router APIs | 多 backend 主调度模块与 kernel 目录。 |
| `flashinfer-ori: flashinfer/grouped_mm` | BF16/FP8/MXFP8/FP4 grouped_mm | 统一 façade 以 cuDNN 为主，BF16 可选 Cake；另导出 SM120 CuTe 分组 GEMM API。 |
| `flashinfer-ori: flashinfer/jit` | `JitSpec`、`gen_jit_spec`、operator module generators、cubin loader | 编译/加载和代码生成基础设施；§3.1、§3.2。 |
| `flashinfer-ori: flashinfer/kda_kernels` | `run_recurrent_kda`、`run_fused_kda_decode`、packed decode | CuTe、PTX、TIRx、persistent/SM120 子实现。 |
| `flashinfer-ori: flashinfer/logits_processor` | `LogitsPipe`、processor/Op/fusion rule/compiler | Python 合法化/融合层复用 sampling API。 |
| `flashinfer-ori: flashinfer/mamba` | SSU、checkpoint/replay、SSDCombined | CUDA + CuTe + Triton + Cake 子路径。 |
| `flashinfer-ori: flashinfer/megamoe` | FP4 megakernel、symmetric buffer、weight transform/heuristic | Hopper CUDA kernel route；与 `moe_ep` 并存。 |
| `flashinfer-ori: flashinfer/mla` | `BatchMLAPagedAttentionWrapper`、sparse MLA、模型特化入口 | façade + 独立 backend planner/run/metadata。 |
| `flashinfer-ori: flashinfer/moe_ep` | `Fleet`、`Handle`、`MoEEpLayer`、split/mega 模式、backend registry | 通信、配置验证、runtime、设备 kernel、vendored source 联合组成 EP 执行栈。 |
| `flashinfer-ori: flashinfer/msa_ops` | MSA score/indexer/topk/attention 与 prepared workspace | CUDA/CuTe-DSL、NVFP4 特化。 |
| `flashinfer-ori: flashinfer/norm` | norm/fused residual/quant/DIT/QK-RoPE | CUDA JIT 与 `kernels` 中 DSL 实现。 |
| `flashinfer-ori: flashinfer/parallel_attention` | `ParallelAttention`、`UnevenCPConfig`、`VarlenCPConfig` | Python ring/Ulysses wrapper 与 attention op manager；组合通信和 attention。 |
| `flashinfer-ori: flashinfer/prims_ts` | fused linear/QKV、batched GEMM、MoE task scheduling | CuTe-DSL experimental primitives；依赖能力探测与 lazy import。 |
| `flashinfer-ori: flashinfer/profiler` | `EventType`、`decode_tag`、`export_to_perfetto_trace` | 解码设备事件 buffer 并导出 Perfetto trace；配对 `flashinfer-ori: include/flashinfer/profiler.cuh`。 |
| `flashinfer-ori: flashinfer/quantization` | packbits、FP4/FP8/MXFP8/NVFP4、scale layout | CUDA/CuTe/cuTile、host reference/dequant 与配置 utilities。 |
| `flashinfer-ori: flashinfer/testing` | `bench_gpu_time`、CUDA event/CUDA Graph/CUPTI 测时、attention FLOPs/带宽计算 | Python 测试性能辅助库，供 benchmarks/tests 和用户脚本复用。 |
| `flashinfer-ori: flashinfer/topk_varlen` | `top_k_varlen`；`warmup_prefill`、`prefill_ready`、`release_gvr2_resources` | CuTe GVR/GVR2、radix/filter kernel，提供显式 warmup/capture readiness。 |
| `flashinfer-ori: flashinfer/trace` | `TraceTemplate`、`Var`、`Tensor`、`Scalar`、`BuildSpec`、`Solution`、`SourceFile` | 问题规格/输入输出/schema/check 与算法替换方案描述；按能力提供 templates。 |
| `flashinfer-ori: flashinfer/trace_apply` | `enable_apply`、`disable_apply`、`stats`；Python/C++ solution loader | 外部方案匹配并替换调用；C++ loader 经 `gen_jit_spec` + TVM FFI 绑定。 |
| `flashinfer-ori: flashinfer/triton` | cascade、SM constraint GEMM；activation/norm/quant/SSD 子 kernel | Triton 实现/辅助路线；不能概括为全库都通过 Triton。 |
| `flashinfer-ori: flashinfer/tuning_configs` | B200/GB200 TRT-LLM fused MoE 预设数据 | 已提交的 Python tuning 数据，区别于用户运行期 profile cache。 |

### 2.4 顶层模块覆盖与支撑功能

以下把顶层模块按职责分组，补齐能力表之外的脚本与辅助模块；所有列出的路径实际存在。
兼容重导出与实现模块分别计入覆盖，避免目录迁移后的同名文件被误认为另一套 kernel。

| 分组 | 顶层文件证据 | 作用与代表符号 |
|---|---|---|
| 包/CLI/版本 | `flashinfer-ori: flashinfer/__init__.py`；`flashinfer-ori: flashinfer/__main__.py`；`flashinfer-ori: flashinfer/version.py`；`flashinfer-ori: flashinfer/py.typed` | 导出表、lazy/conditional imports、CLI、包版本、类型标记。 |
| attention façade | `flashinfer-ori: flashinfer/decode.py`；`flashinfer-ori: flashinfer/prefill.py`；`flashinfer-ori: flashinfer/cascade.py`；`flashinfer-ori: flashinfer/pod.py`；`flashinfer-ori: flashinfer/sparse.py` | 通用 attention、POD、cascade、block sparse API。 |
| attention 专用路由 | `flashinfer-ori: flashinfer/xqa.py`；`flashinfer-ori: flashinfer/sm110_xqa.py`；`flashinfer-ori: flashinfer/cake_fmha.py`；`flashinfer-ori: flashinfer/cake_fmha_request_ordered.py`；`flashinfer-ori: flashinfer/cake_dcp.py` | XQA、SM110、Cake FMHA manifest/route、request-ordered plan、DCP 参数计算。 |
| 选择性/稀疏 attention | `flashinfer-ori: flashinfer/dsa_indexer.py`；`flashinfer-ori: flashinfer/dsa_sparse_attention.py`；`flashinfer-ori: flashinfer/cake_vsa.py`；`flashinfer-ori: flashinfer/cake_vsa_sm90.py`；`flashinfer-ori: flashinfer/nvfp4_attention_sm120.py`；`flashinfer-ori: flashinfer/deepseek_v41.py` | DSA/VSA/NVFP4 与模型特化 façade。 |
| MQA scores | `flashinfer-ori: flashinfer/dense_mqa.py`；`flashinfer-ori: flashinfer/paged_mqa.py`；`flashinfer-ori: flashinfer/sparse_mqa.py`；`flashinfer-ori: flashinfer/sm120_paged_mqa_logits.py` | DeepGEMM 家族 metadata/prepare/logits。 |
| KV/RoPE/concat | `flashinfer-ori: flashinfer/page.py`；`flashinfer-ori: flashinfer/rope.py`；`flashinfer-ori: flashinfer/concat_ops.py`；`flashinfer-ori: flashinfer/mla_kv_pack.py`；`flashinfer-ori: flashinfer/mla_kv_pack_fp8_workloads.json` | page append、positions、RoPE、MLA pack 与已提交 workload 数据。 |
| QK 融合追加 | `flashinfer-ori: flashinfer/cake_fused_qk_rope_append.py`；`flashinfer-ori: flashinfer/cake_fused_qk_rope_fp8_append.py` | QK RMSNorm/RoPE/quant/cache append 的薄入口。 |
| activation/采样 | `flashinfer-ori: flashinfer/activation.py`；`flashinfer-ori: flashinfer/gated_act_mxfp8.py`；`flashinfer-ori: flashinfer/sampling.py`；`flashinfer-ori: flashinfer/cake_sampling.py`；`flashinfer-ori: flashinfer/topk.py` | activation、softmax、sampling/renorm、TopK。 |
| 量化兼容入口 | `flashinfer-ori: flashinfer/fp4_quantization.py`；`flashinfer-ori: flashinfer/fp8_quantization.py` | 指向 `quantization` 子包的兼容层，不是独立实现源。 |
| DeepGEMM/低精度 GEMM | `flashinfer-ori: flashinfer/deep_gemm.py`；`flashinfer-ori: flashinfer/fp4_gemm.py`；`flashinfer-ori: flashinfer/fp4_k_grouped_gemm.py`；`flashinfer-ori: flashinfer/fp8_batched_gemm.py`；`flashinfer-ori: flashinfer/fp8_fp4_gemm.py`；`flashinfer-ori: flashinfer/trtllm_low_latency_gemm.py` | artifact/runtime 与 prepare plan；FP4/FP8 mixed/grouped/batched 与 low-latency GEMM。 |
| GDN/GDP | `flashinfer-ori: flashinfer/gdn_decode.py`；`flashinfer-ori: flashinfer/gdn_prefill.py`；`flashinfer-ori: flashinfer/gdn2_prefill.py`；`flashinfer-ori: flashinfer/gdp_prefill.py` | linear attention 前缀与 recurrent/MTP 状态更新 API。 |
| KDA 公共生命周期 | `flashinfer-ori: flashinfer/kda.py`；`flashinfer-ori: flashinfer/kda_decode.py`；`flashinfer-ori: flashinfer/kda_prefill.py`；`flashinfer-ori: flashinfer/kda_train.py` | recurrent/prefill wrapper、workspace、decode、训练前反向。 |
| KDA backend 入口 | `flashinfer-ori: flashinfer/kda_prefill_cute.py`；`flashinfer-ori: flashinfer/kda_prefill_cute_small_bh.py`；`flashinfer-ori: flashinfer/kda_prefill_persistent.py`；`flashinfer-ori: flashinfer/kda_prefill_ptx.py`；`flashinfer-ori: flashinfer/kda_prefill_tirx.py`；`flashinfer-ori: flashinfer/cake_kda_tf32_runtime.py` | 多编译路线及 prepared launch/rebind；TIRx 分支要求 CUDA-enabled TVM。 |
| mHC/mega | `flashinfer-ori: flashinfer/mhc.py`；`flashinfer-ori: flashinfer/mega_gate.py`；`flashinfer-ori: flashinfer/mega_mhc.py`；`flashinfer-ori: flashinfer/mega_moe_v3.py` | mHC、gate 融合与 V3 MoE runtime facade。 |
| 训练融合 | `flashinfer-ori: flashinfer/cake_rmsnorm_train.py`；`flashinfer-ori: flashinfer/cake_causal_conv1d_train.py`；`flashinfer-ori: flashinfer/cake_mhc_train.py`；`flashinfer-ori: flashinfer/chunked_lm_head.py` | autograd 函数、显式 forward/backward、分块 LM head loss/logprob。 |
| 模型块/专家特化 | `flashinfer-ori: flashinfer/cake_minimax_h3.py`；`flashinfer-ori: flashinfer/kimi_k3_attn_res.py`；`flashinfer-ori: flashinfer/kimi_k3_latent_moe.py`；`flashinfer-ori: flashinfer/kimi_k3_tp12_tail.py`；`flashinfer-ori: flashinfer/kimi_k3_vision.py`；`flashinfer-ori: flashinfer/qwen_3_5_block.py`；`flashinfer-ori: flashinfer/mok.py` | prepared model stage、vision/latent MoE、Qwen block、MOK；实验属性以具体装饰器为准。 |
| 编译/交付 | `flashinfer-ori: flashinfer/aot.py`；`flashinfer-ori: flashinfer/artifacts.py`；`flashinfer-ori: flashinfer/compilation_context.py` | AOT 注册/批编译，artifact 下载校验，目标架构上下文。 |
| 调优/屏蔽 | `flashinfer-ori: flashinfer/autotune_cache.py`；`flashinfer-ori: flashinfer/tactics_blocklist.py`；`flashinfer-ori: flashinfer/tactics_blocklist_gen.py` | managed tuning cache、`MeasurementPolicy`、`TacticsBlocklist` 与生成工具。 |
| 日志/trace | `flashinfer-ori: flashinfer/api_logging.py`；`flashinfer-ori: flashinfer/fi_trace.py` | `flashinfer_api`、统计/dump/replay、`register_fi_trace`；连接 trace template 与方案替换。 |
| 设备/环境辅助 | `flashinfer-ori: flashinfer/utils.py`；`flashinfer-ori: flashinfer/cuda_utils.py`；`flashinfer-ori: flashinfer/collect_env.py`；`flashinfer-ori: flashinfer/green_ctx.py` | dtype/layout/mask 枚举、设备支持 checker、日志等级、环境采集；CUDA green context 的 SM 资源划分/stream 创建。 |
| TRT-LLM 类型适配 | `flashinfer-ori: flashinfer/tllm_enums.py`；`flashinfer-ori: flashinfer/tllm_utils.py` | `RoutingMethodType`、`ActivationType`、`SfLayout`、`WeightLayout`、dtype 推导与 delay kernel。 |

### 2.5 experimental 子树：不能遗漏的能力组

这些目录有时存放内部实现，有时通过 core 中标注的实验函数公开。
本表把大类映射到实际代码路径；更深层目录继续在 §2.7 点名。

| 实验能力组 | 实际目录/文件证据 | 可验证的入口/实现职责 |
|---|---|---|
| b12x 总体 | `flashinfer-ori: flashinfer/experimental/b12x/_api.py`；`flashinfer-ori: flashinfer/experimental/b12x/__init__.py` | `_OPS` 静态 registry、`list_ops`、`find_op`、`Plan`/`PreparationSession`；`__init__` 别名到 `b12x`，CuTe-DSL/Triton 实现消费级 Blackwell 路线。 |
| b12x attention | `flashinfer-ori: flashinfer/experimental/b12x/attention` | paged/dense MLA/sparse MLA/compressed MLA/MLA compress/DSA indexer/QSA/varlen；_api registry 可证逻辑 op，而非按目录猜测。 |
| b12x GEMM/quant | `flashinfer-ori: flashinfer/experimental/b12x/gemm`；`flashinfer-ori: flashinfer/experimental/b12x/quantization` | BF16 GEMV/vocab projection、blockscaled、FP8 linear、MLA/WO/trellis projection；MXFP8/NVFP4，另有 MXFP6 实现目录。 |
| b12x MoE/通信 | `flashinfer-ori: flashinfer/experimental/b12x/moe`；`flashinfer-ori: flashinfer/experimental/b12x/comm` | fused MoE、EP MoE；PCIe/RoCE；权重格式 kernel 及 checkpoint 适配。 |
| b12x sequence/norm | `flashinfer-ori: flashinfer/experimental/b12x/sequence`；`flashinfer-ori: flashinfer/experimental/b12x/norm` | GDN/KDA、MTP feedback、embedding/PLE/hash/engram；hyperconnection/mHC；不等同标准 attention。 |
| b12x 基础设施 | `flashinfer-ori: flashinfer/experimental/b12x/_compat`；`flashinfer-ori: flashinfer/experimental/b12x/_lib`；`flashinfer-ori: flashinfer/experimental/b12x/loader`；`flashinfer-ori: flashinfer/experimental/b12x/preparation`；`flashinfer-ori: flashinfer/experimental/b12x/testing`；`flashinfer-ori: flashinfer/experimental/b12x/tools` | 兼容层、编译/缓存/meta/准备会话、加载与测试；与经典 JitSpec 路线并存，不能统一断言所有 kernel 都走 NVCC shared library。 |
| GQA/SM110 | `flashinfer-ori: flashinfer/experimental/balanced_gqa_decode`；`flashinfer-ori: flashinfer/experimental/sm110_gqa_decode`；`flashinfer-ori: flashinfer/experimental/sm110_xqa` | balanced/SM110 decode route 与 prepared launch 的生成 CUDA 实现。 |
| DSA/训练/KDA | `flashinfer-ori: flashinfer/experimental/cake_dsa_indexer`；`flashinfer-ori: flashinfer/experimental/cake_dsa_train`；`flashinfer-ori: flashinfer/experimental/cake_kda_train`；`flashinfer-ori: flashinfer/experimental/kda_prefill_wrapper` | indexer/topk、DSA train、KDA 前反向与 prefill 包装。 |
| MLA/DCP/NVFP4 | `flashinfer-ori: flashinfer/experimental/cake_mla_nvfp4_paged_decode`；`flashinfer-ori: flashinfer/experimental/cake_mla_varq_dcp_decode`；`flashinfer-ori: flashinfer/experimental/nvfp4_mla_decode`；`flashinfer-ori: flashinfer/experimental/nvfp4_sparse_mla_decode` | paged/variable-Q、DCP 与稀疏 NVFP4 MLA backend；调用方还需匹配 cache layout。 |
| 选择/低精度 attention | `flashinfer-ori: flashinfer/experimental/cake_msa_icp_indexer`；`flashinfer-ori: flashinfer/experimental/msa_nvfp4_decode`；`flashinfer-ori: flashinfer/experimental/nvfp4_attention`；`flashinfer-ori: flashinfer/experimental/deepseek_v41` | ICP indexer、MSA decode、NVFP4 attention、DeepSeek 特定 cache/attention。 |
| MoE/Cake | `flashinfer-ori: flashinfer/experimental/cake_moe_grouped_gemm`；`flashinfer-ori: flashinfer/experimental/cake_stepfun_moe`；`flashinfer-ori: flashinfer/experimental/cake_mxfp8_megamoe_ep16`；`flashinfer-ori: flashinfer/experimental/cake_w8a8_megamoe_ep16`；`flashinfer-ori: flashinfer/experimental/mega_moe_v3` | grouped GEMM、StepFun、EP16、mega pipeline；大多通过 prepared plan 绑定形状/权重/通信资源。 |
| cuDNN Frost MoE | `flashinfer-ori: flashinfer/experimental/cudnn_frost_selected_kernels_moe_grouped_gemm` | BF16/MXFP8/MXFP8×MXFP4/NVFP4 导出 kernel 支路；与 fused_moe 正式 backend 目录存在迁移/适配关系，不能据目录名推定稳定状态。 |
| DeepGEMM 基础/GEMM | `flashinfer-ori: flashinfer/experimental/deepgemm_common`；`flashinfer-ori: flashinfer/experimental/deepgemm_batched_gemm`；`flashinfer-ori: flashinfer/experimental/deepgemm_fp4_gemm`；`flashinfer-ori: flashinfer/experimental/deepgemm_fp8_gemm`；`flashinfer-ori: flashinfer/experimental/deepgemm_kgroup_gemm`；`flashinfer-ori: flashinfer/experimental/deepgemm_mixed_gemm` | artifact、plan/runtime 与 batched/FP4/FP8/K-group/mixed GEMM 特化。 |
| DeepGEMM attention/融合 | `flashinfer-ori: flashinfer/experimental/deepgemm_dense_mqa`；`flashinfer-ori: flashinfer/experimental/deepgemm_sparse_mqa`；`flashinfer-ori: flashinfer/experimental/deepgemm_sm120_paged_mqa_logits`；`flashinfer-ori: flashinfer/experimental/deepgemm_mega_gate`；`flashinfer-ori: flashinfer/experimental/deepgemm_mega_mhc` | MQA score 与 mega gate/mHC；由同名 core prepare API 暴露。 |
| Kimi-K3 | `flashinfer-ori: flashinfer/experimental/cake_kimi_k3_attn_res`；`flashinfer-ori: flashinfer/experimental/kimi_k3_fp8_projection`；`flashinfer-ori: flashinfer/experimental/kimi_k3_fused_router`；`flashinfer-ori: flashinfer/experimental/kimi_k3_latent_moe`；`flashinfer-ori: flashinfer/experimental/kimi_k3_tp12_tail`；`flashinfer-ori: flashinfer/experimental/kimi_k3_vision_tower` | residual、FP8 projection、路由、latent MoE、TP12 tail、vision tower 的模型级生成 CUDA。 |
| 其他模型与训练 | `flashinfer-ori: flashinfer/experimental/cake_lm_head_loss`；`flashinfer-ori: flashinfer/experimental/cake_mok_bf16`；`flashinfer-ori: flashinfer/experimental/cake_nvfp4_per_token`；`flashinfer-ori: flashinfer/experimental/dense_projection_gemm`；`flashinfer-ori: flashinfer/experimental/minimax_h3_varlen_attention`；`flashinfer-ori: flashinfer/experimental/qwen_3_5_block` | LM head loss/logprob、MOK、per-token NVFP4、projection、MiniMax、Qwen block；Qwen 内另含 vendored Frost attention/GEMM。 |

### 2.6 C++/CUDA、依赖、测试基准与独立子项目

| 位置（证据） | 职责、代表符号或内容 | 应如何作为 R2 基准 |
|---|---|---|
| `flashinfer-ori: csrc` | host glue、TVM FFI binding、`.jinja` 特化模板、CUDA kernel、generated/Cake/NV internal 路线；例如 `batch_prefill.cu`、`flashinfer_norm_binding.cu` | 按 kernel/host plan/FFI/export/template 分层对标；不能把 csrc 条目数当算子数。 |
| `flashinfer-ori: include/flashinfer/attention` | attention CUDA 模板、scheduler、variant；支撑 decode/prefill/cascade/MLA | 自定义 attention 是否能复用调度/归约/布局而只替换 variant。 |
| `flashinfer-ori: include/flashinfer/gemm`；`flashinfer-ori: include/flashinfer/fused_moe`；`flashinfer-ori: include/flashinfer/megamoe` | GEMM、MoE、MegaMoE CUDA/CUTLASS 代码与参数结构 | 量化权重 layout、routing metadata、workspace schema 是否有统一表达。 |
| `flashinfer-ori: include/flashinfer/comm`；`flashinfer-ori: include/flashinfer/trtllm`；`flashinfer-ori: include/flashinfer/mamba` | 通信/外部实现适配/SSM 的 C++ 层 | 依赖适配与设备内核要分开评价。 |
| `flashinfer-ori: include/flashinfer/page.cuh`；`flashinfer-ori: include/flashinfer/pos_enc.cuh`；`flashinfer-ori: include/flashinfer/quantization.cuh`；`flashinfer-ori: include/flashinfer/norm` | KV/RoPE/quant/norm 的可复用 device 实现 | 公共数据布局/线程块操作的复用。 |
| `flashinfer-ori: include/flashinfer/profiler.cuh`；`flashinfer-ori: include/flashinfer/logging.h`；`flashinfer-ori: include/flashinfer/cubin_loader.h`；`flashinfer-ori: include/flashinfer/allocator.h` | profiler、日志、cubin loader、allocator；其余 vector/math/MMA/cp_async/layout 头文件为基础构件 | 调试、加载与内存辅助属于架构能力，不能只统计数学算子。 |
| `flashinfer-ori: .gitmodules`；`flashinfer-ori: 3rdparty/cutlass` | NVIDIA CUTLASS 子模块 | CUDA/CUTLASS 源编译与 CuTe-DSL Python 依赖是不同交付路线。 |
| `flashinfer-ori: 3rdparty/cccl`；`flashinfer-ori: 3rdparty/spdlog`；`flashinfer-ori: 3rdparty/nixl` | CCCL/CUB 通用 CUDA primitives、spdlog 日志、NIXL 通信子模块 | 分别核对所需 header/库和 wheel 是否携带，不把名称直接等同运行期必需依赖。 |
| `flashinfer-ori: tests`；`flashinfer-ori: tests/conftest.py` | 算子正确性、边界/布局、backend parity、JIT、CLI、分发、trace、图模式等测试 | 用现有 test 条件建立 device/dtype/shape/模式矩阵；本轮未执行 GPU 测试。 |
| `flashinfer-ori: benchmarks`；`flashinfer-ori: benchmarks/flashinfer_benchmark.py`；`flashinfer-ori: benchmarks/routines` | 通用 benchmark CLI + operator routines、单算子/融合/模型阶段性能脚本 | 性能基准具备 warmup、backend 比较与 workload 入口；脚本存在不证明测量结果。 |
| `flashinfer-ori: flashinfer-cubin/pyproject.toml`；`flashinfer-ori: flashinfer-cubin/build_backend.py`；`flashinfer-ori: flashinfer-cubin/flashinfer_cubin` | 独立 cubin artifact wheel；`get_cubin_dir` 暴露包内 artifact 根目录 | 外部 kernel 二进制交付，与 Python主包/JIT shared library 包分离；详见 §3.2、§4。 |
| `flashinfer-ori: flashinfer-jit-cache/pyproject.toml`；`flashinfer-ori: flashinfer-jit-cache/build_backend.py`；`flashinfer-ori: flashinfer-jit-cache/flashinfer_jit_cache` | 独立 provider shim；`JitCacheProvider`、`get_jit_cache_providers` 经 entry point 发现已安装 provider | 此基线已经拆成 shim + 按架构的 provider payload，不是传统单体 `jit_cache` wheel。 |
| `flashinfer-ori: flashinfer-jit-cache-provider/pyproject.toml`；`flashinfer-ori: flashinfer-jit-cache-provider/package_config.py`；`flashinfer-ori: flashinfer-jit-cache-provider/setup.py` | 独立 provider 构建项目；`get_provider_build_config` 生成 `flashinfer-jit-cache-<arch>` 分发名并携带 AOT payload | provider 携带实际预编译 module；shim 依赖/发现它们；具体版本/架构分发逻辑见 §4。 |

测试覆盖的直接能力分组为：attention、attn_scores、autotuner、CLI、comm、diffusion_ops、experimental、GDN/GDN2/GDP、GEMM/grouped_mm、JIT、KDA、Mamba、mHC、MLA、model_optimizations、MoE/MoE-EP、MSA、norm、prims_ts、quantization、topk_varlen、trace/trace_apply、utils。
证据是 `flashinfer-ori: tests` 的目录树；另有 `test_artifacts.py`、JIT cache wheel/build、环境/CI/sharding/日志等根级测试，不能用这些测试目录推断全部 case 已通过。
下游风格或对照实现还出现在 `flashinfer-ori: benchmarks/bench_attention_sink_triton_sgl_context.py`、`flashinfer-ori: benchmarks/bench_attention_sink_triton_sgl_decode.py`，使用证据的强弱在 §4 单独说明。

**第三方源码可见性限制：** 当前四个 `3rdparty` 子目录均为空（0 个文件），未初始化其子模块。
上表对 CUTLASS/CCCL/spdlog/NIXL 的来源判断来自 `flashinfer-ori: .gitmodules` 及调用侧构建/include 声明，未审阅这些依赖的实现。
本轮未更新 submodule、未下载 artifact、未编译执行 kernel；声明存在、构建入口存在与运行验证通过必须分开记录。

测试不是附属目录：若 R2 只比较 API 名称，容易漏掉输入契约、资源生命周期和回退行为。
以下静态读到的测试函数给出可迁移的验收维度；本轮未运行，表中不宣称通过。

| 能力契约 | 具体测试文件 | 已读到的测试符号/检查意图 |
|---|---|---|
| sink attention 生命周期 | `flashinfer-ori: tests/attention/test_attention_sink.py` | `test_attention_sink_incremental_generation`、`test_attention_sink_chunk_prefill`、`test_attention_sink_varlen`。 |
| auto 保留语义 | `flashinfer-ori: tests/attention/test_auto_backend_upgrade.py` | `test_auto_declines_when_semantics_would_be_dropped`、`test_explicit_backend_is_respected`、`test_auto_matches_fa2_numerically`。 |
| GEMM 输入/输出契约 | `flashinfer-ori: tests/gemm/test_bmm_bf16.py` | 重复调用复用 module/output；拒绝不支持 K、非转置 B、奇数 N、错位指针和 input/output overlap。 |
| LoRA template 选择 | `flashinfer-ori: tests/gemm/test_cake_quant_lora_gemm.py` | `test_select_template_names_the_violated_requirement`、NVFP4 prepared route 和 launch grid。 |
| grouped_mm 特殊 expert 分布 | `flashinfer-ori: tests/grouped_mm/test_grouped_mm_bf16.py` | empty/single expert、预分配输出、FP32 输出、graph cache reuse、indptr dtype/长度。 |
| MoE LoRA 写入行为 | `flashinfer-ori: tests/moe/test_bgmv_moe_lora_delta.py` | 两阶段 LoRA delta、`test_gemm2_inactive_slot_zeroed`。 |
| EP 配置约束 | `flashinfer-ori: tests/moe_ep/test_constraints.py` | expert/world-size 整除、local expert 数量、NIXL hidden size/token cap、bootstrap world size。 |
| 通信负例 | `flashinfer-ori: tests/comm/test_allreduce_negative.py` | 不支持的量化 pattern/layout、RMSNorm 缺参、buffer 容量边界。 |
| 通信 prepare/backend 路由 | `flashinfer-ori: tests/comm/test_all_gather_matmul_dispatch.py` | exported callable、prepare 参数转发、非法 backend 提前拒绝、auto 按设备保持 route。 |
| linear attention state pool | `flashinfer-ori: tests/gdn/test_cudnn_state_pool_contract.py` | `test_pool_check_does_not_read_indices`、写入前验证 pool、frontend 版本准入。 |
| KDA scratch 并发所有权 | `flashinfer-ori: tests/kda/test_cudnn_linear_attention_workspace.py` | `test_workspace_is_owned_by_stream_or_capture`、`test_independent_calls_do_not_share_concurrent_scratch`。 |
| SSD CUDA Graph 数据更新 | `flashinfer-ori: tests/mamba/test_cake_ssd_combined_capture.py` | prepared graph bitwise replay、拒绝未预热 shape、static input 修改可见、captured workspace 固定、prepared eager 无分配。 |
| 训练 façade 一致性 | `flashinfer-ori: tests/norm/test_cake_rmsnorm_train_api.py` | `test_function_signature_is_exported`、`test_facade_entry_points_match_the_backend`。 |
| 量化 scale 布局 | `flashinfer-ori: tests/quantization/test_per_token_group_quant_8bit_cutile.py` | column-major/TMA-aligned/UE8M0 scale、非 2 次幂 group size。 |
| 调优缓存持久化 | `flashinfer-ori: tests/autotuner/test_autotune_cache_v2.py` | 新进程不重 profile、损坏 entry 退化为 miss、环境隔离、tuple tactic roundtrip、发布失败不破坏 tuning。 |
| AOT 注册覆盖 | `flashinfer-ori: tests/jit/test_cutlass_fused_moe_aot.py` | `test_aot_registers_cutlass_fused_moe_for_sm110`；这是注册检查，不能替代 SM110 实机数值测试。 |
| 外部替换方案契约 | `flashinfer-ori: tests/trace_apply/test_trace_apply.py` | schema roundtrip、stateful plan/run namespace、历史 MLA wrapper key、out/LSE/in-place writeback。 |
| artifact 并发/错误处理 | `flashinfer-ori: tests/test_artifacts.py` | 并发只下载一次、空 cache、索引不完整、manifest 不可达/缓存 fallback、subdir 文件列表。 |

### 2.7 深层目录覆盖清单

以下是静态扫描到的 **429 个非直接子目录**；只表达物理覆盖，不增加未经验证的 API/功能承诺。
每格均为完整仓库相对路径；按顶层能力组排序，每行至多 4 个目录。
`csrc`、`kernels`、`resources`、`artifacts`、`shim`、`sources` 等目录是实现/资源层，公开入口仍以上述能力表为准。

| 所属直接能力组 | 深层目录（实际存在） |
|---|---|
| attention | `flashinfer-ori: flashinfer/attention/cute_dsl`；`flashinfer-ori: flashinfer/attention/kernels`；`flashinfer-ori: flashinfer/attention/kernels/cutile`；`flashinfer-ori: flashinfer/attention/prims_ts` |
| attention | `flashinfer-ori: flashinfer/attention/prims_ts/_block_sparse`；`flashinfer-ori: flashinfer/attention/prims_ts/kernels`；`flashinfer-ori: flashinfer/attention/prims_ts/kernels/fmha_context`；`flashinfer-ori: flashinfer/attention/prims_ts/kernels/fmha_decode` |
| attention | `flashinfer-ori: flashinfer/attention/prims_ts/kernels/fmha_decode/fmha_decode_resources`；`flashinfer-ori: flashinfer/attention/prims_ts/kernels/mla_decode`；`flashinfer-ori: flashinfer/attention/prims_ts/kernels/mla_decode/helpers`；`flashinfer-ori: flashinfer/attention/prims_ts/kernels/mla_decode/throughput_2cta` |
| attention | `flashinfer-ori: flashinfer/attention/prims_ts/kernels/mla_decode/throughput_latency_1cta`；`flashinfer-ori: flashinfer/attention/prims_ts/kernels/mla_decode/throughput_latency_1cta/resources` |
| attn_scores | `flashinfer-ori: flashinfer/attn_scores/kernels` |
| cake_causal_conv1d_train_backend | `flashinfer-ori: flashinfer/cake_causal_conv1d_train_backend/csrc` |
| cake_fused_qk_rope_append_backend | `flashinfer-ori: flashinfer/cake_fused_qk_rope_append_backend/csrc` |
| cake_fused_qk_rope_fp8_append_backend | `flashinfer-ori: flashinfer/cake_fused_qk_rope_fp8_append_backend/csrc` |
| cake_mhc_train_backend | `flashinfer-ori: flashinfer/cake_mhc_train_backend/csrc` |
| cake_nvfp4_svdquant_gemm_backend | `flashinfer-ori: flashinfer/cake_nvfp4_svdquant_gemm_backend/csrc` |
| cake_rmsnorm_train_backend | `flashinfer-ori: flashinfer/cake_rmsnorm_train_backend/csrc` |
| cake_sage_block_sparse_attention_backend | `flashinfer-ori: flashinfer/cake_sage_block_sparse_attention_backend/csrc` |
| cake_ulysses_backend | `flashinfer-ori: flashinfer/cake_ulysses_backend/csrc` |
| cake_vsa_sm90_backend | `flashinfer-ori: flashinfer/cake_vsa_sm90_backend/csrc` |
| comm | `flashinfer-ori: flashinfer/comm/all_gather_matmul`；`flashinfer-ori: flashinfer/comm/fp8_rowwise_gemm_allreduce`；`flashinfer-ori: flashinfer/comm/mnnvl_cutedsl`；`flashinfer-ori: flashinfer/comm/mnnvl_cutedsl/kernel_bt` |
| comm | `flashinfer-ori: flashinfer/comm/mnnvl_cutedsl/kernel_ht`；`flashinfer-ori: flashinfer/comm/mnnvl_cutedsl/kernel_ll`；`flashinfer-ori: flashinfer/comm/pcie_ipc_collectives` |
| cute_dsl | `flashinfer-ori: flashinfer/cute_dsl/attention`；`flashinfer-ori: flashinfer/cute_dsl/attention/dsa`；`flashinfer-ori: flashinfer/cute_dsl/attention/fmha`；`flashinfer-ori: flashinfer/cute_dsl/attention/fmha/helpers` |
| cute_dsl | `flashinfer-ori: flashinfer/cute_dsl/attention/fmha/sm120`；`flashinfer-ori: flashinfer/cute_dsl/attention/fusion`；`flashinfer-ori: flashinfer/cute_dsl/attention/monolithic`；`flashinfer-ori: flashinfer/cute_dsl/attention/roles` |
| cute_dsl | `flashinfer-ori: flashinfer/cute_dsl/attention/scheduler`；`flashinfer-ori: flashinfer/cute_dsl/attention/wrappers`；`flashinfer-ori: flashinfer/cute_dsl/sparse`；`flashinfer-ori: flashinfer/cute_dsl/sparse/bsa_utils` |
| cute_dsl | `flashinfer-ori: flashinfer/cute_dsl/sparse/sm100_blk128`；`flashinfer-ori: flashinfer/cute_dsl/sparse/sm100_blk64`；`flashinfer-ori: flashinfer/cute_dsl/sparse/sm120_blk64` |
| experimental | `flashinfer-ori: flashinfer/experimental/b12x`；`flashinfer-ori: flashinfer/experimental/b12x/_compat`；`flashinfer-ori: flashinfer/experimental/b12x/_compat/b12x`；`flashinfer-ori: flashinfer/experimental/b12x/_lib` |
| experimental | `flashinfer-ori: flashinfer/experimental/b12x/_lib/quant`；`flashinfer-ori: flashinfer/experimental/b12x/attention`；`flashinfer-ori: flashinfer/experimental/b12x/attention/_shared`；`flashinfer-ori: flashinfer/experimental/b12x/attention/_shared/contiguous` |
| experimental | `flashinfer-ori: flashinfer/experimental/b12x/attention/_shared/cute`；`flashinfer-ori: flashinfer/experimental/b12x/attention/_shared/mla`；`flashinfer-ori: flashinfer/experimental/b12x/attention/compressed_sparse_mla`；`flashinfer-ori: flashinfer/experimental/b12x/attention/dense_mla` |
| experimental | `flashinfer-ori: flashinfer/experimental/b12x/attention/dsa_indexer`；`flashinfer-ori: flashinfer/experimental/b12x/attention/mla_compress`；`flashinfer-ori: flashinfer/experimental/b12x/attention/paged`；`flashinfer-ori: flashinfer/experimental/b12x/attention/qsa` |
| experimental | `flashinfer-ori: flashinfer/experimental/b12x/attention/sparse_mla`；`flashinfer-ori: flashinfer/experimental/b12x/attention/varlen`；`flashinfer-ori: flashinfer/experimental/b12x/comm`；`flashinfer-ori: flashinfer/experimental/b12x/comm/pcie` |
| experimental | `flashinfer-ori: flashinfer/experimental/b12x/comm/roce`；`flashinfer-ori: flashinfer/experimental/b12x/gemm`；`flashinfer-ori: flashinfer/experimental/b12x/gemm/_bmm`；`flashinfer-ori: flashinfer/experimental/b12x/gemm/_shared` |
| experimental | `flashinfer-ori: flashinfer/experimental/b12x/gemm/bf16_gemv`；`flashinfer-ori: flashinfer/experimental/b12x/gemm/bf16_vocab_projection`；`flashinfer-ori: flashinfer/experimental/b12x/gemm/block_fp8_linear`；`flashinfer-ori: flashinfer/experimental/b12x/gemm/blockscaled` |
| experimental | `flashinfer-ori: flashinfer/experimental/b12x/gemm/mla_query_projection`；`flashinfer-ori: flashinfer/experimental/b12x/gemm/mxfp8_linear`；`flashinfer-ori: flashinfer/experimental/b12x/gemm/tensor_fp8_linear`；`flashinfer-ori: flashinfer/experimental/b12x/gemm/trellis_linear` |
| experimental | `flashinfer-ori: flashinfer/experimental/b12x/gemm/wo_projection`；`flashinfer-ori: flashinfer/experimental/b12x/loader`；`flashinfer-ori: flashinfer/experimental/b12x/moe`；`flashinfer-ori: flashinfer/experimental/b12x/moe/_shared` |
| experimental | `flashinfer-ori: flashinfer/experimental/b12x/moe/_shared/kernels`；`flashinfer-ori: flashinfer/experimental/b12x/moe/_shared/kernels/w4a16`；`flashinfer-ori: flashinfer/experimental/b12x/moe/_shared/kernels/w4a8`；`flashinfer-ori: flashinfer/experimental/b12x/moe/_shared/kernels/w6a8` |
| experimental | `flashinfer-ori: flashinfer/experimental/b12x/moe/_shared/kernels/w8a8`；`flashinfer-ori: flashinfer/experimental/b12x/moe/checkpoints`；`flashinfer-ori: flashinfer/experimental/b12x/moe/ep_moe`；`flashinfer-ori: flashinfer/experimental/b12x/moe/fused_moe` |
| experimental | `flashinfer-ori: flashinfer/experimental/b12x/norm`；`flashinfer-ori: flashinfer/experimental/b12x/norm/hyperconnection`；`flashinfer-ori: flashinfer/experimental/b12x/norm/mhc`；`flashinfer-ori: flashinfer/experimental/b12x/preparation` |
| experimental | `flashinfer-ori: flashinfer/experimental/b12x/quantization`；`flashinfer-ori: flashinfer/experimental/b12x/quantization/mxfp6`；`flashinfer-ori: flashinfer/experimental/b12x/quantization/mxfp8`；`flashinfer-ori: flashinfer/experimental/b12x/quantization/nvfp4` |
| experimental | `flashinfer-ori: flashinfer/experimental/b12x/sequence`；`flashinfer-ori: flashinfer/experimental/b12x/sequence/_shared`；`flashinfer-ori: flashinfer/experimental/b12x/sequence/_shared/delta_prefill`；`flashinfer-ori: flashinfer/experimental/b12x/sequence/embedding` |
| experimental | `flashinfer-ori: flashinfer/experimental/b12x/sequence/engram`；`flashinfer-ori: flashinfer/experimental/b12x/sequence/gdn_decode`；`flashinfer-ori: flashinfer/experimental/b12x/sequence/gdn_prefill`；`flashinfer-ori: flashinfer/experimental/b12x/sequence/kda_prefill` |
| experimental | `flashinfer-ori: flashinfer/experimental/b12x/sequence/mtp_feedback`；`flashinfer-ori: flashinfer/experimental/b12x/sequence/ple`；`flashinfer-ori: flashinfer/experimental/b12x/sequence/ple_embedding`；`flashinfer-ori: flashinfer/experimental/b12x/sequence/ple_hash` |
| experimental | `flashinfer-ori: flashinfer/experimental/b12x/testing`；`flashinfer-ori: flashinfer/experimental/b12x/testing/reference`；`flashinfer-ori: flashinfer/experimental/b12x/testing/startup`；`flashinfer-ori: flashinfer/experimental/b12x/tools` |
| experimental | `flashinfer-ori: flashinfer/experimental/balanced_gqa_decode`；`flashinfer-ori: flashinfer/experimental/balanced_gqa_decode/csrc`；`flashinfer-ori: flashinfer/experimental/balanced_gqa_decode/csrc/cake_balanced_gqa_decode`；`flashinfer-ori: flashinfer/experimental/cake_dsa_indexer` |
| experimental | `flashinfer-ori: flashinfer/experimental/cake_dsa_indexer/csrc`；`flashinfer-ori: flashinfer/experimental/cake_dsa_indexer/csrc/cake_device_helpers`；`flashinfer-ori: flashinfer/experimental/cake_dsa_indexer/csrc/cake_dsa_indexer_topk`；`flashinfer-ori: flashinfer/experimental/cake_dsa_train` |
| experimental | `flashinfer-ori: flashinfer/experimental/cake_dsa_train/csrc`；`flashinfer-ori: flashinfer/experimental/cake_dsa_train/csrc/cake_dsa_h64_train`；`flashinfer-ori: flashinfer/experimental/cake_kda_train`；`flashinfer-ori: flashinfer/experimental/cake_kda_train/csrc` |
| experimental | `flashinfer-ori: flashinfer/experimental/cake_kimi_k3_attn_res`；`flashinfer-ori: flashinfer/experimental/cake_kimi_k3_attn_res/csrc`；`flashinfer-ori: flashinfer/experimental/cake_lm_head_loss`；`flashinfer-ori: flashinfer/experimental/cake_lm_head_loss/csrc` |
| experimental | `flashinfer-ori: flashinfer/experimental/cake_lm_head_loss/csrc/cake_lm_head_loss`；`flashinfer-ori: flashinfer/experimental/cake_lm_head_loss/csrc/cake_lm_head_loss/shim`；`flashinfer-ori: flashinfer/experimental/cake_mla_nvfp4_paged_decode`；`flashinfer-ori: flashinfer/experimental/cake_mla_nvfp4_paged_decode/csrc` |
| experimental | `flashinfer-ori: flashinfer/experimental/cake_mla_nvfp4_paged_decode/csrc/cake_device_helpers`；`flashinfer-ori: flashinfer/experimental/cake_mla_nvfp4_paged_decode/csrc/cake_mla_nvfp4_paged_decode`；`flashinfer-ori: flashinfer/experimental/cake_mla_varq_dcp_decode`；`flashinfer-ori: flashinfer/experimental/cake_mla_varq_dcp_decode/csrc` |
| experimental | `flashinfer-ori: flashinfer/experimental/cake_mla_varq_dcp_decode/csrc/cake_mla_varq_dcp_decode`；`flashinfer-ori: flashinfer/experimental/cake_moe_grouped_gemm`；`flashinfer-ori: flashinfer/experimental/cake_moe_grouped_gemm/csrc`；`flashinfer-ori: flashinfer/experimental/cake_moe_grouped_gemm/csrc/cake_moe_grouped_gemm` |
| experimental | `flashinfer-ori: flashinfer/experimental/cake_mok_bf16`；`flashinfer-ori: flashinfer/experimental/cake_mok_bf16/csrc`；`flashinfer-ori: flashinfer/experimental/cake_mok_bf16/csrc/sm_100a`；`flashinfer-ori: flashinfer/experimental/cake_mok_bf16/csrc/sm_107a` |
| experimental | `flashinfer-ori: flashinfer/experimental/cake_msa_icp_indexer`；`flashinfer-ori: flashinfer/experimental/cake_mxfp8_megamoe_ep16`；`flashinfer-ori: flashinfer/experimental/cake_mxfp8_megamoe_ep16/csrc`；`flashinfer-ori: flashinfer/experimental/cake_mxfp8_megamoe_ep16/csrc/cake_mxfp8_megamoe_ep16` |
| experimental | `flashinfer-ori: flashinfer/experimental/cake_mxfp8_megamoe_ep16/csrc/cake_mxfp8_megamoe_ep16/cute`；`flashinfer-ori: flashinfer/experimental/cake_nvfp4_per_token`；`flashinfer-ori: flashinfer/experimental/cake_nvfp4_per_token/csrc`；`flashinfer-ori: flashinfer/experimental/cake_nvfp4_per_token/csrc/cake_nvfp4_per_token` |
| experimental | `flashinfer-ori: flashinfer/experimental/cake_stepfun_moe`；`flashinfer-ori: flashinfer/experimental/cake_stepfun_moe/csrc`；`flashinfer-ori: flashinfer/experimental/cake_stepfun_moe/csrc/cake_device_helpers`；`flashinfer-ori: flashinfer/experimental/cake_stepfun_moe/csrc/cake_stepfun_moe` |
| experimental | `flashinfer-ori: flashinfer/experimental/cake_w8a8_megamoe_ep16`；`flashinfer-ori: flashinfer/experimental/cake_w8a8_megamoe_ep16/csrc`；`flashinfer-ori: flashinfer/experimental/cake_w8a8_megamoe_ep16/csrc/cake_w8a8_megamoe_ep16`；`flashinfer-ori: flashinfer/experimental/cudnn_frost_selected_kernels_moe_grouped_gemm` |
| experimental | `flashinfer-ori: flashinfer/experimental/cudnn_frost_selected_kernels_moe_grouped_gemm/bf16`；`flashinfer-ori: flashinfer/experimental/cudnn_frost_selected_kernels_moe_grouped_gemm/mxfp8`；`flashinfer-ori: flashinfer/experimental/cudnn_frost_selected_kernels_moe_grouped_gemm/mxfp8_mxfp4`；`flashinfer-ori: flashinfer/experimental/cudnn_frost_selected_kernels_moe_grouped_gemm/nvfp4` |
| experimental | `flashinfer-ori: flashinfer/experimental/deepgemm_batched_gemm`；`flashinfer-ori: flashinfer/experimental/deepgemm_common`；`flashinfer-ori: flashinfer/experimental/deepgemm_dense_mqa`；`flashinfer-ori: flashinfer/experimental/deepgemm_fp4_gemm` |
| experimental | `flashinfer-ori: flashinfer/experimental/deepgemm_fp8_gemm`；`flashinfer-ori: flashinfer/experimental/deepgemm_kgroup_gemm`；`flashinfer-ori: flashinfer/experimental/deepgemm_mega_gate`；`flashinfer-ori: flashinfer/experimental/deepgemm_mega_mhc` |
| experimental | `flashinfer-ori: flashinfer/experimental/deepgemm_mixed_gemm`；`flashinfer-ori: flashinfer/experimental/deepgemm_sm120_paged_mqa_logits`；`flashinfer-ori: flashinfer/experimental/deepgemm_sparse_mqa`；`flashinfer-ori: flashinfer/experimental/deepseek_v41` |
| experimental | `flashinfer-ori: flashinfer/experimental/dense_projection_gemm`；`flashinfer-ori: flashinfer/experimental/dense_projection_gemm/csrc`；`flashinfer-ori: flashinfer/experimental/dense_projection_gemm/csrc/cake_dense_projection_gemm`；`flashinfer-ori: flashinfer/experimental/dense_projection_gemm/csrc/cake_device_helpers` |
| experimental | `flashinfer-ori: flashinfer/experimental/kda_prefill_wrapper`；`flashinfer-ori: flashinfer/experimental/kimi_k3_fp8_projection`；`flashinfer-ori: flashinfer/experimental/kimi_k3_fp8_projection/csrc`；`flashinfer-ori: flashinfer/experimental/kimi_k3_fp8_projection/csrc/cake_device_helpers` |
| experimental | `flashinfer-ori: flashinfer/experimental/kimi_k3_fp8_projection/csrc/cake_kimi_k3_fp8_projection`；`flashinfer-ori: flashinfer/experimental/kimi_k3_fused_router`；`flashinfer-ori: flashinfer/experimental/kimi_k3_fused_router/csrc`；`flashinfer-ori: flashinfer/experimental/kimi_k3_fused_router/csrc/cake_kimi_k3_fused_router` |
| experimental | `flashinfer-ori: flashinfer/experimental/kimi_k3_latent_moe`；`flashinfer-ori: flashinfer/experimental/kimi_k3_latent_moe/csrc`；`flashinfer-ori: flashinfer/experimental/kimi_k3_latent_moe/csrc/cake_kimi_k3_latent_moe`；`flashinfer-ori: flashinfer/experimental/kimi_k3_tp12_tail` |
| experimental | `flashinfer-ori: flashinfer/experimental/kimi_k3_tp12_tail/csrc`；`flashinfer-ori: flashinfer/experimental/kimi_k3_tp12_tail/csrc/cake_kimi_k3_tp12_tail`；`flashinfer-ori: flashinfer/experimental/kimi_k3_vision_tower`；`flashinfer-ori: flashinfer/experimental/kimi_k3_vision_tower/csrc` |
| experimental | `flashinfer-ori: flashinfer/experimental/kimi_k3_vision_tower/csrc/cake_kimi_k3_vision_tower`；`flashinfer-ori: flashinfer/experimental/mega_moe_v3`；`flashinfer-ori: flashinfer/experimental/minimax_h3_varlen_attention`；`flashinfer-ori: flashinfer/experimental/minimax_h3_varlen_attention/csrc` |
| experimental | `flashinfer-ori: flashinfer/experimental/minimax_h3_varlen_attention/csrc/cake_device_helpers`；`flashinfer-ori: flashinfer/experimental/minimax_h3_varlen_attention/csrc/cake_minimax_h3_varlen_attention`；`flashinfer-ori: flashinfer/experimental/msa_nvfp4_decode`；`flashinfer-ori: flashinfer/experimental/msa_nvfp4_decode/csrc` |
| experimental | `flashinfer-ori: flashinfer/experimental/msa_nvfp4_decode/csrc/cake_msa_nvfp4_decode`；`flashinfer-ori: flashinfer/experimental/nvfp4_attention`；`flashinfer-ori: flashinfer/experimental/nvfp4_attention/csrc`；`flashinfer-ori: flashinfer/experimental/nvfp4_mla_decode` |
| experimental | `flashinfer-ori: flashinfer/experimental/nvfp4_mla_decode/csrc`；`flashinfer-ori: flashinfer/experimental/nvfp4_mla_decode/csrc/cake_nvfp4_mla_decode`；`flashinfer-ori: flashinfer/experimental/nvfp4_sparse_mla_decode`；`flashinfer-ori: flashinfer/experimental/nvfp4_sparse_mla_decode/csrc` |
| experimental | `flashinfer-ori: flashinfer/experimental/nvfp4_sparse_mla_decode/csrc/cake_nvfp4_sparse_mla_decode`；`flashinfer-ori: flashinfer/experimental/qwen_3_5_block`；`flashinfer-ori: flashinfer/experimental/qwen_3_5_block/_cudnn_frost`；`flashinfer-ori: flashinfer/experimental/qwen_3_5_block/_cudnn_frost/engines` |
| experimental | `flashinfer-ori: flashinfer/experimental/qwen_3_5_block/_cudnn_frost/frost`；`flashinfer-ori: flashinfer/experimental/qwen_3_5_block/_cudnn_frost/frost/tile_dsl`；`flashinfer-ori: flashinfer/experimental/qwen_3_5_block/_cudnn_frost/gated_attention_block`；`flashinfer-ori: flashinfer/experimental/qwen_3_5_block/_cudnn_frost/gated_attention_block/kernels` |
| experimental | `flashinfer-ori: flashinfer/experimental/qwen_3_5_block/_cudnn_frost/gemm`；`flashinfer-ori: flashinfer/experimental/qwen_3_5_block/_cudnn_frost/gemm/frost`；`flashinfer-ori: flashinfer/experimental/qwen_3_5_block/_cudnn_frost/gemm/frost/kernel_templates`；`flashinfer-ori: flashinfer/experimental/qwen_3_5_block/_cudnn_frost/gemm/frost/sm100` |
| experimental | `flashinfer-ori: flashinfer/experimental/qwen_3_5_block/_cudnn_frost/gemm/frost/sm100/kernel_templates`；`flashinfer-ori: flashinfer/experimental/qwen_3_5_block/_cudnn_frost/sdpa`；`flashinfer-ori: flashinfer/experimental/qwen_3_5_block/_cudnn_frost/sdpa/bwd`；`flashinfer-ori: flashinfer/experimental/qwen_3_5_block/_cudnn_frost/sdpa/fwd` |
| experimental | `flashinfer-ori: flashinfer/experimental/qwen_3_5_block/_cudnn_frost/sdpa/fwd/kernels`；`flashinfer-ori: flashinfer/experimental/qwen_3_5_block/_cudnn_frost/sdpa/fwd/kernels/sm100`；`flashinfer-ori: flashinfer/experimental/qwen_3_5_block/_cudnn_frost/sdpa/fwd/kernels/sm107`；`flashinfer-ori: flashinfer/experimental/qwen_3_5_block/_vendoring` |
| experimental | `flashinfer-ori: flashinfer/experimental/sm110_gqa_decode`；`flashinfer-ori: flashinfer/experimental/sm110_gqa_decode/csrc`；`flashinfer-ori: flashinfer/experimental/sm110_gqa_decode/csrc/prepared`；`flashinfer-ori: flashinfer/experimental/sm110_gqa_decode/csrc/sm110_gqa_decode` |
| experimental | `flashinfer-ori: flashinfer/experimental/sm110_gqa_decode/csrc/sm110_gqa_decode/sm_110a`；`flashinfer-ori: flashinfer/experimental/sm110_xqa`；`flashinfer-ori: flashinfer/experimental/sm110_xqa/csrc`；`flashinfer-ori: flashinfer/experimental/sm110_xqa/csrc/sm110_xqa` |
| experimental | `flashinfer-ori: flashinfer/experimental/sm110_xqa/csrc/sm110_xqa/sm_110a` |
| fused_moe | `flashinfer-ori: flashinfer/fused_moe/backends`；`flashinfer-ori: flashinfer/fused_moe/backends/cudnn_frost`；`flashinfer-ori: flashinfer/fused_moe/backends/cudnn_frost/artifacts`；`flashinfer-ori: flashinfer/fused_moe/backends/cudnn_frost/artifacts/bf16` |
| fused_moe | `flashinfer-ori: flashinfer/fused_moe/backends/cudnn_frost/artifacts/bf16/sources`；`flashinfer-ori: flashinfer/fused_moe/backends/cudnn_frost/artifacts/mxfp8`；`flashinfer-ori: flashinfer/fused_moe/backends/cudnn_frost/artifacts/mxfp8/sources`；`flashinfer-ori: flashinfer/fused_moe/backends/cudnn_frost/artifacts/mxfp8_mxfp4` |
| fused_moe | `flashinfer-ori: flashinfer/fused_moe/backends/cudnn_frost/artifacts/mxfp8_mxfp4/sources`；`flashinfer-ori: flashinfer/fused_moe/backends/cudnn_frost/artifacts/nvfp4`；`flashinfer-ori: flashinfer/fused_moe/backends/cudnn_frost/artifacts/nvfp4/sources`；`flashinfer-ori: flashinfer/fused_moe/backends/cudnn_frost/bf16` |
| fused_moe | `flashinfer-ori: flashinfer/fused_moe/backends/cudnn_frost/csrc`；`flashinfer-ori: flashinfer/fused_moe/backends/cudnn_frost/mxfp8`；`flashinfer-ori: flashinfer/fused_moe/backends/cudnn_frost/mxfp8_mxfp4`；`flashinfer-ori: flashinfer/fused_moe/backends/cudnn_frost/nvfp4` |
| fused_moe | `flashinfer-ori: flashinfer/fused_moe/backends/prims_ts`；`flashinfer-ori: flashinfer/fused_moe/backends/trtllm`；`flashinfer-ori: flashinfer/fused_moe/cake`；`flashinfer-ori: flashinfer/fused_moe/cake_cute` |
| fused_moe | `flashinfer-ori: flashinfer/fused_moe/cute_dsl`；`flashinfer-ori: flashinfer/fused_moe/cute_dsl/blackwell`；`flashinfer-ori: flashinfer/fused_moe/cute_dsl/blackwell_sm12x`；`flashinfer-ori: flashinfer/fused_moe/cute_dsl/blackwell_sm12x/_moe_dynamic` |
| fused_moe | `flashinfer-ori: flashinfer/fused_moe/cute_dsl/blackwell_sm12x/_moe_utils`；`flashinfer-ori: flashinfer/fused_moe/cute_dsl/common`；`flashinfer-ori: flashinfer/fused_moe/cute_dsl/hopper`；`flashinfer-ori: flashinfer/fused_moe/cute_dsl/rubin` |
| fused_moe | `flashinfer-ori: flashinfer/fused_moe/cutile`；`flashinfer-ori: flashinfer/fused_moe/shared` |
| gdn_kernels | `flashinfer-ori: flashinfer/gdn_kernels/blackwell`；`flashinfer-ori: flashinfer/gdn_kernels/delta_rule_dsl`；`flashinfer-ori: flashinfer/gdn_kernels/experimental`；`flashinfer-ori: flashinfer/gdn_kernels/experimental/kernel` |
| gemm | `flashinfer-ori: flashinfer/gemm/kernels`；`flashinfer-ori: flashinfer/gemm/kernels/cute_dsl`；`flashinfer-ori: flashinfer/gemm/kernels/cutile`；`flashinfer-ori: flashinfer/gemm/kernels/native_bf16_fp4` |
| gemm | `flashinfer-ori: flashinfer/gemm/kernels/sm12x_cute` |
| grouped_mm | `flashinfer-ori: flashinfer/grouped_mm/cudnn`；`flashinfer-ori: flashinfer/grouped_mm/cute_sm120_fp8_groupwise`；`flashinfer-ori: flashinfer/grouped_mm/cute_sm120_mxfp8_groupwise` |
| jit | `flashinfer-ori: flashinfer/jit/attention`；`flashinfer-ori: flashinfer/jit/attention/fmha_v2`；`flashinfer-ori: flashinfer/jit/gemm`；`flashinfer-ori: flashinfer/jit/gemm/cutlass` |
| jit | `flashinfer-ori: flashinfer/jit/mamba` |
| kda_kernels | `flashinfer-ori: flashinfer/kda_kernels/persistent`；`flashinfer-ori: flashinfer/kda_kernels/ptx`；`flashinfer-ori: flashinfer/kda_kernels/sm120_prefill`；`flashinfer-ori: flashinfer/kda_kernels/tirx` |
| mla | `flashinfer-ori: flashinfer/mla/_batch_mla`；`flashinfer-ori: flashinfer/mla/_batch_mla/_backends`；`flashinfer-ori: flashinfer/mla/_sparse_mla_sm120` |
| moe_ep | `flashinfer-ori: flashinfer/moe_ep/backends`；`flashinfer-ori: flashinfer/moe_ep/backends/mega`；`flashinfer-ori: flashinfer/moe_ep/backends/mega/kernel`；`flashinfer-ori: flashinfer/moe_ep/backends/mega/kernel/sm100` |
| moe_ep | `flashinfer-ori: flashinfer/moe_ep/backends/mega/kernel/sm100/bf16_bf16_bf16_cutedsl`；`flashinfer-ori: flashinfer/moe_ep/backends/mega/kernel/sm100/bf16_bf16_bf16_rank_major_cuda`；`flashinfer-ori: flashinfer/moe_ep/backends/mega/kernel/sm100/bf16_mxfp8_bf16_cutedsl`；`flashinfer-ori: flashinfer/moe_ep/backends/mega/kernel/sm100/bf16_nvfp4_bf16_cutedsl` |
| moe_ep | `flashinfer-ori: flashinfer/moe_ep/backends/mega/kernel/sm100/common`；`flashinfer-ori: flashinfer/moe_ep/backends/mega/kernel/sm100/fp8_fp4_bf16_deepgemm`；`flashinfer-ori: flashinfer/moe_ep/backends/mega/kernel/sm100/mxfp8_mxfp8_bf16_cutedsl`；`flashinfer-ori: flashinfer/moe_ep/backends/mega/kernel/sm100/nvfp4_nvfp4_bf16_cutedsl` |
| moe_ep | `flashinfer-ori: flashinfer/moe_ep/backends/mega/kernel/sm103`；`flashinfer-ori: flashinfer/moe_ep/backends/mega/kernel/sm107`；`flashinfer-ori: flashinfer/moe_ep/backends/mega/kernel/sm107/mxfp8_mxfp4_bf16_cutedsl`；`flashinfer-ori: flashinfer/moe_ep/backends/mega/kernel/sm107/mxfp8_mxfp8_bf16_cutedsl` |
| moe_ep | `flashinfer-ori: flashinfer/moe_ep/backends/mega/kernel/sm107/nvfp4_nvfp4_bf16_cutedsl`；`flashinfer-ori: flashinfer/moe_ep/backends/mega/kernel/sm120`；`flashinfer-ori: flashinfer/moe_ep/backends/mega/kernel/sm120/mxfp4_mxfp8_bf16_cake`；`flashinfer-ori: flashinfer/moe_ep/backends/mega/kernel/sm120/mxfp4_mxfp8_bf16_cutedsl` |
| moe_ep | `flashinfer-ori: flashinfer/moe_ep/backends/mega/kernel/sm120/mxfp8_mxfp8_bf16_cutedsl`；`flashinfer-ori: flashinfer/moe_ep/backends/mega/kernel/sm90`；`flashinfer-ori: flashinfer/moe_ep/backends/mega/kernel/sm90/bf16_bf16_bf16_pull_cutedsl`；`flashinfer-ori: flashinfer/moe_ep/backends/mega/kernel/sm90/bf16_bf16_bf16_push_cake` |
| moe_ep | `flashinfer-ori: flashinfer/moe_ep/backends/mega/kernel/sm90/fp8_fp8_bf16_pull_cutedsl`；`flashinfer-ori: flashinfer/moe_ep/backends/mega/kernel/sm90/fp8_fp8_bf16_push_cuda`；`flashinfer-ori: flashinfer/moe_ep/backends/mega/kernel/sm90/fp8_mxfp4_bf16_pull_cutedsl`；`flashinfer-ori: flashinfer/moe_ep/backends/split` |
| moe_ep | `flashinfer-ori: flashinfer/moe_ep/backends/split/comm`；`flashinfer-ori: flashinfer/moe_ep/backends/split/comm/cake`；`flashinfer-ori: flashinfer/moe_ep/backends/split/comm/nccl_ep`；`flashinfer-ori: flashinfer/moe_ep/backends/split/comm/nixl_ep` |
| moe_ep | `flashinfer-ori: flashinfer/moe_ep/backends/split/comm/nvlink_one_sided`；`flashinfer-ori: flashinfer/moe_ep/backends/split/comm/nvlink_two_sided`；`flashinfer-ori: flashinfer/moe_ep/backends/split/kernel`；`flashinfer-ori: flashinfer/moe_ep/backends/split/kernel/fused_moe` |
| moe_ep | `flashinfer-ori: flashinfer/moe_ep/backends/split/kernel/identity`；`flashinfer-ori: flashinfer/moe_ep/core`；`flashinfer-ori: flashinfer/moe_ep/core/comm`；`flashinfer-ori: flashinfer/moe_ep/core/kernel` |
| moe_ep | `flashinfer-ori: flashinfer/moe_ep/core/runtime`；`flashinfer-ori: flashinfer/moe_ep/core/validation`；`flashinfer-ori: flashinfer/moe_ep/cute_dsl`；`flashinfer-ori: flashinfer/moe_ep/cute_dsl/megamoe` |
| moe_ep | `flashinfer-ori: flashinfer/moe_ep/cute_dsl/megamoe/bf16_nvfp4`；`flashinfer-ori: flashinfer/moe_ep/kernel_src`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/blackwell_bf16_rank_major`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/blackwell_bf16_rank_major/src` |
| moe_ep | `flashinfer-ori: flashinfer/moe_ep/kernel_src/sm100`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm100/cutedsl_megamoe`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm100/cutedsl_megamoe/shim`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm100/cutedsl_megamoe/src` |
| moe_ep | `flashinfer-ori: flashinfer/moe_ep/kernel_src/sm100/cutedsl_megamoe/src/common`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm100/cutedsl_megamoe/src/moe_bf16_glu`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm100/cutedsl_megamoe/src/moe_mxfp8_bf16_glu`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm100/cutedsl_megamoe/src/moe_mxfp8_glu` |
| moe_ep | `flashinfer-ori: flashinfer/moe_ep/kernel_src/sm100/cutedsl_megamoe/src/moe_nvfp4_swapab`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm100/cutedsl_megamoe/src/src`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm100/next_cutedsl_megamoe`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm100/next_cutedsl_megamoe/shim` |
| moe_ep | `flashinfer-ori: flashinfer/moe_ep/kernel_src/sm100/next_cutedsl_megamoe/src`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm100/next_cutedsl_megamoe/src/blackwell_genphase_sources`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm100/next_cutedsl_megamoe/src/blackwell_genphase_sources/communication`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm100/next_cutedsl_megamoe/src/blackwell_genphase_sources/communication/nvlink_domain` |
| moe_ep | `flashinfer-ori: flashinfer/moe_ep/kernel_src/sm100/next_cutedsl_megamoe/src/blackwell_genphase_sources/helpers`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm100/next_cutedsl_megamoe/src/blackwell_genphase_sources/kernel_src`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm100/next_cutedsl_megamoe/src/blackwell_genphase_sources/kernel_src/blackwell`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm100/next_cutedsl_megamoe/src/blackwell_genphase_sources/kernel_src/blackwell/inference` |
| moe_ep | `flashinfer-ori: flashinfer/moe_ep/kernel_src/sm100/next_cutedsl_megamoe/src/blackwell_genphase_sources/kernel_src/blackwell/inference/mega`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm100/next_cutedsl_megamoe/src/blackwell_genphase_sources/kernel_src/schedulers`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm107`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm107/next_cutedsl_megamoe` |
| moe_ep | `flashinfer-ori: flashinfer/moe_ep/kernel_src/sm107/next_cutedsl_megamoe/shim`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm107/next_cutedsl_megamoe/src`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm107/next_cutedsl_megamoe/src/sources`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm107/next_cutedsl_megamoe/src/sources/communication` |
| moe_ep | `flashinfer-ori: flashinfer/moe_ep/kernel_src/sm107/next_cutedsl_megamoe/src/sources/communication/nvlink_domain`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm107/next_cutedsl_megamoe/src/sources/helpers`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm107/next_cutedsl_megamoe/src/sources/kernel_src`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm107/next_cutedsl_megamoe/src/sources/kernel_src/rubin` |
| moe_ep | `flashinfer-ori: flashinfer/moe_ep/kernel_src/sm107/next_cutedsl_megamoe/src/sources/kernel_src/rubin/inference`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm107/next_cutedsl_megamoe/src/sources/kernel_src/rubin/inference/mega`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm107/next_cutedsl_megamoe/src/sources/kernel_src/schedulers`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm120` |
| moe_ep | `flashinfer-ori: flashinfer/moe_ep/kernel_src/sm120/cake_mxfp4_megamoe`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm120/cake_mxfp4_megamoe/shim`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm120/cake_mxfp4_megamoe/src`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm120/split_cutedsl_megakernel` |
| moe_ep | `flashinfer-ori: flashinfer/moe_ep/kernel_src/sm120/split_cutedsl_megakernel/shim`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm120/split_cutedsl_megakernel/src`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm120/split_cutedsl_megakernel/src/common`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm120/split_cutedsl_megakernel/src/moe_sm120_mxfp4mxfp8_split` |
| moe_ep | `flashinfer-ori: flashinfer/moe_ep/kernel_src/sm120/split_cutedsl_megakernel/src/moe_sm120_mxfp4mxfp8_split/runtime`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm120/split_cutedsl_megakernel/src/src`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm120/swapab_cutedsl_megakernel`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm120/swapab_cutedsl_megakernel/shim` |
| moe_ep | `flashinfer-ori: flashinfer/moe_ep/kernel_src/sm120/swapab_cutedsl_megakernel/src`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm120/swapab_cutedsl_megakernel/src/common`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm120/swapab_cutedsl_megakernel/src/moe_mxfp8_glu`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm120/swapab_cutedsl_megakernel/src/moe_nvfp4_swapab` |
| moe_ep | `flashinfer-ori: flashinfer/moe_ep/kernel_src/sm120/swapab_cutedsl_megakernel/src/moe_sm120_mxfp8_swapab`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm120/swapab_cutedsl_megakernel/src/src`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm90`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm90/cake_bf16_megamoe` |
| moe_ep | `flashinfer-ori: flashinfer/moe_ep/kernel_src/sm90/cake_bf16_megamoe/shim`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm90/cake_bf16_megamoe/src`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm90/pull_style_cutedsl_megakernel`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm90/pull_style_cutedsl_megakernel/shim` |
| moe_ep | `flashinfer-ori: flashinfer/moe_ep/kernel_src/sm90/pull_style_cutedsl_megakernel/src`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm90/pull_style_cutedsl_megakernel/src/common`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm90/pull_style_cutedsl_megakernel/src/moe_hopper_bf16`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm90/pull_style_cutedsl_megakernel/src/moe_hopper_fp8` |
| moe_ep | `flashinfer-ori: flashinfer/moe_ep/kernel_src/sm90/pull_style_cutedsl_megakernel/src/moe_nvfp4_swapab`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm90/pull_style_cutedsl_megakernel/src/src`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm90/push_style_megamoe`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm90/push_style_megamoe/shim` |
| moe_ep | `flashinfer-ori: flashinfer/moe_ep/kernel_src/sm90/push_style_megamoe/src`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm90/push_style_megamoe/src/a2a`；`flashinfer-ori: flashinfer/moe_ep/kernel_src/sm90/push_style_megamoe/src/fp8_gemm`；`flashinfer-ori: flashinfer/moe_ep/modes` |
| msa_ops | `flashinfer-ori: flashinfer/msa_ops/cute_dsl` |
| norm | `flashinfer-ori: flashinfer/norm/kernels` |
| prims_ts | `flashinfer-ori: flashinfer/prims_ts/batched_gemm`；`flashinfer-ori: flashinfer/prims_ts/batched_gemm/tools`；`flashinfer-ori: flashinfer/prims_ts/gemm`；`flashinfer-ori: flashinfer/prims_ts/moe` |
| quantization | `flashinfer-ori: flashinfer/quantization/kernels`；`flashinfer-ori: flashinfer/quantization/kernels/cutile` |
| topk_varlen | `flashinfer-ori: flashinfer/topk_varlen/kernels` |
| trace | `flashinfer-ori: flashinfer/trace/templates` |
| trace_apply | `flashinfer-ori: flashinfer/trace_apply/loaders` |
| triton | `flashinfer-ori: flashinfer/triton/kernels` |

## 3. 协同机制

这里把“同一能力有多个 kernel”与“这些 kernel 如何成为可交付 API”分开。
主要链路是 Python 参数校验/布局适配 → 特化与后端选择 → module 准备 → plan → run；
JIT/AOT 负责准备可调用模块，autotune 负责选 tactic，CUDA Graph 负责重放已经准备好的执行路径。
以下结论以具体实现为准，不能把一种后端的缓存、注册或图支持推广到所有能力域。

### 3.1 JIT 子系统

#### 3.1.1 核心对象与责任边界

| 角色 | 关键符号与实际职责 | 源码证据 |
|---|---|---|
| 公共生命周期 | `JitSpec` 抽象 `lock_path/is_compiled/get_library_path/try_load/build/load`；`build_and_load()` 实现快路径、文件锁、锁内复查、禁用 JIT 检查 | `flashinfer-ori: flashinfer/jit/core.py` |
| NVCC 编译单元 | `JitSpecNvcc` 保存 name、sources、C++/CUDA/link/include flags、device linking、adapter、embedded cubin factory、逐源 CUDA flags | `flashinfer-ori: flashinfer/jit/core.py` |
| 构建规格工厂 | `gen_jit_spec()` 校验架构，补充语言标准、数据类型宏、debug/优化/fast-math 等 flags，构造并注册 `JitSpecNvcc` | `flashinfer-ori: flashinfer/jit/core.py` |
| 注册表 | `JitSpecRegistry` 以 name 保存首次注册的 spec 与创建时间，提供状态/统计；不是 PyTorch dispatcher，也不是覆盖全仓的 kernel 注册表 | `flashinfer-ori: flashinfer/jit/core.py` |
| 编译规则 | `generate_ninja_build_for_op()` 生成 host/CUDA compile、链接及可选 embed/device-link 规则；`run_ninja()` 执行子进程 | `flashinfer-ori: flashinfer/jit/cpp_ext.py` |
| 路径与 AOT 发现 | `FLASHINFER_*` 路径常量、`AOTProvider/AOTArtifact`、`get_aot_artifacts()` | `flashinfer-ori: flashinfer/jit/env.py` |
| 架构上下文 | `CompilationContext` 从环境或可见 GPU 获取目标集合，正规化架构后缀并输出 `-gencode` | `flashinfer-ori: flashinfer/compilation_context.py` |
| 模板生成 | `gen_*_module()`、`gen_customize_*_module()`、`get_*_uri()` 将 API 特化信息映射到代码与 name | `flashinfer-ori: flashinfer/jit/attention/modules.py` |
| 文件增量写入 | `write_if_different()` 内容相同则不重写，保留 Ninja 可用的文件时间信息 | `flashinfer-ori: flashinfer/jit/utils.py` |
| CuTe-DSL 规格 | `JitSpecCuteDsl` 继承共同生命周期，单实例表示一个 kernel 特化，产物为导出的对象文件 | `flashinfer-ori: flashinfer/jit/cute_dsl_core.py` |
| cubin 连接 | `get_artifact()` 下载/校验字节；`setup_cubin_loader()` 给已加载动态库装入回调 | `flashinfer-ori: flashinfer/jit/cubin_loader.py` |

注意：架构上下文文件在 `flashinfer-ori: flashinfer/compilation_context.py`，不是 JIT 子目录中的同名文件。
这里的 `JitSpec` 已是抽象基类，不能把它等同于历史上仅包含 NVCC 参数的数据结构。
`build_jit_specs()` 当前显式只接受 `JitSpecNvcc`，并非所有 JIT 技术的统一批编译入口。
证据：`flashinfer-ori: flashinfer/jit/core.py` 的类型检查。

#### 3.1.2 一条可逐层追踪的 batch decode 调用链

1. Python `get_batch_decode_module(*args)` 有 `functools.cache`，先取得 URI，再调用 generator。
   证据：`flashinfer-ori: flashinfer/decode.py`，`get_batch_decode_module`。
2. `gen_batch_decode_module()` 组织标准 attention variant 和额外参数，进入 `gen_customize_batch_decode_module()`。
   证据：`flashinfer-ori: flashinfer/jit/attention/modules.py`。
3. generator 用 `dtype_map`/`dtype_map_kv` 将 Torch dtype 转成 C++ 类型，并将 head dimension、位置编码、window/cap 等写入模板参数。
   证据：`flashinfer-ori: flashinfer/jit/utils.py`；`flashinfer-ori: flashinfer/jit/attention/modules.py`。
4. 读取 config 模板与实例化模板，执行 `jinja2.Template(...).render(**kwargs)`。
   真实模板：`flashinfer-ori: csrc/batch_decode_customize_config.jinja`；`flashinfer-ori: csrc/batch_decode_kernel_inst.jinja`。
5. 将渲染的配置、kernel 实例化代码和静态 launcher/binding 副本写入生成目录，交给 `gen_jit_spec(uri, source_paths, ...)`。
   静态源文件：`flashinfer-ori: csrc/batch_decode.cu`；`flashinfer-ori: csrc/batch_decode_jit_binding.cu`。
6. `JitSpec.build_and_load()` 先 `try_load()`；NVCC 分支尝试 AOT，失败后加 `FileLock`，再次尝试，再 `build()` 和 `load()`。
   证据：`flashinfer-ori: flashinfer/jit/core.py`，`JitSpec.build_and_load/JitSpecNvcc.try_load`。
7. `build()` 写 Ninja；host compiler/NVCC 产生对象文件，链接成动态库；`load()` 用 `tvm_ffi.load_module()` 加载。
   证据：`flashinfer-ori: flashinfer/jit/core.py`；`flashinfer-ori: flashinfer/jit/cpp_ext.py`。
8. C++ 的 `TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan, ...)`、`workspace_size`、`run` 决定 Python module 的属性名。
   证据：`flashinfer-ori: csrc/batch_decode_jit_binding.cu`。
9. Python 取出 `mod.plan/mod.workspace_size/mod.run`，包装成 `SimpleNamespace`；run 包装还转换 RoPE reciprocal 参数。
   证据：`flashinfer-ori: flashinfer/decode.py`，`get_batch_decode_module`。
10. wrapper 的 `plan()` 保存模块和 plan 信息，后续 `run()` 将 workspace、plan vector、Tensor、标量交给上述 FFI 函数。
    此处装饰器是否真的注册 Torch op，必须看 §3.3；模块可调用本身来自 TVM-FFI。

```text
get_batch_decode_module
  → gen_batch_decode_module → gen_customize_batch_decode_module
  → Jinja render + write_if_different → gen_jit_spec
  → JitSpecNvcc.build_and_load
      → AOT .so，或 Ninja/NVCC 编译 → .so
  → tvm_ffi.load_module
  → C++ typed export: plan / workspace_size / run
  → Python SimpleNamespace → wrapper.plan / wrapper.run
```

#### 3.1.3 缓存 key：三种身份不可混为一个哈希

| 路线 | 身份组成与失效机制 | 证据 |
|---|---|---|
| 常规 attention NVCC | URI 明文编码 op family、Q/KV/O dtype、index dtype、QK/VO head dim、position mode、sliding-window/cap 开关；prefill 另含 backend、reduction 等，具体以各 `get_*_uri()` 为准 | `flashinfer-ori: flashinfer/jit/attention/modules.py` |
| 外层 workspace | FlashInfer version + 排序后的目标架构集合；`FLASHINFER_WORKSPACE_BASE` 改变根目录 | `flashinfer-ori: flashinfer/jit/env.py`，`_get_workspace_dir_name` |
| NVCC 增量有效性 | URI 不是源内容哈希；已有本地 `.so` 仍通过 Ninja 依赖/命令检查，不能仅凭文件存在判断最新；header depfile 与构建命令参与增量重建 | `flashinfer-ori: flashinfer/jit/core.py`，`JitSpecNvcc.try_load/build`；`flashinfer-ori: flashinfer/jit/cpp_ext.py` |
| CuTe-DSL | module 名+编译架构+kernel 特化名；meta 比较 arch、所有已安装 `nvidia-cutlass-dsl*` distribution 名称/版本、指定 key files 的路径和内容 SHA-256 | `flashinfer-ori: flashinfer/jit/cute_dsl_core.py`，`_get_cute_dsl_version/_hash_source_files/JitSpecCuteDsl` |
| Cake 生成程序 | program+精确 target+20 位 SHA-256 前缀；摘要覆盖 device/binding 两源字节与 target/options JSON | `flashinfer-ori: flashinfer/jit/cake_program.py`，`content_digest/CakeProgram` |
| FlashKDA NVRTC | source hash、source name、module/kernel、target/arch、compile options、schema 等组成 JSON 身份；receipt 再核验 cubin hash/size | `flashinfer-ori: flashinfer/jit/flash_kda_nvrtc.py`，`prepare_generated_flash_kda_cubin` |

常规 NVCC 的目录 key 没有统一嵌入 compiler version、所有依赖内容或每个编译 flag 的哈希。
flags 的变化由重新生成的 Ninja 命令处理；其他生成器可以自行把内容摘要放进 name。
因此 R2 应分别对标“特化身份”“增量有效性”“环境隔离”，不能只问有没有 cache。
上述判断来自 `flashinfer-ori: flashinfer/jit/core.py` 和 `flashinfer-ori: flashinfer/jit/cpp_ext.py` 的实际构建路径。

下面是运行期生成目录的**模式示意**，不是本次检出的源码文件列表：

```text
<base>/.cache/flashinfer/<version>/<sorted_archs>/
  generated/<uri>/              模板展开源码、配置与复制的 binding
  cached_ops/<uri>/             build.ninja、对象文件、<uri>.so
  cached_ops/tmp/               模块锁、批量 prebuild Ninja 与全局锁
  cached_ops/<module_arch_cute_dsl>/
    <kernel>.o                  CuTe-DSL export_to_c 产物
    meta.json                   架构、DSL版本、源摘要提交标记
  flashinfer_jit.log
<base>/.cache/flashinfer/cubins/  cubin 默认缓存，另有 wheel/env 优先级
```

目录模式的源码定义：`flashinfer-ori: flashinfer/jit/env.py`、`flashinfer-ori: flashinfer/jit/core.py`、`flashinfer-ori: flashinfer/jit/cute_dsl_core.py`。
主 wheel 内源码数据路径也是安装后布局；不能要求这些生成目录在只读源码 checkout 已存在。

#### 3.1.4 并行、去重与错误边界

| 层次 | 实际机制 | 关键符号/文件 |
|---|---|---|
| 同一进程 | 热 getter 使用 `functools.cache` 缓存模块；其结果与 spec registry 是不同对象 | `get_batch_decode_module`，`flashinfer-ori: flashinfer/decode.py` |
| 跨进程同 module | `build_and_load()` 使用模块文件锁，锁内再次探测；NVCC 本地 cache 通过 Ninja no-op 实现重用 | `JitSpec.build_and_load`，`flashinfer-ori: flashinfer/jit/core.py` |
| 编译文件并行 | Ninja 根据 dependency graph 并行；`MAX_JOBS` 传为 `-j`；每个 module 独立工作目录减少日志竞争 | `run_ninja/_get_num_workers`，`flashinfer-ori: flashinfer/jit/cpp_ext.py` |
| NVCC 内部并行 | `FLASHINFER_NVCC_THREADS` 转成 `--threads`，默认 1，异常值回默认 | `get_nvcc_parallelism_flags`，`flashinfer-ori: flashinfer/jit/cpp_ext.py` |
| 批量 prebuild | 各 spec 写 subninja，再由全局 `flashinfer_jit.lock` 保护总 Ninja；可用 `FLASHINFER_JIT_PREBUILD_MAX_JOBS` | `build_jit_specs`，`flashinfer-ori: flashinfer/jit/core.py` |
| DSL 模块一致性 | 同 family 的 kernel 共用 module lock，避免 meta 失效清目录与兄弟 kernel export 竞争 | `JitSpecCuteDsl.lock_path`，`flashinfer-ori: flashinfer/jit/cute_dsl_core.py` |
| DSL 批编译辅助 | `compile_and_persist()` 可锁外 compile、锁内提交；不是默认 `build_and_load()` 的调用路径 | `flashinfer-ori: flashinfer/jit/cute_dsl_core.py` |
| DSL 原子提交 | 临时对象 `os.replace`，然后写 meta；meta 是完整性提交标记；export 失败允许保留进程内 kernel | `JitSpecCuteDsl._export/build/load`，`flashinfer-ori: flashinfer/jit/cute_dsl_core.py` |
| 禁用编译 | `FLASHINFER_DISABLE_JIT` 缺失预编译产物时抛 `MissingJITCacheError(spec=...)`；不表示所有独立 DSL/第三方编译器被统一禁用 | `JitSpec.build_and_load/JitSpecNvcc.build`，`flashinfer-ori: flashinfer/jit/core.py` |

`FLASHINFER_DISABLE_JIT` 的 NVCC 严格模式只接受可加载的 AOT/provider，
即使本地 JIT cache 已有 `.so`，`try_load()` 仍返回 miss，并在 Ninja 新鲜度检查之前报错。
CuTe-DSL 的有效 `.o` 则可以通过自己的 `try_load()` 命中；两种缓存的严格模式语义不同。
证据：`flashinfer-ori: flashinfer/jit/core.py`，`build_and_load/JitSpecNvcc.try_load`。

CuTe-DSL 的 `load()` 在磁盘对象存在时优先重新加载产物；否则可返回刚编译的内存对象。
实际代码优先于文件中个别“避免重新加载”的旧注释。
`FLASHINFER_CUTE_DSL_DISABLE_CACHE=1` 或 key file 无法读取时，便利函数直接调用 `compile_fn()`；
这条绕过共同 spec 生命周期的路径应单独审视禁用 JIT 与离线策略。
证据：`flashinfer-ori: flashinfer/jit/cute_dsl_core.py`，`build_and_load_cute_dsl_kernel/load`。

#### 3.1.5 与其他编译技术的拼接

CuTe-DSL：`build_and_load_cute_dsl_kernel()` 接收调用者提供的零参 compile closure；
`JitSpecCuteDsl` 产出 `.o`，通过 `cute.runtime.load_module(..., enable_tvm_ffi=True)` 找到导出函数。
它复用锁/生命周期抽象，但不等同于 NVCC `.so` provider 分发。
证据：`flashinfer-ori: flashinfer/jit/cute_dsl_core.py`。

NVRTC：`prepare_generated_flash_kda_cubin()` 调用 `cuda.bindings.nvrtc` 的 create/compile/get-CUBIN；
返回的 mapping 可交给 `embedded_cubin_factory`，Ninja 用 `tvm_ffi.utils.embed_cubin` 将 cubin 嵌入 host object。
证据：`flashinfer-ori: flashinfer/jit/flash_kda_nvrtc.py`；`flashinfer-ori: flashinfer/jit/cpp_ext.py`，`embed_cubin` rule。
这证明 NVRTC 在当前仓库有实际用途，但不表示主 attention/GEMM 全部使用 NVRTC。

延迟加载还可在已经加载主 module 后发生：`_LazyPagedKVStrideModule.get()` 针对独立 K/V stride variant
用线程锁和 per-device module 缓存准备第二个 module；捕获期未预热则报错。
证据：`flashinfer-ori: flashinfer/jit/attention/_lazy.py`，`get/prewarm/_check_not_capturing`。
这条链路解释了为何“安装 AOT wheel”与“所有运行分支都可立即捕获”不是同一条保证。

核查测试入口：`flashinfer-ori: tests/jit/test_jit_cpp_ext.py`、`flashinfer-ori: tests/jit/test_cute_dsl_cache.py`、
`flashinfer-ori: tests/jit/test_jit_cache_providers.py`、`flashinfer-ori: tests/jit/test_paged_kv_lazy_loader.py`。
`build_jit_specs()` 自身不检查 `FLASHINFER_DISABLE_JIT`；该变量是上述运行期加载/build路径的策略，
不能概括为禁止任何 AOT 或直接批编译入口。证据：`flashinfer-ori: flashinfer/jit/core.py`。
本轮核查源码与测试意图，未执行 GPU 编译和这些运行期测试。

### 3.2 AOT 与 artifact 分发

#### 3.2.1 预编译范围与粒度

AOT 在此基线不是另一套 kernel 实现，而是提前执行同一组 JIT generator，生成 `JitSpec`、编译 `.so`，再打包。
调用链为 `aot.compile_and_package_modules` → `get_default_config` / `detect_sm_capabilities` → `gen_all_modules` → `build_jit_specs` → `copy_built_kernels`。
证据：`flashinfer-ori: flashinfer/aot.py`，以上同名函数。

预编译粒度是**模块 URI / 特化组合**，不是每一个运行期 tensor shape。
同一个模块可能包含多种 dtype dispatch、多个 CUDA kernel，运行期 batch size、序列长度通常仍是参数。
`gen_all_modules` 最后按 `jit_spec.name` 去重，因此同 URI 被多个特化枚举入口重复生成时只保留一份。
证据：`flashinfer-ori: flashinfer/aot.py`，`gen_all_modules` 末尾去重循环；`flashinfer-ori: flashinfer/jit/activation.py`，`gen_act_and_mul_module` 与模板内 dtype dispatch。

| AOT 枚举域 | 本基线默认组合 / 过滤规则 | 关键文件与函数 |
|---|---|---|
| FA2 attention | Q/O 为 FP16、BF16；KV 为 FP16、BF16、FP8 E4M3；同字节不同 dtype 组合被过滤；QK/VO head dim 默认 `(64,64)`、`(128,128)`、`(256,256)`；sliding-window、soft-cap 各取真假；index 为 int32、position encoding 为 0 | `flashinfer-ori: flashinfer/aot.py`，`get_default_config`、`gen_attention`、`gen_fa2` |
| FA2 特殊 scale | 16-bit Q/O × FP8 KV × 相同 head dim/window/soft-cap 组合，额外 `use_token_head_sf=True`；该分支 decode 复用 prefill module | `flashinfer-ori: flashinfer/aot.py`，`gen_fa2`、`gen_attention` |
| FA3 attention | SM90 条件启用；QKV 同 dtype；16-bit 输出与输入相同；默认 dims `(192,128)`、`(128,128)`、`(64,64)`、`(256,256)`；FP8 跳过 QK dim 192/64 | `flashinfer-ori: flashinfer/aot.py`，`gen_fa3` |
| MLA | FP16/BF16，CKV=512、KPE=64，FA2 与条件 FA3；SM100/103 另加 MLA module | `flashinfer-ori: flashinfer/aot.py`，`gen_attention` |
| Gemma / attention sink | Gemma dim=256、window+soft-cap；OAI OSS sink dim=64、FP16/BF16、window 开关、FA2/条件 FA3 | `flashinfer-ori: flashinfer/aot.py`，`gen_attention` |
| XQA | CUDA 版本必须大于 12.8，且目标支持；输入 FP16/BF16；FP8 KV 开关；page size=16/32/64/128；head dim=64/128/256；group ratio=1/2/4/8；window 开关 | `flashinfer-ori: flashinfer/aot.py`，`gen_xqa`、`gen_all_modules` |
| XQA MLA | SM120/121，Q/KV 为 FP8 E4M3、head dim=576、group ratio=128、无 sliding window，遍历 page size | `flashinfer-ori: flashinfer/aot.py`，`gen_xqa` |
| Activation | `act_func_def_str` 的 silu、gelu、gelu_tanh；每种 activation 一个模块，dtype 在 C++ dispatch | `flashinfer-ori: flashinfer/aot.py`，`add_act` 分支；`flashinfer-ori: flashinfer/jit/activation.py` |
| GEMM / MoE | `add_moe=True`；按 SM80/90/100/103/107/110/120/121 能力加入 GEMM、FP4、CUTLASS fused-MoE、TRT-LLM 等 generator；部分 Cake 模块按 exact architecture 分开 | `flashinfer-ori: flashinfer/aot.py`，`gen_all_modules` 的 `add_moe` 分支 |
| 通信 | `add_comm=True`；TRT-LLM、MNNVL、DCP、NVLink one/two-sided、vLLM custom-allreduce、PCIe IPC AG/RS；各有架构与 toolkit gate | `flashinfer-ori: flashinfer/aot.py`，`add_comm` 分支 |
| 基础算子 | `add_misc=True`：API log stats、cascade、mHC、norm、page、packbits、RoPE、sampling、topk；另含架构相关 softmax、FP4 KV、sampling | `flashinfer-ori: flashinfer/aot.py`，`add_misc` 分支 |
| RMSNorm+SiLU | SM100/103，LUT 中 C/token/dtype=`bf16/fp8/nvfp4`；另有 C=768/1280/1536/2048/2560/3072/4096/5120/6144/8192 fallback knobs | `flashinfer-ori: flashinfer/aot.py`，`add_misc` 分支；`flashinfer-ori: flashinfer/jit/rmsnorm_silu.py` |
| SSM state update | state 为 BF16/int16/FP32 三类组合；input/weight BF16、A FP32、index int64；dim=64、dstate=128、ntokens=1/4/6/8；两个索引 dtype 各 int32/int64；SM80、SM90 分支 | `flashinfer-ori: flashinfer/aot.py`，`_ssu_dtype_combos` 及后续循环 |
| 模型专用与精确架构包 | Cake FMHA/MSA、MiniMax H3、FlashKDA/Cake KDA、FP8 grouped GEMM、SM120 sparse MLA 等，按具体可用性函数与 exact target 纳入 | `flashinfer-ori: flashinfer/aot.py`，`gen_all_modules` |
| 默认不等于覆盖全集 | 默认 head dim 不含 512；independent paged-KV stride module 不在默认 cache；自定义 attention URI 无自动全量枚举 | `flashinfer-ori: flashinfer/aot.py`，`get_default_config`；`flashinfer-ori: flashinfer/jit/attention/_lazy.py`，`get` |

`compile_and_package_modules` 强制要求显式设置 `FLASHINFER_CUDA_ARCH_LIST`。
它重设 `jit_env` 中的源码、include、workspace、generated、cached_ops 路径，再调用通用编译器。
`copy_built_kernels` 按每个 spec 的 name 复制 `.so`，逻辑产物布局为 `jit_cache/<module_name>/<module_name>.so`。
这里的逻辑产物路径是构建时生成的模板，不是宣称源码检出已存在每个 `.so`。
证据：`flashinfer-ori: flashinfer/aot.py`，`compile_and_package_modules`、`copy_built_kernels`。

#### 3.2.2 shim、架构 provider 与 cubin wheel

**实际分发已变为 shim + 分架构 provider。**
`flashinfer-jit-cache` 不再直接打包单体二进制全集；其 backend 的 `_prepare_build` 还会移除旧单体输出。
shim 读取 `FLASHINFER_JIT_CACHE_PROVIDER_ARCHS`，写出每个 `flashinfer-jit-cache-smXX==version` 依赖。
真正的 provider 工程要求 `FLASHINFER_JIT_CACHE_PROVIDER_ARCH` 选定一个目标架构。
`get_provider_build_config` 将其转换为分发名 `flashinfer-jit-cache-smXX` 和 Python namespace `flashinfer_jit_cache.providers.smXX`。
证据：`flashinfer-ori: flashinfer-jit-cache/build_backend.py`，`_write_provider_requirements`、`_prepare_build`；`flashinfer-ori: flashinfer-jit-cache-provider/package_config.py`，`get_provider_build_config`。

Provider 构建调用链：`build_wheel` → `_build_aot_modules` → `aot.compile_and_package_modules(skip_prebuilt=False)` → `_write_provider_manifest`。
manifest 包含 schema_version=1、provider_id、distribution、version、cuda_architectures、modules。
`setup.py` 打包 manifest 与 `jit_cache/**/*.so`，注册 entry point 组 `flashinfer.jit_cache.providers`。
provider 的 `get_provider` 读取 manifest，并补入安装后的真实 cache 目录。
证据：`flashinfer-ori: flashinfer-jit-cache-provider/build_backend.py`；`flashinfer-ori: flashinfer-jit-cache-provider/setup.py`；`flashinfer-ori: flashinfer-jit-cache-provider/flashinfer_jit_cache_provider/__init__.py`。

运行期调用链：shim `get_jit_cache_providers` → `jit.env._get_aot_providers` → `get_aot_artifacts(module_name)` → `JitSpecNvcc.try_load` → `tvm_ffi.load_module`。
先接受主包保留的 AOT 目录中存在的模块，再在 provider manifest 中筛选模块名、版本、CUDA binary target 兼容性。
主包版本与 shim 允许 CUDA local-version 后缀；provider 与 shim 要求同版本。
不兼容 shim/provider 被 warning 后忽略，避免连 CLI 修复命令都无法 import。
证据：`flashinfer-ori: flashinfer-jit-cache/flashinfer_jit_cache/__init__.py`；`flashinfer-ori: flashinfer/jit/env.py`，`_check_jit_cache_version`、`_check_jit_cache_provider_version`、`get_aot_artifacts`。

`get_aot_artifacts` 对每个可见目标选兼容分数最高的 provider。
一份 AOT 足以覆盖全部目标时直接返回 module；多 GPU 异构目标可由 `_HeterogeneousAOTModule` 根据调用参数的 CUDA device 分派。
AOT 加载失败时 warning 并继续尝试；无可用 AOT 时进入 `build`，由 Ninja 判断本地 JIT `.so` 是否仍新鲜。
`FLASHINFER_DISABLE_JIT` 开启而发生 miss 会抛 `MissingJITCacheError`，不会假装离线缓存覆盖所有组合。
证据：`flashinfer-ori: flashinfer/jit/core.py`，`JitSpecNvcc.try_load`、`_HeterogeneousAOTModule`、`JitSpecNvcc.build`。

`flashinfer-cubin` 的 wheel 则把远端 artifact 下载结果打包进 `flashinfer_cubin` 的 `cubins` 数据目录。
其 backend `_download_cubins` 临时覆盖 `FLASHINFER_CUBIN_DIR`，调用主包 `artifacts.download_artifacts`，结束后恢复环境变量。
`get_cubin_dir` / `get_cubin_path` / `list_cubins` 提供定位接口，本身不提供 tensor 算子 API。
证据：`flashinfer-ori: flashinfer-cubin/build_backend.py`；`flashinfer-ori: flashinfer-cubin/pyproject.toml`；`flashinfer-ori: flashinfer-cubin/flashinfer_cubin/__init__.py`。

cubin 路径优先级是显式 `FLASHINFER_CUBIN_DIR` → 同版本已安装 cubin wheel → 默认磁盘缓存。
版本不匹配的 cubin wheel 被忽略，转为按需下载；不能把“安装过 cubin wheel”视为一定命中。
证据：`flashinfer-ori: flashinfer/jit/env.py`，`_get_cubin_dir`、`_check_cubin_version`。

#### 3.2.3 运行期 AOT 选择与 JIT 回落的准确顺序

`JitSpecNvcc.try_load()` 的候选来自 `get_aot_artifacts(module_name)`：

1. 优先检查主包内嵌 AOT 路径；若存在，作为架构元数据未声明的内嵌 artifact 返回。
2. 否则读取可见目标架构，对 provider 的 module manifest、目标架构兼容性及 `.so` 实际存在性筛选。
3. 对每个目标优先 exact/较优兼容架构，去重后返回一个或多个 artifact；不是简单按安装顺序取首个 wheel。
4. 逐个加载已选 `.so`；加载失败记录 warning，尝试其他已选 artifact；没有可加载产物则返回 cache miss。
   选择阶段已为每个 target 取最优 provider，加载失败不重新搜索该 target 的次优 provider。
5. 单 artifact 覆盖全部目标时直接返回 module；否则 `_HeterogeneousAOTModule` 根据调用 Tensor 所在设备/当前设备分派。
6. 不被已加载 artifact 覆盖的调用进入延迟 JIT fallback；fallback 也受模块锁和禁用 JIT 检查约束。

证据：`flashinfer-ori: flashinfer/jit/env.py`，`get_aot_artifacts/_provider_compatibility_score/_cuda_architectures_for_call`；
`flashinfer-ori: flashinfer/jit/_cuda_architecture.py`，`cuda_binary_target_compatibility_score`；
`flashinfer-ori: flashinfer/jit/core.py`，`JitSpecNvcc.try_load/_HeterogeneousAOTModule`。

版本检查也参与可用性：main version 与 cache shim/provider version 需匹配，cache 允许 CUDA local-version suffix；
provider 与 shim 还要求一致。失配的包记录 warning 后忽略，使安装修复命令仍可 import。
`FLASHINFER_DISABLE_VERSION_CHECK` 可绕过检查；源码安装 version unknown 有特殊分支。
证据：`flashinfer-ori: flashinfer/jit/env.py`，`_get_aot_providers/_check_jit_cache_provider_version`。

这条路径只能证明主 NVCC module 的 AOT→JIT 选择。
不能把“任意已存在的本地 JIT `.so`”写成 AOT：其新鲜度仍需 Ninja 检查。
也不能把所有 CuTe-DSL 对象缓存都视为 provider wheel 的默认内容。

#### 3.2.4 cubin artifact：远端版本钉住、校验和动态库回调

| 环节 | 真实行为 | 关键文件/符号 |
|---|---|---|
| 远端根地址 | 默认 NVIDIA artifactory，环境变量 `FLASHINFER_CUBINS_REPOSITORY` 可覆盖 | `flashinfer-ori: flashinfer/jit/cubin_loader.py`，同名常量 |
| artifact 身份 | `ArtifactPath` 固定 TRTLLM FMHA/BMM/GEMM、DeepGEMM、DSL FMHA 等发布路径；`CheckSumHash` 固定 manifest SHA-256 | `flashinfer-ori: flashinfer/artifacts.py` |
| 本地目录优先级 | `FLASHINFER_CUBIN_DIR` → 版本相容的 cubin wheel → 默认 cache；版本失配 wheel 被忽略 | `flashinfer-ori: flashinfer/jit/env.py`，`_get_cubin_dir` |
| 按需取得 | `get_artifact(name, sha256)` 默认先 `load_cubin()` 读并校验，miss 再下载，最后再次加载/校验；失败可能返回空 bytes，调用者另行断言/报错 | `flashinfer-ori: flashinfer/jit/cubin_loader.py` |
| 离线约束 | `FLASHINFER_NO_DOWNLOAD` 为真时，缺失/不匹配 artifact 直接报错；这与禁用源码 JIT 是两个开关 | `flashinfer-ori: flashinfer/jit/cubin_loader.py`，`get_artifact` |
| 下载去重 | destination 文件锁；锁内按预期 hash 复查；UUID 临时文件 + `os.replace` 原子提交；HTTP retry 带 backoff/jitter | `flashinfer-ori: flashinfer/jit/cubin_loader.py`，`download_file` |
| header 一致性 | `get_meta_hash()` 从 manifest 解析头文件摘要；`ensure_symlink()` 和 `verify_symlinked_headers()` 处理生成目录映射及旧头文件失效 | `flashinfer-ori: flashinfer/jit/cubin_loader.py` |
| 全量打包下载 | `get_subdir_file_list()` 枚举文件；`download_artifacts()` 默认四线程、每线程 session、跳过已校验缓存、汇总失败、逐文件验 hash、清理不属本次清单的旧文件 | `flashinfer-ori: flashinfer/artifacts.py` |
| Python→C callback | `setup_cubin_loader()` 以 `ctypes.CDLL` 查找 C 导出，保留 callback 引用防 GC | `flashinfer-ori: flashinfer/jit/cubin_loader.py` |
| C→Python→C 数据流 | native `getCubin(name,sha)` → callback → `get_artifact()` → `FlashInferSetCurrentCubin()` → thread-local string | `flashinfer-ori: include/flashinfer/cubin_loader.h` |

可核查的 TRTLLM FMHA 准备链：

```text
gen_trtllm_gen_fmha_module
  → ArtifactPath + CheckSumHash
  → get_artifact(checksum manifest, pinned hash)
  → get_meta_hash → get_artifact(metadata header, header hash)
  → gen_jit_spec("fmha_gen", launcher/reduction/quant sources)
  → .build_and_load()
  → for library_path in spec.get_library_paths(): setup_cubin_loader(path)
  → native getCubin → Python artifact callback → CUDA backend launcher
```

证据：`flashinfer-ori: flashinfer/jit/attention/modules.py`，`gen_trtllm_gen_fmha_module`；
`flashinfer-ori: flashinfer/decode.py`，`get_trtllm_gen_fmha_module`；
`flashinfer-ori: csrc/trtllm_fmha_kernel_launcher.cu`；`flashinfer-ori: include/flashinfer/cubin_loader.h`。
这里可同时需要 host launcher `.so` 与设备 cubin，因此 cubin wheel 不能自动替代所有 JIT-cache provider。

校验边界：默认按需路径先校验被钉住的 manifest，再取其中 header hash；
`FLASHINFER_CUBIN_CHECKSUM_DISABLED` 为任意非空值时，`load_cubin()` 直接返回 bytes 而跳过摘要比较。
全量打包的 `verify_cubin()` 路径没有这项绕过。
全量下载的 `get_checksums()` 先解析下载的清单，`download_artifacts()` 完成时再统一校验清单及条目。
本地代码提供 SHA-256 完整性链，未发现这里另有签名/证书公钥验证流程；不把 hash 校验写成独立的发布者签名认证。
关闭校验会削弱按需加载的上述完整性链，应作为部署配置事实单独记录。
证据：`flashinfer-ori: flashinfer/artifacts.py`，`get_checksums/download_artifacts`。

对应测试证据：`flashinfer-ori: tests/test_artifacts.py`；
`flashinfer-ori: tests/utils/test_load_cubin_compile_race_condition.py`。
本轮没有实际联网下载、构建 wheel 或验证远端发布 pin 现在仍可访问。

### 3.3 算子注册与 PyTorch 集成

**本基线存在两套注册路径；不能把所有 `@register_custom_op` 都当成已注册的 PyTorch op。**
`flashinfer-ori: flashinfer/utils.py` 的 `register_custom_op`、`register_fake_op` 在 PyTorch `<2.4` 与 `>=2.4` 两个分支中均返回 identity decorator。
较新版本分支保留了 `torch.library.custom_op` / `torch.library.register_fake` 的注释代码，并注明 custom_op 调用开销问题。
因此这些辅助装饰器仍表达接口意图，但在所调研 commit 不执行 dispatcher 注册。

| 路径类别 | 关键符号与实际行为 | 证据 |
|---|---|---|
| 传统统一辅助注册器 | `register_custom_op` 接收 `mutates_args`、`device_types`、`schema`，实际原样返回函数；`register_fake_op` 同样为空操作 | `flashinfer-ori: flashinfer/utils.py`，400–445 行 |
| 传统算子的模块适配器 | `get_act_and_mul_module` 用 JIT 加载符号，再构造 `_act_and_mul`、`_fake_act_and_mul`，返回 `SimpleNamespace` | `flashinfer-ori: flashinfer/activation.py`，48–70 行 |
| 传统 attention 的执行边界 | `get_batch_decode_module` 返回 `plan`、`workspace_size`、`run`；注释明确 `plan` 不属于模型执行逻辑，不放入 CUDA Graph / `torch.compile` | `flashinfer-ori: flashinfer/decode.py`，624–634 行 |
| 直接有效注册的量化 op | `_fp4_quantize_custom_op` 使用 `@torch.library.custom_op("flashinfer::fp4_quantize", mutates_args=())`；其 `.register_fake` 根据输入布局生成输出元数据 | `flashinfer-ori: flashinfer/quantization/fp4_quantization.py`，1232 行起 |
| 直接有效注册的 MoE 工具 | `reswizzle_sf` 注册为 `trtllm::reswizzle_sf`，另有真实 `torch.library.register_fake` | `flashinfer-ori: flashinfer/fused_moe/utils.py`，162、202 行 |
| 实验性功能的完整 op 边界 | blockscaled GEMM 分别注册 functional、workspace-mutating、out-mutating 变体，并分别提供 fake | `flashinfer-ori: flashinfer/experimental/b12x/gemm/blockscaled/_ops.py`，77 行起 |

传统激活调用链是：

```text
get_act_and_mul_module(act_func_name)
  → gen_act_and_mul_module(act_func_name).build_and_load()
  → getattr(module, "<act>_and_mul")
  → Python _act_and_mul(out, input, enable_pdl)
  → 已加载模块中的同名执行函数
```

该链证据位于 `flashinfer-ori: flashinfer/activation.py` 的 `get_act_and_mul_module`。
这里 `mutates_args=("out",)` 与 no-op fake 声明了预期的原地输出契约，但不能据此推导 dispatcher 的 mutation 跟踪已经生效。
真实注册路径则不同：`_fp4_quantize_custom_op` 进入 `fp4_quantize` 执行实现，fake 按 uint8 的两值打包与 scale-factor 布局创建 `new_empty`，不执行 GPU 量化。
证据：`flashinfer-ori: flashinfer/quantization/fp4_quantization.py`，`_fp4_quantize_custom_op`、`_fp4_quantize_fake`。

`torch.compile` 能力应以具体入口和测试方式为单位标记：

- `flashinfer-ori: tests/conftest.py` 的 `_monkeypatch_add_torch_compile` 先执行 eager warmup，再以 `fullgraph=True`、`backend="inductor"`、`mode="max-autotune-no-cudagraphs"` 包装指定函数。
- 这是一条测试工具链，不证明所有 API 默认都可完整追踪；该调研未执行此测试选项。
- `flashinfer-ori: tests/attention/test_batch_decode_kernels.py` 的 `test_single_decode_torch_compile_cuda_graph` 在测试脚本模块层自行 `torch.library.define` / `impl` / `register_fake` 包住 `single_decode_with_kv_cache`，然后以 `mode="reduce-overhead", fullgraph=True` 编译外层 op。
- `flashinfer-ori: tests/attention/test_batch_prefill_kernels.py` 的 `test_single_prefill_torch_compile_cuda_graph` 使用同样的外部包裹方法；其证据是“可通过明确的 op 边界被消费”，不是“所有 Python 路径自然可追踪”。
- `flashinfer-ori: tests/moe/test_alphamoe_fused_router.py` 的 `test_alphamoe_fused_router_torch_compile` 直接注明辅助注册器当前禁用，并使用 `backend="eager", fullgraph=False`，允许 graph break。
- `flashinfer-ori: tests/experimental/b12x/gemm/test_blockscaled.py` 存在 `fullgraph=True` 用例，与实验子系统直接注册的 op 路径配套；测试源码存在和本机已运行通过必须区分。

对新增算子的约束是：声明 dtype、shape、alias/mutation 契约，并为真正注册的 op 编写 fake；若沿用 `utils` 中的辅助装饰器，必须把其当前空实现列入集成风险。
禁止把 `@flashinfer_api` 日志/trace 装饰器与 `torch.library` 注册混为一层；前者的定义位于 `flashinfer-ori: flashinfer/api_logging.py`。

### 3.4 图模式（CUDA Graph）

**attention 图模式的主设计是“捕获外 plan + 稳定地址的 metadata/workspace + 捕获 run + 原位更新后 replay”。**
`use_cuda_graph=True` 主要配置资源与规划约束，并不意味着 wrapper 自己负责创建调用方的整个 CUDA Graph。
`flashinfer-ori: tests/attention/test_batch_decode_kernels.py` 的 `test_cuda_graph_batch_decode_with_paged_kv_cache` 给出完整实证流程：先 `plan`、warmup，再由调用方 `with torch.cuda.graph(g): wrapper.run(...)`，随后循环 `plan(...)` 并 `g.replay()`。

具体调用链和地址稳定机制：

1. `BatchDecodeWithPagedKVCacheWrapper.__init__` 保存调用方 float workspace，并创建 device int workspace 和 pinned host int workspace；图模式检查调用方传入的 `indptr`、`indices`、`last_page_len` buffers。
2. `plan` 固定 batch size；对 page indices 做容量检查，对图模式的 `q_len_per_req` 和 auto 已解析后端做冻结检查。
3. `plan` 用 `.copy_` 更新原有 metadata tensors，调用底层 planner 写入已分配的 int/float workspace，保存 `_plan_info`。
4. `run` 使用这些 buffers、`_plan_info`、Q/K/V 与输出指针调用加载后的 `run` 或 `paged_run`。
5. capture 完成后的 metadata 更新通过下一次捕获外 `plan` 生效；replay 重放既有 kernel 图，不重新执行 Python 的 backend 选择。

证据：`flashinfer-ori: flashinfer/decode.py`，`BatchDecodeWithPagedKVCacheWrapper.__init__`（1395 行起）、`plan`（2175–2250、2554–2590 行）、`run`（3444、3491 行）。

paged prefill 同样固定 batch size，并增加 `max_total_num_rows` 容量约束。
未显式给容量时首个 plan 固定最大行数；后续不得超过它，custom mask 模式还需要预先提供 `custom_mask_buf` 与 `mask_indptr_buf`。
`plan` 通过 `.copy_` 更新 `qo_indptr`、page metadata、packed mask；非 graph 模式则允许 `.to(device)` 替换这些 tensor 引用。
证据：`flashinfer-ori: flashinfer/prefill.py`，`BatchPrefillWithPagedKVCacheWrapper.__init__`、`plan`（3708–3786 行）。

C++ planner 也配合固定图规模，而非仅 Python 端保留 tensor：
`DecodePlanInfo` 保存 `padded_batch_size`、workspace offsets、`enable_cuda_graph`、`split_kv`。
`DecodePlanImpl` 在 graph 模式中使用 padding 和 `block_valid_mask`，将有效 work 与预分配容量分离，并通过 pinned host buffer 准备调度信息。
证据：`flashinfer-ori: include/flashinfer/attention/scheduler.cuh`，`DecodePlanInfo`、`DecodePlanImpl`；`flashinfer-ori: csrc/batch_decode.cu` 的 plan/run 负责该信息的序列化与反序列化。

以下是源码检索后确认的**显式具有 `use_cuda_graph` 构造参数的 wrapper 类**；不把内部 runner、scratch dataclass 与测试函数混入公开 wrapper 清单。

| wrapper | 参数作用或继承关系 | 关键文件 |
|---|---|---|
| `BatchDecodeWithPagedKVCacheWrapper` | 稳定 CSR buffers、固定 batch 与每请求 Q 长度 | `flashinfer-ori: flashinfer/decode.py` |
| `BatchPrefillWithPagedKVCacheWrapper` | 稳定 paged metadata、mask 与最大 Q 行数 | `flashinfer-ori: flashinfer/prefill.py` |
| `BatchPrefillWithRaggedKVCacheWrapper` | 固定 ragged `qo_indptr` / `kv_indptr` 与 mask buffers | `flashinfer-ori: flashinfer/prefill.py` |
| `BatchMLAPagedAttentionWrapper` | 保存 MLA metadata buffers，由具体 backend 能力决定能否 graph replan | `flashinfer-ori: flashinfer/mla/_batch_mla/_wrapper.py` |
| `PODWithPagedKVCacheWrapper` | 为其中 batch decode 部分提供稳定 page metadata | `flashinfer-ori: flashinfer/pod.py` |
| `MultiLevelCascadeAttentionWrapper` | 将参数传入每层 `BatchPrefillWithPagedKVCacheWrapper` | `flashinfer-ori: flashinfer/cascade.py` |
| `BatchAttentionWithAttentionSinkWrapper` | 继承 paged prefill，并通过自定义 variant 提供 sinks | `flashinfer-ori: flashinfer/attention/_core.py` |
| `BatchPrefillWithCausalBidirectionalRangesWrapper` | 继承 paged prefill 的图模式契约，附加 variant 的特化限制 | `flashinfer-ori: flashinfer/attention/_core.py` |
| `BatchDecodeCuteDSLWrapper` | CuTe-DSL decode wrapper 的图模式参数 | `flashinfer-ori: flashinfer/cute_dsl/attention/wrappers/batch_decode.py` |
| `BatchDecodePagedCuteDSLWrapper` | CuTe-DSL paged decode wrapper 的图模式参数 | `flashinfer-ori: flashinfer/cute_dsl/attention/wrappers/batch_decode.py` |
| `BatchPrefillCuteDSLWrapper` | CuTe-DSL prefill wrapper 的图模式参数 | `flashinfer-ori: flashinfer/cute_dsl/attention/wrappers/batch_prefill.py` |
| `CuteDslMoEWrapper` | W4A4 创建持久 stream/events 以支持异步清零；W4A16 不需要这些资源 | `flashinfer-ori: flashinfer/fused_moe/cute_dsl/fused_moe.py` |
| `CuteDslMxfp8Mxfp4MoEWrapper` | 兼容类向父类传参数；源码标记 deprecated | `flashinfer-ori: flashinfer/fused_moe/cute_dsl/fused_moe.py` |
| `B12xMoEWrapper` | 按 `max_num_tokens` 预分配 workspace，支持共享静态/动态 workspace | `flashinfer-ori: flashinfer/fused_moe/cute_dsl/b12x_moe.py` |

另外，`CUDAGraphBatchDecodeWithPagedKVCacheWrapper` 的构造函数不暴露此开关，而是固定向父类传 `use_cuda_graph=True`；证据：`flashinfer-ori: flashinfer/decode.py`，3557–3606 行。
`BlockSparseAttentionWrapper` 与 `VariableBlockSparseAttentionWrapper` 的 `_use_cuda_graph` 在当前构造流程中设为 `False`，不可因其复用 attention planner 就宣称公开支持该开关；证据：`flashinfer-ori: flashinfer/sparse.py`，460、1739 行。
实验子系统的 workspace/scratch 对象也存在 `use_cuda_graph` 字段；这类资源接口不等同于上述 wrapper，证据包括 `flashinfer-ori: flashinfer/experimental/b12x/attention/paged/workspace.py` 与 `flashinfer-ori: flashinfer/experimental/b12x/attention/_shared/workspace.py`。

**并非所有后端都允许同一个 graph wrapper 任意 replan。**
MLA wrapper 检查既有后端的 `supports_cuda_graph_replan`；若能力不满足，第二次 plan 抛出异常，要求稳定的 captured metadata pointers。
FA2/FA3 的可重规划路径显式保留旧 int workspace；其余组合需要逐后端看 capability，不能用“上游支持 CUDA Graph”代替兼容矩阵。
证据：`flashinfer-ori: flashinfer/mla/_batch_mla/_wrapper.py`，539–556、641–648 行；`flashinfer-ori: flashinfer/mla/_batch_mla/_backends/_capabilities.py`。
MLA `backend="autotune"` 还要求 capture 前在 `autotune(True)` 中 warmup `run`，准备已选后端；capture 中不允许首次开始 profiling。
证据：`flashinfer-ori: flashinfer/mla/_batch_mla/_auto_policy.py`，`_BatchMLAPagedAttentionAutotuneBackend`、`run_from_wrapper`。

与 `torch.compile` / piecewise graph 的关系也必须分层：

- CUDA Graph 解决执行捕获与地址稳定，`torch.compile` 解决 Python/Tensor 程序追踪；上节测试以外层 custom op 将两者组合。
- `flashinfer-ori: flashinfer/fused_moe/utils.py` 定义 `set_piecewise_cuda_graph_flag` / `get_piecewise_cuda_graph_flag`，但对该仓源码进行符号检索只找到定义，未找到内部消费点；不能据此认定存在全局图切分调度器。
- `flashinfer-ori: tests/moe/test_cute_dsl_bf16_moe.py` 的 `test_cute_dsl_bf16_moe_cuda_graph` 明确覆盖 MoE sort、两次 GEMM 与辅助 stream fork/join，注释将其对应到 vLLM piecewise graph 使用场景。
- 该测试更新静态 routing/activation buffers 后 replay，提供算子区域可被捕获的证据；vLLM 端实际如何切图、当前集成版本与端到端运行结果均未验证。

### 3.5 后端选择与 autotune

**`backend=` 是每个 API 自己的策略入口；`auto` 没有全仓通用的候选顺序。**
支持检查、静态 heuristic、实际 profiling 是三层职责，不应合并为“auto 总会测所有后端”。

| 机制/入口 | 选择规则 | 证据 |
|---|---|---|
| attention 基础 helper | `determine_attention_backend` 在 SM90a 且 FA3 能满足 mask、位置编码、dtype、head dimensions 时返回 `fa3`，其余返回 `fa2` | `flashinfer-ori: flashinfer/utils.py`，574 行起 |
| paged decode | `_auto_prefers_cudnn` 检查 cuDNN 可用性、架构、dtype、shape 与 graph metadata 约束；graph 模式无用户 block tables 时通常不走该 auto 分支 | `flashinfer-ori: flashinfer/decode.py`，1490–1558 行 |
| GEMM 支持过滤 | `backend_requirement` 根据 common checker、backend checker、compute capability 筛选，再调用算子自己的 `heuristic_func` | `flashinfer-ori: flashinfer/utils.py`，1255、1459 行起 |
| BF16 GEMM heuristic | `_heuristic_func_mm_bf16` 在无 bias/PDL 时按受支持情况排列 `cutlass`、`tgv`、`cudnn`、`cublaslt`，之后追加 `cute-dsl` / `tinygemm`；有 bias/PDL 时改变顺序与集合 | `flashinfer-ori: flashinfer/gemm/gemm_base.py`，623–665 行 |
| MLA auto | `_BatchMLAPagedAttentionAutoBackend.plan_from_wrapper` 逐候选尝试计划；SM80 优先 FA2，SM90 优先 FA3→FA2，SM12x 优先 FA2→XQA→cuTile，SM100 按 Q/KV 长度、heads、work 与图模式排序 | `flashinfer-ori: flashinfer/mla/_batch_mla/_auto_policy.py`，53、122–280 行 |
| MLA 显式 autotune | `backend="autotune"` 在 plan 时准备可用候选，run 时委托 `AutoTuner` 选择；与 `backend="auto"` 是不同策略 | `flashinfer-ori: flashinfer/mla/_batch_mla/_wrapper.py`、`flashinfer-ori: flashinfer/mla/_batch_mla/_auto_policy.py` |

`backend_requirement` 暴露 `is_backend_supported`、`is_compute_capability_supported` 与 `suitable_auto_backends`，允许调用方查询支持状况。
标注 `@experimental_backend` 的候选只有在显式允许 experimental auto 时进入自动选择；同一过滤后的集合也供 autotuner 使用。
若 checker 定义在实验命名空间却没有实验标记，装饰阶段会报错；这一约束避免实验实现无意进入默认路径。
证据：`flashinfer-ori: flashinfer/utils.py`，`backend_requirement`、`experimental_backend`。

以 BF16 GEMM 为例，实际组织链是：

```text
mm_bf16 → backend_requirement 的支持检查与 heuristic
        → mm_bf16.suitable_auto_backends
        → bf16_gemm_sm100(..., backends)
        → 候选 TunableRunner + TuningConfig
        → AutoTuner.choose_one(custom_op, runners, config, inputs)
        → 选中 runner.forward(..., tactic=...)
```

注意该例还存在二次组装：`bf16_gemm_sm100` 并不按传入列表遍历，而是用成员资格判断，依次加入 `cudnn`、`cublaslt`、`cutlass`、`tgv`、`tinygemm`、`cute-dsl` runners；因此 heuristic 列表主要筛选候选，最终 fallback 以重新组装后的第一个 runner 为准。
证据：`flashinfer-ori: flashinfer/gemm/gemm_base.py`，2578–2635 行；R2 对标时应追到实际 runner 构造处，不能只读 heuristic 返回列表。

入口与具体调度证据：`flashinfer-ori: flashinfer/gemm/gemm_base.py`，`mm_bf16`、`bf16_gemm_sm100`；调优机制证据：`flashinfer-ori: flashinfer/autotuner/autotuner.py`，`TunableRunner`、`TuningConfig`、`AutoTuner.choose_one`。
`TunableRunner` 提供候选 tactic 与实际 forward 接口；`TuningConfig` 描述 dynamic tensor specs、shape 约束、bucket 映射、profiling 模式与输入准备钩子。
只有进入 tuning context 且缓存未命中，才进行输入合成与 tactic 测量；普通 serving 调用查缓存，miss 使用第一个 runner 和 `tactic=-1` 的 fallback。
`choose_one` 全程使用全局锁串行 GPU profiling，源码建议多 GPU 并行调优采用每 GPU 一个进程。

缓存是“编译缓存之外”的另一套数据：

| 层次 | key / 内容 / 顺序 | 关键符号 |
|---|---|---|
| 内存结果 | `custom_op`、runner 类名、运行期 `hash(runner)`、最近 shape profile、runner extras | `ProfilingCacheKey`、`AutoTuner._get_cache_key` |
| 稳定文件 key | 去掉跨进程不稳定的 runner hash，保留 op、类名、profile 与 extras | `ProfilingCacheKey.key_fields`、`file_key` |
| 常规查找 | 当前内存 winner → 用户加载配置 → managed store（如已附着）→ bundled legacy 配置 → fallback；内存/用户/managed 来源验证 tactic；bundled legacy 仅在环境变量启用且非 tuning 时查找 | `AutoTuner.search_cache` |
| 旧版持久化 | `autotune(cache=<JSON>)` 进入时 load、tune 退出时 save；可单独 `load_configs` / `save_configs` | `autotune`、`AutoTuner.load_configs`、`save_configs` |
| 环境有效性 | 记录 FlashInfer、CUDA、cuBLAS、cuDNN backend/frontend、CuTe-DSL 栈版本和 GPU 名称，防止 tactic index 在不兼容环境中误用 | `_collect_metadata`、`_classify_metadata_mismatches` |

`bundled legacy` 读取开关为 `FLASHINFER_AUTOTUNER_LOAD_FROM_FILE=1`，不是默认总会加载的缓存层。
上述表格证据统一为 `flashinfer-ori: flashinfer/autotuner/autotuner.py`。
`DynamicTensorSpec` 与 `OptimizationProfile` 让多个运行 shape 映射到同一测量 bucket，默认/override rounding 和 extras 决定该结果可复用的范围。
不同 runner 的 extras 由 `get_cache_key_extras` 提供，不能假定仅 shape 相同就可以复用 tactic。
测量策略还跟踪 CUDA Graph replay 次数与 hot/cold L2 provenance；缓存命中不代表任意测量政策都可重复用作 tuning winner。

`flashinfer-ori: flashinfer/autotune_cache.py` 新增独立的实验性 `autotune_v2`，共享 runner、bucket、profiling 机制，但使用 FlashInfer 管理的逐条持久化：

1. `autotune_v2(mode="tune"|"replay", persistent_cache=...)` 区分调优与纯 serving；`replay` 不 profiling。
2. `ManagedAutotuneCache` 对环境 manifest 加 schema，取 SHA-256 的前 16 位为环境目录；entry key 再散列为文件名。
3. 默认根目录由 `get_default_cache_root` 返回 `FLASHINFER_CACHE_DIR` 下的 `autotune`，可用 `FLASHINFER_AUTOTUNE_CACHE_DIR` 覆盖。
4. 逻辑布局是 cache root → schema → environment hash → `entries`；另保存 manifest。这里是运行期目录结构，不是仓库源码路径。
5. `lookup` 检查 entry 内嵌 key 与请求完全相同，损坏/缺失当作 miss，并缓存正负查找结果。
6. `publish` 逐 winner 原子写入，不把正常 tuning 成败绑定到磁盘持久化成功；`_atomic_write_json` 实现替换式写入。
7. `MeasurementPolicy` 的 eager/graph 及 L2 policy 参与存储环境标识；`autotune_v2_reload` 可清除相关 memo 并重新读取已附着 store。
8. `persistent_cache=True` 的附着在 context 退出后仍有效；`persistent_cache=False` 的 context 使用独立内存 serving 语义；v2 不支持嵌套 v2 context。

autotune 与 capture 的边界是明确执行检查：`AutoTuner.choose_one` 在 cache miss、准备 profiling 前检查 `torch.cuda.is_current_stream_capturing()` 并拒绝在外层 active capture 中启动。
因此应先完成 JIT 与调优 warmup，再捕获 serving；warm-cache lookup 和明确跳过的算子可在该边界内运行。
证据：`flashinfer-ori: flashinfer/autotuner/autotuner.py`，`choose_one`（2385 行附近）；本轮未在 GPU 上验证 profiling 时间或 winner 质量。

### 3.6 plan / run 执行模型

**两阶段模型把依赖 batch metadata、shape 与后端的准备工作从每次 tensor 执行中移出，并通过 wrapper 保存其结果。**
该模型在 batch attention 最明显，不能推导所有算子都有 `plan()`；例如 `SegmentGEMMWrapper` 的公开主执行入口就是 `run`。
证据：`flashinfer-ori: flashinfer/decode.py` 的 `BatchDecodeWithPagedKVCacheWrapper`；`flashinfer-ori: flashinfer/gemm/gemm_base.py` 的 `SegmentGEMMWrapper`。

| 生命周期 | 保存/产生的信息 | 上游关键文件与符号 |
|---|---|---|
| 构造 | 调用方 float workspace、内部 device/pinned-host int workspace、layout、后端请求、可选自定义 JIT 模块、图模式 metadata buffers | `flashinfer-ori: flashinfer/decode.py`，`BatchDecodeWithPagedKVCacheWrapper.__init__` |
| workspace 估算 | 给定元数据与 shape 走同类调度估算，返回 float/int workspace 所需字节数 | `flashinfer-ori: csrc/batch_decode.cu`，`BatchDecodeWithPagedKVCacheWorkspaceSize` |
| plan 输入规范化 | dtype 规范化，读取 CPU metadata，检查 head/layout/mask/capacity，选择 backend 和特化模块 | `flashinfer-ori: flashinfer/prefill.py`，`BatchPrefillWithPagedKVCacheWrapper.plan` |
| plan 调度物化 | split-KV、tile/request indices、workspace offsets、padding/有效标记，返回可序列化 plan 信息 | `flashinfer-ori: include/flashinfer/attention/scheduler.cuh`，`DecodePlanInfo`、`PrefillPlanInfo` |
| wrapper 持有状态 | `_cached_module`、`_plan_info`、metadata buffers、cached dtypes、scale/window/rope 等参数 | `flashinfer-ori: flashinfer/decode.py`，`plan` |
| run | 解包 KV、核验计划 dtype、选择/校验 out/lse，然后调用既有 module；底层恢复 offsets 并执行 | `flashinfer-ori: flashinfer/decode.py`，`run`；`flashinfer-ori: csrc/batch_decode.cu`，`BatchDecodeWithPagedKVCacheRun` |
| 复用或重配 | 同计划可用于多个层的 Q/KV 数据；换 batch metadata 则重新 plan；`reset_workspace_buffer` 可替换资源但捕获图仍须遵守指针约束 | `flashinfer-ori: flashinfer/prefill.py`，`reset_workspace_buffer` 与 `plan` |

FA2 CUDA-core batch decode 的端到端调用链可以具体写成：

```text
BatchDecodeWithPagedKVCacheWrapper.plan
  → get_batch_decode_module(各 dtype、head dims、位置编码与开关)
  → self._cached_module.plan(float_workspace, int_workspace, pinned_int_workspace, host_indptr, ...)
  → C++ BatchDecodeWithPagedKVCachePlan
  → DecodePlan / DecodePlanImpl
  → DecodePlanInfo.ToVector() → Python self._plan_info
BatchDecodeWithPagedKVCacheWrapper.run(q, paged_kv_cache)
  → self._cached_module.run(workspaces, self._plan_info, q, k, v, metadata, out, lse, ...)
  → C++ BatchDecodeWithPagedKVCacheRun
  → DecodePlanInfo.FromVector() + workspace base/offset
  → 特化的 batch decode dispatch
```

该链证据：`flashinfer-ori: flashinfer/decode.py`，`get_batch_decode_module` / `plan` / `run`；`flashinfer-ori: csrc/batch_decode.cu`；`flashinfer-ori: include/flashinfer/attention/scheduler.cuh`。
这里 plan 信息不是直接持有每个设备指针的 Python DAG：核心 C++ plan 是 offsets、标志与调度规模序列，run 结合当次传入的 workspace 基址恢复对应指针。

其他后端保留两阶段外观，但内部实现不同：

- Tensor-core decode 可复用 batch prefill module 的 planner，并走 `paged_run`；证据：`flashinfer-ori: flashinfer/decode.py`，`plan` 的 `use_tensor_cores` 分支及 `run`。
- `trtllm-gen` decode 的 module `plan()` 返回 `None`，但 Python plan 仍准备 KV lengths、block tables 与 counter buffer，所以不能说“该后端完全没有准备阶段”。证据：`flashinfer-ori: flashinfer/decode.py`，2429–2488 行。
- cuDNN 路径保存 `_cudnn_prepared`、`_cudnn_plan` / lengths views，区分 prepared execution graph 与调用方 CUDA Graph；证据：`flashinfer-ori: flashinfer/decode.py` 与 `flashinfer-ori: flashinfer/prefill.py` 的 wrapper 初始化及 cuDNN 分支。
- MLA 采用更明确的 `_MLAPlanArguments`、`MLAPlanMetadata`、`MLAInputContract`、backend capabilities 与 `_planned_backend`；`plan_from_wrapper` 返回已准备 backend，`run_from_wrapper` 执行。证据：`flashinfer-ori: flashinfer/mla/_batch_mla/_wrapper.py`、`flashinfer-ori: flashinfer/mla/_batch_mla/_planning.py`。
- `MultiLevelCascadeAttentionWrapper` 给多个 prefill wrappers 共享 float workspace、分别管理各层 metadata，并在 `run` 合并状态；证据：`flashinfer-ori: flashinfer/cascade.py`，`__init__`、`plan`、`run`。

workspace 复用也分两类：wrapper 持有调用方资源，和无状态函数通过进程缓存持有 scratch。
`flashinfer-ori: flashinfer/utils.py` 的 `_get_cache_buf(name, num_bytes, device)` 以 `(name, device)` 为 key 扩容，并把旧 buffer 留在 `_cache_buf_retired`，避免仅因扩容就释放旧引用。
这不等于无限并发安全；流、并发调用和捕获重用仍应按具体 wrapper 契约验证。
例如 `CuteDslMxfp8Mxfp4MoEWrapper` 明确要求每 stream 一实例且不支持重入；证据：`flashinfer-ori: flashinfer/fused_moe/cute_dsl/fused_moe.py`，1995 行起。

## 4. 对外交付与扩展

### 4.1 Python 交付面与 import 行为

主分发名为 `flashinfer-python`，Python 导入名为 `flashinfer`。
顶层 export 是选定 API 的便捷入口，不代表所有子模块里的公开函数都会进入顶层。
证据：`flashinfer-ori: pyproject.toml` 的 project.name；`flashinfer-ori: flashinfer/__init__.py` 的显式 import。

| 顶层 API 分组 | 代表性公开符号 | 源码证据 |
|---|---|---|
| Attention | `BatchAttention`、`BatchDecodeWithPagedKVCacheWrapper`、`BatchPrefillWithPagedKVCacheWrapper`、`BatchPrefillWithRaggedKVCacheWrapper`、`BatchMLAPagedAttentionWrapper`、single decode/prefill、XQA | `flashinfer-ori: flashinfer/__init__.py` |
| Attention 组合 | `MultiLevelCascadeAttentionWrapper`、shared-prefix wrapper、`merge_state`、`PODWithPagedKVCacheWrapper`、`BlockSparseAttentionWrapper` | `flashinfer-ori: flashinfer/__init__.py` |
| GEMM / MoE | `mm_bf16`、`mm_fp8`、`mm_fp4`、`bmm_fp8`、`grouped_mm_bf16`、`SegmentGEMMWrapper`、`cutlass_fused_moe`、`trtllm_*_moe` | `flashinfer-ori: flashinfer/__init__.py` |
| 基础算子 | activation、RMSNorm/LayerNorm、RoPE、packbits、FP4/FP8 quantization、page append、sampling、top-k | `flashinfer-ori: flashinfer/__init__.py` |
| 线性注意力 / 模型算子 | `chunk_gated_delta_rule`、`chunk_gated_delta_rule2`、`chunk_gated_delta_product`、`recurrent_kda`、`fused_kda_decode`、`gdn_fused_decode_step`、`mamba`、MiniMax H3 | `flashinfer-ori: flashinfer/__init__.py` |
| 调优 / trace / build | `jit`、`autotune`、`autotune_v2`、`autotune_v2_reload`、`MeasurementPolicy`、`fi_trace` | `flashinfer-ori: flashinfer/__init__.py` |
| 部分训练与融合扩展 | `CakeRMSNormFunction`、`cake_rmsnorm`、`CakeCausalConv1dFunction`、Cake mHC train 系列 | `flashinfer-ori: flashinfer/__init__.py` |

不能把这个基线描述成“全包统一 lazy import”。
顶层大量 `from ... import ...` 是 eager Python import，但实际 CUDA module 通常由 `get_*_module` 在调用时编译加载。
`_PRIMS_TS_LAZY_EXPORTS` 的七个 MoE 符号经 `__getattr__` 获取并写入 globals，`__dir__` 补足可发现性。
可选 CuTe-DSL 高层 API 在 `contextlib.suppress(ImportError)` 下导入，部分 norm FP4 export 在 try/except 中导入。
`gdn_fused_decode_step` 本身总是 export，内部重依赖在 capability probe / first call 时加载，便于调用方用 `getattr` 做版本能力判断。
证据：`flashinfer-ori: flashinfer/__init__.py`，`__getattr__`、`__dir__` 及条件 import。

另一个细粒度 lazy 机制是 independent paged-KV stride module：`_LazyPagedKVStrideModule.get` 首次 eager 调用才 build/load。
它显式禁止 capture 中首次编译/加载，要求先调用 `prewarm_paged_kv_stride_variant('independent')`。
因此 lazy initialization 与 CUDA Graph 的边界是显式 warmup 协议。
证据：`flashinfer-ori: flashinfer/jit/attention/_lazy.py`，`_check_not_capturing`、`get`、`prewarm`。

`FLASHINFER_TRACE_APPLY` 可在 import 时 opt-in 安装替换逻辑；失败记录 warning 并继续 import。
`FLASHINFER_LOGLEVEL>=1` 可输出版本与 commit；该逻辑也保护 import 不因日志异常中断。
证据：`flashinfer-ori: flashinfer/__init__.py`，`_enable_apply_from_env` 调用、`_log_import_version`。

### 4.2 CLI 不是独立执行引擎

CLI 由 `pyproject.toml` 注册 `flashinfer = flashinfer.__main__:cli`，使用 Click。
`python -m flashinfer` 与 console script 进入同一 command group。
以下按功能归并，但点名实际命令：

| CLI 职责 | 子命令 / 参数 | 实际衔接 |
|---|---|---|
| 环境诊断 | `collect-env`、`show-config` | 版本、CUDA/NVCC、缓存路径、已下载 cubin、模块状态 |
| cubin 管理 | `list-cubins`、`download-cubin`、`install-cubin-wheel`、`clear-cubin` | artifact 下载或 pip 安装匹配 wheel |
| AOT cache 安装 | `install-jit-cache-wheel`，兼容别名 `download-jit-cache` | `all/minimal` 两种安装模式，`--sm` 选 provider |
| 一次安装 kernel wheel | `download-kernels` | 组合 cubin 与 jit-cache wheel 安装；可 `--dry-run` |
| JIT 诊断 | `module-status`、`list-modules`、`export-compile-commands` | `register_default_modules` / registry / `JitSpec.get_compile_commands` |
| 维护与复现 | `clear-cache`、`generate-tactics-blocklist`、`replay` | 缓存清理、MoE tactics blocklist、API dump replay |

表中实现与 Click decorator 均见 `flashinfer-ori: flashinfer/__main__.py`，`cli`、各 `*_cmd` 函数。
`_ensure_modules_registered` 在 registry 为空时调用 `aot.register_default_modules`，注册 spec 不等于完成 CUDA 编译。
`_install_jit_cache_wheel` 的 minimal 模式按 GPU target 选择 provider；`all` 模式安装 shim 依赖集合。
证据：`flashinfer-ori: flashinfer/__main__.py`，`_ensure_modules_registered`、`_install_jit_cache_wheel`、`_select_jit_cache_provider_tags`。

### 4.3 wheel 职责与有无编译器的边界

| 分发/独立工程 | 所交付内容 | 运行期职责 | 构建证据 |
|---|---|---|---|
| 主包 `flashinfer-python` | Python API、JIT generators、C++/CUDA 源码与 headers、选定 third-party headers、部分子包 manifests | API 参数检查、dispatch、plan/run、按需编译/加载 | `flashinfer-ori: pyproject.toml`；`flashinfer-ori: build_backend.py` |
| `flashinfer-jit-cache` | CUDA 版本对应的 provider 依赖与发现 shim | entry-point 发现与 manifest 验证 | `flashinfer-ori: flashinfer-jit-cache/pyproject.toml`；`flashinfer-ori: flashinfer-jit-cache/build_backend.py` |
| `flashinfer-jit-cache-provider` | 每架构 wheel 的 `.so` 集合、manifest、entry point | 提供同名 JIT module 的 AOT 实例 | `flashinfer-ori: flashinfer-jit-cache-provider/setup.py`；`flashinfer-ori: flashinfer-jit-cache-provider/build_backend.py` |
| `flashinfer-cubin` | artifact cubin 数据及目录定位函数 | 使 cubin consumer 从本地 wheel 读二进制 | `flashinfer-ori: flashinfer-cubin/pyproject.toml`；`flashinfer-ori: flashinfer-cubin/build_backend.py` |

主包 `_create_data_dir` 将 `csrc`、`include`、CUTLASS、spdlog、CCCL 放入 package data。
wheel/sdist 使用复制，editable 使用 symlink；这解释了 JIT 为何能在 wheel 安装后继续找到源代码。
Cake export 的部分子包打包 `csrc` 和 `build.json`，注释明确不打包预生成 AOT libraries。
证据：`flashinfer-ori: build_backend.py`，`_create_data_dir`、`_prepare_for_wheel`、`_prepare_for_editable`；`flashinfer-ori: pyproject.toml`，package-data。

主包 build backend 没有一个“有 NVCC 才构建整个 FlashInfer”的总开关。
它默认尝试 NIXL-EP native build，先检查 meson/ninja/nvcc/git/pkg-config/UCX/libibverbs 等依赖。
默认或 legacy `BUILD_NVEP=1` 为 best-effort，缺依赖 warning 后跳过；显式 `BUILD_NIXL_EP=1` 使缺依赖成为安装错误。
`BUILD_NIXL_EP=0` 或 `BUILD_NVEP=0` 可关闭相关构建；NCCL-EP 当前通过基础依赖 `nccl-extensions` wheel 提供。
证据：`flashinfer-ori: build_backend.py`，`_backend_enabled`、`_nixl_buildable`、`_gate_backend`、`_build_nvep_if_enabled`。

cuTile 编译链另有 best-effort 补装逻辑：有符合要求的系统 tileiras 时跳过，否则尝试安装编译侧 wheel。
`FLASHINFER_BUILD_NO_PIP=1` 禁止这些自动 pip 操作；缺少链且补装失败时，首次调用相应 cuTile kernel 才可能暴露错误。
证据：`flashinfer-ori: build_backend.py`，`_system_cuda_tile_compiler_available`、`_install_cuda_tile_compile_deps`；`flashinfer-ori: build_utils.py`，`get_cuda_tile_compile_dependency_requirements`。

不带编译器的运行环境可以消费**已有且兼容的** AOT / cubin；未覆盖特化仍需要 JIT 工具链。
`get_cuda_version` 可在 NVCC 不存在时回落到 `torch.version.cuda`，这只是版本探测 fallback，不提供 CUDA 编译能力。
证据：`flashinfer-ori: flashinfer/jit/cpp_ext.py`，`get_cuda_version`；`flashinfer-ori: flashinfer/jit/core.py`，`JitSpecNvcc.try_load`、`JitSpecNvcc.build`。

Provider wheel 使用 `cp39/abi3` tag，并检查可选 manylinux platform tag 的合法性。
shim 虽主要为元数据，也因 CPU platform 的依赖不同而标为 platform wheel。
本轮没有实际构建 wheel，因此上述结论是构建源码保证，不是 wheel 内容解压后的实测。
证据：`flashinfer-ori: flashinfer-jit-cache-provider/build_backend.py`，`PlatformSpecificBdistWheel`、`_provider_platform_tag`；`flashinfer-ori: flashinfer-jit-cache/build_backend.py`，`PlatformSpecificBdistWheel`。

### 4.4 下游消费：已证实的层次与证据边界

`README.md` 的 Adoption 列出了 SGLang、vLLM、TensorRT-LLM 等。
这个声明只证明本仓宣称被这些框架采用；它不自动证明每个新 backend 已在这些框架已发布版本集成。
证据：`flashinfer-ori: README.md`，Adoption。

| 消费场景 | 本仓可确认的 API 层 | 证据等级与边界 |
|---|---|---|
| Serving attention 通用入口 | Python `BatchDecodeWithPagedKVCacheWrapper` 构造、`plan`、`run`，调用方持有 paged-KV metadata / workspace | `flashinfer-ori: benchmarks/bench_batch_decode.py`；这是运行用例，不是下游仓库调用点证明 |
| vLLM MoE EP 集成方案 | `fleet.create_handle(HandleParams(topk_ids), knobs)` → `handle.dispatch` → 框架 expert GEMM → `handle.combine` → `handle.complete` | `flashinfer-ori: docs/design_docs/vllm_moe_ep_integration.md`；文档明确描述集成分支，不能直接外推全部 vLLM release |
| vLLM EP 契约当前实现 | `NcclEpFleet.create_handle`、`NcclEpHandle.dispatch/combine/complete`，bootstrap 可接受 process_group / native communicator | `flashinfer-ori: flashinfer/moe_ep/backends/split/comm/nccl_ep/fleet.py`；`flashinfer-ori: flashinfer/moe_ep/backends/split/comm/nccl_ep/handle.py` |
| vLLM / SGLang 风格 custom-allreduce | `flashinfer.comm.vllm_init_custom_ar`、`vllm_register_buffer`、`vllm_all_reduce`、`vllm_dispose`，另测 graph buffer IPC metadata | `flashinfer-ori: tests/comm/test_vllm_custom_allreduce.py`；开头说明 adapted from，证明兼容调用模式而非当前下游已采用 |
| SGLang 稀疏选择契约 | `top_k_page_table_transform`、`top_k_ragged_transform` 与 SGLang-style reference 对齐 | `flashinfer-ori: tests/utils/test_topk.py`，`test_compare_with_sglang_style_page_table/ragged`；是本仓对照实现 |
| vLLM / SGLang 调优缓存 | 文档记录 v1 `autotune(..., cache=...)` + `save_configs/load_configs` 的框架持久化流程；v2 改由 FlashInfer 管 store | `flashinfer-ori: docs/design_docs/autotuner_v2.md`；v2 发布迁移状态未由下游源码验证 |
| TRT-LLM | README 确认 adoption；本包提供 `trtllm_*` Python entry points、复用 TRT-LLM kernel / artifact | `flashinfer-ori: README.md`；`flashinfer-ori: flashinfer/__init__.py`；具体 TRT-LLM 发布版调用文件本轮未验证 |
| 外部非 PyTorch 消费 | `gen_act_and_mul_module().build_and_load()` 后由 `jax_tvm_ffi.register_ffi_target` 注册，再从 JAX FFI 调用 | `flashinfer-ori: docs/tutorials/jax_tvm_ffi/flashinfer_jax_tvm_ffi.py`；已读教程代码，本轮未运行 |

外部集成最常见的稳定分界是 tensor API、wrapper 状态 API、通信对象协议，而不是要求使用方直接包含每个 CUDA kernel。
这是上述实例与接口的归纳；尚不能据本仓反推下游所有算子、所有 dtype、所有 CUDA Graph 配置的使用率。
尤其 benchmark 导入 vLLM/SGLang 作性能参考，或从 TRT-LLM 移植测试，都不能反转成“该框架已调用 FlashInfer 此 API”。

### 4.5 端到端新增自定义算子清单

下面以**在现有 activation 家族新增一种逐元素 gated activation**为具体可执行路径。
所有路径都指现有参考实现；这是接入清单，本轮不执行任何源码修改、编译或发布。
如新增操作不适合该 generic kernel，可采用随后给出的独立 CUDA translation-unit 模式。

1. **先定义 Python 契约与数值 reference。**
   明确 input/out dtype、形状、是否原地写、stream、`enable_pdl`、错误条件；activation 现有约束是最后维度等于 `2*hidden_size`。
   修改/扩展参照：`flashinfer-ori: flashinfer/activation.py` 的 `_check_act_and_mul_input`、`_check_shape`、`silu_and_mul`。
   reference 测试参照：`flashinfer-ori: tests/utils/test_activation.py` 的 `test_fused_silu_mul`。

2. **实现 device math，并检查 generic kernel 能否复用。**
   在 `act_func_def_str` 增加唯一 activation 名称到 CUDA device function 的映射；现有 `silu_def_cu_str`/`gelu_def_cu_str` 是模板。
   generic `act_and_mul_kernel` 负责读取输入两半、调用 activation 再乘法。
   证据：`flashinfer-ori: flashinfer/jit/activation.py`；`flashinfer-ori: include/flashinfer/activation.cuh`。

3. **生成 CUDA host launcher 与 FFI binding。**
   `get_act_and_mul_cu_str` 渲染 `activation_templ`；模板选择 vector width、获取当前 stream、launch kernel、检查 CUDA error。
   同一模板用 `TVM_FFI_DLL_EXPORT_TYPED_FUNC` export `<act_name>_and_mul`，不需为此模板额外引入 pybind11。
   证据：`flashinfer-ori: flashinfer/jit/activation.py`，`activation_templ`；`flashinfer-ori: csrc/tvm_ffi_utils.h`。

4. **接入 JIT spec 与稳定 module name。**
   `gen_act_and_mul_module(act_func_name)` 用 `write_if_different` 写生成 CUDA 源文件，再 `gen_jit_spec(name, sources)`。
   新的 activation 名称自然形成独立模块；如引入新的编译期 knob，必须同步纳入 URI / generator 参数，避免错误复用。
   证据：`flashinfer-ori: flashinfer/jit/activation.py`，`gen_act_and_mul_module`；`flashinfer-ori: flashinfer/jit/core.py`，`gen_jit_spec`。

5. **把共享库变成 Python 可调用对象。**
   复用 `get_act_and_mul_module` 的 `@functools.cache`，调用 `.build_and_load()`，按 export 名 `getattr(module, fname)` 取得 FFI function。
   `JitSpecNvcc.load` 内部执行 `tvm_ffi.load_module`；Python closure 把 torch Tensor 传给 `TensorView` FFI launcher。
   证据：`flashinfer-ori: flashinfer/activation.py`，`get_act_and_mul_module`；`flashinfer-ori: flashinfer/jit/core.py`，`JitSpecNvcc.load`。

6. **建立 op / fake 契约，并单独确认注册是否真实生效。**
   activation 现有样式为 `register_custom_op(..., mutates_args=("out",))` + `register_fake_op`，fake 对无返回的 out op 为空实现。
   **本基线的这两个 helper 均为 no-op decorator**，不能仅照抄装饰器就承诺 `torch.compile` 已兼容。
   需要真正 `torch.library` 节点时，可参照直接使用 `torch.library.custom_op` 与 `torch.library.register_fake` 的 `reswizzle_sf`，定义真实 mutation schema 与 fake shape/dtype/device 推导。
   证据：`flashinfer-ori: flashinfer/utils.py`，`register_custom_op`、`register_fake_op`；`flashinfer-ori: flashinfer/fused_moe/utils.py`，`reswizzle_sf`。

7. **封装公开函数、日志和顶层 export。**
   仿 `silu_and_mul` 检查输入、允许调用者提供 out、否则分配 output，再调用缓存 module。
   用 `flashinfer_api` 接入日志；如需 trace 描述，参照已有 activation trace template；按需要在顶层显式 export。
   证据：`flashinfer-ori: flashinfer/activation.py`；`flashinfer-ori: flashinfer/trace/templates/activation.py`；`flashinfer-ori: flashinfer/__init__.py`。

8. **测试真实边界，而非只测 symbol 可 import。**
   扩展 activation tests 的奇数 hidden size、batch/sequence、PDL、数值 reference 覆盖，并增加 out mutation 与非法 shape 检查。
   若本次承诺 compile/graph 支持，应验证 eager、fake、compile、CUDA Graph replay 的一致性；不能把 helper decorator 存在当成测试通过。
   已有 warmup fixture 展示 `build_jit_specs` 预热；这些是接入时应执行的检查，本轮未执行 GPU 测试。
   证据：`flashinfer-ori: tests/utils/test_activation.py`，`warmup_jit`、三个 `test_fused_*`。

9. **纳入 AOT 枚举，检查特化遗漏。**
   activation 的 `add_act` 分支已经遍历 `act_func_def_str`，新增映射会进入 AOT；全新算子家族则需在 `gen_all_modules` 加入对应 generator 及架构 gate。
   调整必要的默认组合后由 `compile_and_package_modules` 统一构建；不要把仅开发机 warmup 的本地 cache 当成交付结果。
   证据：`flashinfer-ori: flashinfer/aot.py`，`gen_all_modules`、`get_default_config`、`compile_and_package_modules`。

10. **确认 wheel 源码、provider manifest 与离线命中。**
    generator 在主包内可被发现；其 source/header 必须在 package-data 中；provider 构建后 manifest 必须出现新增模块 name。
    在匹配版本/SM 的干净环境关闭 JIT 后运行该 API，验证会命中 AOT；缺模块时 `MissingJITCacheError` 提供诊断。
    证据：`flashinfer-ori: pyproject.toml`；`flashinfer-ori: flashinfer-jit-cache-provider/build_backend.py`，`_write_provider_manifest`；`flashinfer-ori: tests/jit/test_jit_cache_providers.py`；`flashinfer-ori: flashinfer/jit/core.py`。

独立 CUDA 文件模式的完整参考链为：
`quantization.cuh::PackBits` → `quantization.cu::packbits(TensorView,...)` → `flashinfer_quantization_binding.cu::TVM_FFI_DLL_EXPORT_TYPED_FUNC` → `gen_quantization_module` → `get_quantization_module().packbits` → public `packbits` → AOT `add_misc`。
证据依次为：`flashinfer-ori: include/flashinfer/quantization.cuh`；`flashinfer-ori: csrc/quantization.cu`；`flashinfer-ori: csrc/flashinfer_quantization_binding.cu`；`flashinfer-ori: flashinfer/jit/quantization.py`；`flashinfer-ori: flashinfer/quantization/packbits.py`；`flashinfer-ori: flashinfer/aot.py`。
reference 测试见 `flashinfer-ori: tests/utils/test_quantization.py`，`numpy_packbits_ref`、`test_packbits`、`test_segment_packbits`。

还存在无需改主仓 kernel 的 attention variant 扩展接口。
`test_single_decode_mask` 提供 `AttentionVariantBase` 子类字符串、额外 tensor/scalar 名称和 dtype，交给 `gen_customize_single_decode_module(...).build_and_load()`。
随后 `single_decode_with_kv_cache_with_jit_module` 接收该 module；batch 版本经 wrapper 的 `jit_args` 注入并继续使用 `plan/run`。
`test_flash_sigmoid` 展示替换 logits transform，`test_batch_prefill_variant_owns_mask` 展示 variant 自己拥有 mask 语义。
证据：`flashinfer-ori: tests/utils/test_jit_example.py`；`flashinfer-ori: flashinfer/jit/attention/modules.py`，`gen_customize_*`；`flashinfer-ori: flashinfer/decode.py`；`flashinfer-ori: flashinfer/prefill.py`。

这条外部 extension API 并不自动给客户任意 URI 生成官方 AOT provider。
要离线分发定制组合，仍需将 spec 加入自己的 AOT 构建/打包清单；这里是根据 `gen_all_modules` 显式枚举机制得出的工程要求。
证据：`flashinfer-ori: flashinfer/aot.py`，`gen_all_modules`；`flashinfer-ori: flashinfer-jit-cache-provider/build_backend.py`，`_build_aot_modules`。

## 5. 可供昇腾侧对标的基准清单

表中 ID 是 R2/R3 的稳定差分主键。R2 建议为每行追加“昇腾入口/实现”“已支持组合”“缺口”“验证证据”，
同时引用 §2 能力域，不以一个 backend 或算子存在替代整个机制已经对齐。
本表“上游实现”仍是本 commit 的静态事实，测试文件存在不代表本轮执行通过。

| 机制 | 上游实现 | 关键文件 | 对标要点 |
|---|---|---|---|
| JI-01 特化规格 | `JitSpec` 统一 try_load/build/load；`JitSpecNvcc` 保存源与 flags | `flashinfer-ori: flashinfer/jit/core.py` | 是否有独立于具体算子的构建契约；能否扩展工具链 |
| JI-02 模板生成 | Jinja config/kernel instance + 静态 launcher/binding；按内容差异写文件 | `flashinfer-ori: flashinfer/jit/attention/modules.py`；`flashinfer-ori: flashinfer/jit/utils.py`；`flashinfer-ori: csrc/batch_decode_kernel_inst.jinja` | dtype/shape/variant 特化如何表达和复用；避免重复生成 |
| JI-03 编译缓存身份 | op URI 编码 dtype/head dim/功能开关；版本与排序架构隔离 workspace | `flashinfer-ori: flashinfer/jit/attention/modules.py`；`flashinfer-ori: flashinfer/jit/env.py` | 特化字段完备性、同名冲突、架构与版本隔离 |
| JI-04 增量有效性 | NVCC 本地 `.so` 仍通过 Ninja source/header/command 依赖检查 | `flashinfer-ori: flashinfer/jit/core.py`；`flashinfer-ori: flashinfer/jit/cpp_ext.py` | 源/头/编译参数变更是否失效；不能只检查文件存在 |
| JI-05 编译并发与去重 | per-module FileLock+复查、进程 getter cache、Ninja并行、NVCC threads | `flashinfer-ori: flashinfer/jit/core.py`；`flashinfer-ori: flashinfer/jit/cpp_ext.py`；`flashinfer-ori: flashinfer/decode.py` | 多worker首用不重复/不竞争；编译资源是否可限制 |
| JI-06 批量预热 | `build_jit_specs` 合并 subninja，单独 prebuild jobs 设置 | `flashinfer-ori: flashinfer/jit/core.py` | 能否按清单并行预热，明确支持哪类 spec |
| JI-07 架构上下文 | `CompilationContext` 正规化目标、显式环境覆盖、能力过滤 | `flashinfer-ori: flashinfer/compilation_context.py` | SoC/设备目标、工具链能力与部署目标的区分 |
| JI-08 Python callable 形成 | `TVM_FFI_DLL_EXPORT_TYPED_FUNC` → `tvm_ffi.load_module` → module属性/适配器 | `flashinfer-ori: csrc/batch_decode_jit_binding.cu`；`flashinfer-ori: flashinfer/jit/core.py` | ABI、Tensor/stream传递、错误边界与Python绑定步骤 |
| JI-09 DSL持久缓存 | CuTe对象+架构/编译栈/源摘要meta；原子提交，失效重编译 | `flashinfer-ori: flashinfer/jit/cute_dsl_core.py` | triton-ascend/其他DSL是否有独立可核查的失效与持久化规则 |
| JI-10 生成程序身份 | Cake source/binding/target/options 摘要进module名 | `flashinfer-ori: flashinfer/jit/cake_program.py` | 自动生成算子接入、数值编译选项、内容寻址 |
| JI-11 NVRTC嵌入路线 | 构建期产cubin并核验receipt，再嵌入host object | `flashinfer-ori: flashinfer/jit/flash_kda_nvrtc.py`；`flashinfer-ori: flashinfer/jit/cpp_ext.py` | 是否有离线编译设备代码与host ABI组合的等价能力 |
| JI-12 懒加载与capture预热 | 独立stride module首次eager加载，capture前显式prewarm | `flashinfer-ori: flashinfer/jit/attention/_lazy.py` | 冷路径何时触发；异步worker和建图前准备是否可控 |
| AO-01 AOT清单粒度 | dtype/head dim/位置编码等矩阵加架构特定generator，非所有组合 | `flashinfer-ori: flashinfer/aot.py` | 默认预编译覆盖清单是否可枚举，漏项如何发现 |
| AO-02 wheel职责切分 | 主包源码/API、cache shim、SM provider `.so`、cubin数据 | `flashinfer-ori: pyproject.toml`；`flashinfer-ori: flashinfer-jit-cache`；`flashinfer-ori: flashinfer-jit-cache-provider`；`flashinfer-ori: flashinfer-cubin` | 安装体积、目标芯片拆包、按需依赖和离线交付 |
| AO-03 provider发现 | shim发现provider，manifest声明module/架构，检查版本 | `flashinfer-ori: flashinfer/jit/env.py`；`flashinfer-ori: flashinfer-jit-cache/flashinfer_jit_cache/__init__.py` | provider扩展点、版本矩阵、是否可同时安装多目标包 |
| AO-04 AOT优先级 | 内嵌AOT优先；按设备兼容性选provider；失败回退JIT | `flashinfer-ori: flashinfer/jit/env.py`；`flashinfer-ori: flashinfer/jit/core.py` | miss/unloadable/不兼容是否可诊断；回退策略是否可配置 |
| AO-05 异构设备分发 | `_HeterogeneousAOTModule` 按Tensor/当前设备分发并按需fallback | `flashinfer-ori: flashinfer/jit/core.py`；`flashinfer-ori: flashinfer/jit/_cuda_architecture.py` | 多型号设备混布时是否会加载错误目标二进制 |
| AO-06 严格预编译运行 | NVCC禁用JIT只接受AOT/provider，已有本地JIT `.so` 仍可能报缺失 | `flashinfer-ori: flashinfer/jit/core.py` | 无编译器环境、覆盖检查、意外编译可观测性；DSL例外需另查 |
| AO-07 artifact版本与完整性 | 默认发布pin+manifest hash+文件hash；按需路径可显式关闭校验 | `flashinfer-ori: flashinfer/artifacts.py`；`flashinfer-ori: flashinfer/jit/cubin_loader.py` | 设备二进制版本来源、校验、防旧缓存混入 |
| AO-08 下载并发与离线 | 文件锁+复查+原子replace、retry、线程池；NO_DOWNLOAD独立控制 | `flashinfer-ori: flashinfer/jit/cubin_loader.py`；`flashinfer-ori: flashinfer/artifacts.py` | 多进程下载、断网、mirror、本地包优先级与错误反馈 |
| AO-09 cubin与host组合 | ctypes回调把带hash的cubin交还native loader | `flashinfer-ori: flashinfer/jit/cubin_loader.py`；`flashinfer-ori: include/flashinfer/cubin_loader.h` | host launcher与设备代码能否独立交付及一致升级 |
| RT-01 注册辅助器有效性 | 当前 `register_custom_op` / `register_fake_op` 都为 identity | `flashinfer-ori: flashinfer/utils.py` | 区分声明意图与有效 dispatcher 注册，不能只比装饰器数量 |
| RT-02 有效 custom op 与 fake | FP4 量化直接注册，fake 返回布局一致的量化值/scale 元数据 | `flashinfer-ori: flashinfer/quantization/fp4_quantization.py` | functional/in-place、shape、dtype、alias 契约是否完整 |
| RT-03 compile 真实边界 | attention 回归用例在外层自行注册 op 后 fullgraph compile | `flashinfer-ori: tests/attention/test_batch_decode_kernels.py` | 区分直接 compile、外层包装、graph break 三种支持级别 |
| RT-04 图模式 wrapper | decode/prefill/POD/cascade/MLA 明确图模式资源接口 | `flashinfer-ori: flashinfer/decode.py`；`flashinfer-ori: flashinfer/prefill.py`；`flashinfer-ori: flashinfer/mla/_batch_mla/_wrapper.py` | 列出支持 wrapper、后端、dtype、shape，不只给布尔结论 |
| RT-05 图 metadata 更新 | plan 捕获外对固定 CSR/mask buffers copy_，replay 读新值 | `flashinfer-ori: flashinfer/prefill.py` | 指针稳定、容量上限、固定 batch 与动态序列长度 |
| RT-06 图计划容量 | planner padding、valid mask 与 offsets 固定图调度资源 | `flashinfer-ori: include/flashinfer/attention/scheduler.cuh` | 图规模和设备调度元数据是否一起稳定 |
| RT-07 图重规划限制 | MLA 根据 `supports_cuda_graph_replan` 拒绝不支持的二次 plan | `flashinfer-ori: flashinfer/mla/_batch_mla/_wrapper.py`；`flashinfer-ori: flashinfer/mla/_batch_mla/_backends/_capabilities.py` | 后端迁移、形状变化、重新捕获规则是否显式 |
| RT-08 piecewise 集成边界 | MoE 区域 capture/replay 测试；全局 flag 未找到内部调用点 | `flashinfer-ori: tests/moe/test_cute_dsl_bf16_moe.py`；`flashinfer-ori: flashinfer/fused_moe/utils.py` | 核区域可捕获与框架切图管理分开评估 |
| RT-09 后端支持检查 | `backend_requirement`、架构 checker、experimental 过滤 | `flashinfer-ori: flashinfer/utils.py` | 支持矩阵能否查询；不支持是否提前报错 |
| RT-10 局部 auto 策略 | GEMM heuristic 与 MLA 架构/工作量策略独立 | `flashinfer-ori: flashinfer/gemm/gemm_base.py`；`flashinfer-ori: flashinfer/mla/_batch_mla/_auto_policy.py` | 逐 API 比顺序、条件、fallback；不能只比 backend 名称 |
| RT-11 运行期 autotune | runner/tactic/profile、cache miss 测量、普通调用 fallback | `flashinfer-ori: flashinfer/autotuner/autotuner.py` | profiling 生命周期、动态 shape bucket、正确性与性能隔离 |
| RT-12 调优缓存 identity | op/runner/profile/extras；磁盘剔除进程 identity；环境版本校验 | `flashinfer-ori: flashinfer/autotuner/autotuner.py` | shape 外参数、版本升级失效和跨进程复用 |
| RT-13 managed autotune | v2 按环境和 entry 散列、原子发布、tune/replay 分离 | `flashinfer-ori: flashinfer/autotune_cache.py` | 缓存目录治理、损坏回退、测量政策与持久化 |
| RT-14 调优/捕获边界 | active outer capture 中禁止开始 profiling，要求 warmup | `flashinfer-ori: flashinfer/autotuner/autotuner.py` | 首次编译/调优是否可在建图前完成 |
| RT-15 两阶段执行 | module plan→C++ planner→plan vector；run 恢复 offset 并执行 | `flashinfer-ori: flashinfer/decode.py`；`flashinfer-ori: csrc/batch_decode.cu`；`flashinfer-ori: include/flashinfer/attention/scheduler.cuh` | 状态持有、调度准备、多个层复用及失效条件 |
| RT-16 workspace 管理 | caller-owned workspace、pinned/device int staging、全局 scratch 扩容保留旧引用 | `flashinfer-ori: flashinfer/decode.py`；`flashinfer-ori: flashinfer/utils.py` | 资源所有权、容量估算、复用、生命周期与并发限制 |
| EX-01 源码与依赖交付 | 主wheel复制source/header，editable链接；可选native构建有best-effort/strict gate | `flashinfer-ori: pyproject.toml`；`flashinfer-ori: build_backend.py` | 安装后自定义编译、无工具链安装、必要与可选依赖是否区分 |
| EX-02 Python导出与lazy | 大量eager import、选定__getattr__ export、首次调用加载module | `flashinfer-ori: flashinfer/__init__.py`；`flashinfer-ori: flashinfer/activation.py` | 稳定入口、可选依赖失败、首次调用成本与能力探测 |
| EX-03 CLI与诊断 | wheel安装/下载、环境、module状态、compile commands、dump replay | `flashinfer-ori: flashinfer/__main__.py` | 开发部署可观测性；空registry不应误认为无实现 |
| EX-04 自定义普通算子 | device math→TensorView launcher→TVM export→JitSpec→Python→测试→AOT | `flashinfer-ori: flashinfer/jit/activation.py`；`flashinfer-ori: flashinfer/activation.py`；`flashinfer-ori: tests/utils/test_activation.py`；`flashinfer-ori: flashinfer/aot.py` | 新增文件/注册点、契约测试、分发接入是否能走完整流程 |
| EX-05 外部attention variant | 额外Tensor/scalar、variant声明、jit_args，复用wrapper plan/run | `flashinfer-ori: tests/utils/test_jit_example.py`；`flashinfer-ori: flashinfer/jit/attention/modules.py` | 外部客户能否不改核心调度器扩展mask/score/output语义 |
| EX-06 通信集成生命周期 | Fleet持久资源、Handle逐步dispatch/combine/complete | `flashinfer-ori: flashinfer/moe_ep/backends/split/comm/nccl_ep/fleet.py`；`flashinfer-ori: flashinfer/moe_ep/backends/split/comm/nccl_ep/handle.py` | 框架资源所有权、communicator/bootstrap与每步状态 |
| EX-07 日志与设备profiler | flashinfer_api的日志/trace入口；设备事件转Perfetto | `flashinfer-ori: flashinfer/api_logging.py`；`flashinfer-ori: flashinfer/profiler/__init__.py`；`flashinfer-ori: include/flashinfer/profiler.cuh` | 错误重放、可观测开关、kernel内事件及开销边界 |
| EX-08 外部方案替换 | TraceTemplate/Solution描述；trace_apply载入Python/C++方案 | `flashinfer-ori: flashinfer/trace/solution.py`；`flashinfer-ori: flashinfer/trace_apply/loaders/cpp.py`；`flashinfer-ori: flashinfer/trace_apply/apply.py` | 可否按问题规格接入客户实现并验证替换契约 |

表内的“未验证”不会被R2自动当成昇腾缺口；应先比较静态机制，再为双方建立同口径运行验证。
§2的CAP编号用于算子能力差分，§5的JI/AO/RT/EX编号用于架构机制差分，两者应交叉引用。

## 6. 未验证项与调研局限

| 未验证项 | 已确认到的边界 | 本轮无法下结论的原因 / 后续验证方式 |
|---|---|---|
| GPU 数值正确性与性能 | 可定位 public API、backend、测试与benchmark | 本轮只做静态调研，未执行CUDA kernel；应按GPU/dtype/shape/后端跑参数矩阵 |
| 全部声明组合是否可构建 | JIT生成器与AOT默认枚举可追踪 | 未调用AOT builders、未编译 `.so`/DSL对象；存在入口不保证所有组合编译成功 |
| wheel实际内容及公开发布状态 | 构建脚本声明的职责、manifest、entry point、安装布局可核查 | 未构建或解包发布wheel，未查询包索引；不把源码能力等同于PyPI某个版本已交付 |
| 远端artifact可达性 | 发布pin、manifest摘要、下载/校验流程可读 | 未联网获取 artifact；无法证明所有pin仍可下载或镜像同步完整 |
| 第三方库内部实现 | `.gitmodules`和构建/include引用可确认依赖意图 | `3rdparty` 子模块目录在本次检出没有文件内容，不能对其中kernel作源码级结论 |
| vLLM/SGLang/TRT-LLM发布版调用点 | README adoption、局部集成设计、比较benchmark、模拟契约测试有不同级别证据 | 未检出/研究这些框架对应版本；尤其“对照某框架”不能证明该框架实际消费此API |
| 全包torch.compile兼容性 | no-op helper与直接注册路线、外层op包装测试已分清 | 未运行compile测试；不能从fake函数存在推导全包fullgraph可追踪 |
| piecewise graph全局调度 | MoE捕获区域测试和flag定义存在 | 未发现该flag的仓内消费点；下游切图、生命周期和端到端效果未验证 |
| 并发与异构运行实效 | 文件锁、线程锁、per-device AOT分发、workspace限制有实现 | 未做多进程冷启动/多GPU混布/多stream压力测试；抽象存在不代表所有路径并发安全 |
| 实验后端稳定性 | experimental目录及显式auto opt-in策略可定位 | 不构成兼容性承诺；型号专用、模型专用和训练辅助路径需分别建立成熟度标记 |
| 自定义算子完整接入耗时 | §4给出可追踪步骤和已有范例 | 本轮禁止新增代码，因此未实现一个真实新算子验证全流程；建议R2以最小样例演练 |

范围限制与证据规则：

- 此报告只把用户指定 commit 作为“上游”基线，不额外认证它与某个公开 release/tag 完全一致。
- 目录普查覆盖全部 Python 子目录；能力表按域归并。深层目录存在性不等于逐个kernel函数已经审阅。
- 典型调用链已回到源码；测试文件用于说明现成验证入口，没有将其存在写成“本轮通过测试”。
- 不用自动生成数量或代码行数替代功能支持度；R2应沿§5主键核对语义、组合限制、交付与验证。
- 未调研昇腾本仓的实现、其他本地参考仓库，也未开展R2/R3的缺口判断。
- 未将仓库中的大篇幅说明文件直接复制为机制结论；核心机制以本报告列出的实现文件为证据。

交付自检口径：源码引用以 `flashinfer-ori:` 为前缀，可从Markdown机械抽取并检查存在性；
§2目录覆盖表用于对照实际目录集合，§5所有证据路径纳入同一检查。
代码框中的cache/wheel布局是明确标注的生成模式，不冒充当前源码检出中的实体文件。
本轮验证不需要import FlashInfer，避免只读调研触发JIT、下载或cache写入。
