T10 交付报告 · 2026-10-23

完成 plan 期的 FIA v1 最大 workspace 查询及实际使用，接在 T9 的 caller-owned byte workspace、AttentionWorkspaceContract 和两阶段 plan 发布上。本仓没有捕获或重放图；修改留在工作区，没有 commit/push。没有修改 docs/tasks/、docs/TASKS.md、execution_identity.py 或 planner.py。

环境：`/root/miniforge3/envs/flashinfer-a5/bin/python`，torch `2.10.0+cpu`，torch_npu `2.10.0.post7`，设备 `Ascend950PR_9579`。

**先给接口实测。** 两个函数的 inspect.signature 都是通用包装器签名 `(*args: _P.args, **kwargs: _P.kwargs) -> ~_T`，不能据此猜参数。下方 dispatcher schema 是本机真实可调用签名；v1 docstring 漏了 schema 中的 rope 参数。共同输入为 FP16 `q=(1,1,8,128)`、`k/v=(1,1024,2,128)`、BSND、8/2 heads、scale=128**-0.5，真实返回均为 NPU Tensor：`shape=(16777216,)`、`dtype=float16`、element_size=2、实际容量 `33554432` 字节。不是整数，也不能把 numel 当作字节数。

命令：

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python tools/t10_max_workspace_audit.py --interfaces
```

完整实测 schema、返回元数据和参数敏感性输出：

```text
torch: 2.10.0+cpu torch_npu: 2.10.0.post7
NPU: Ascend950PR_9579
_npu_fused_infer_attention_score_get_max_workspace inspect.signature: (*args: _P.args, **kwargs: _P.kwargs) -> ~_T
dispatcher: npu::_npu_fused_infer_attention_score_get_max_workspace(Tensor query, Tensor key, Tensor value, *, Tensor? pse_shift=None, Tensor? atten_mask=None, SymInt[]? actual_seq_lengths=None, SymInt[]? actual_seq_lengths_kv=None, Tensor? dequant_scale1=None, Tensor? quant_scale1=None, Tensor? dequant_scale2=None, Tensor? quant_scale2=None, Tensor? quant_offset2=None, Tensor? antiquant_scale=None, Tensor? antiquant_offset=None, Tensor? key_antiquant_scale=None, Tensor? key_antiquant_offset=None, Tensor? value_antiquant_scale=None, Tensor? value_antiquant_offset=None, Tensor? block_table=None, Tensor? query_padding_size=None, Tensor? kv_padding_size=None, Tensor? key_shared_prefix=None, Tensor? value_shared_prefix=None, SymInt[]? actual_shared_prefix_len=None, Tensor? query_rope=None, Tensor? key_rope=None, Tensor? key_rope_antiquant_scale=None, int num_heads=1, float scale=1., int pre_tokens=2147483647, int next_tokens=2147483647, str input_layout="BSH", int num_key_value_heads=0, int sparse_mode=0, int inner_precise=0, int block_size=0, int antiquant_mode=0, int key_antiquant_mode=0, int value_antiquant_mode=0, bool softmax_lse_flag=False) -> Tensor
return: Tensor shape= (16777216,) dtype= torch.float16 device= npu:0 numel= 16777216 element_size= 2 bytes= 33554432 data_ptr= 19792501145600
parameter_probe: _npu_fused_infer_attention_score_get_max_workspace kv4096 dtype= torch.float16 numel= 17956864 bytes= 35913728
parameter_probe: _npu_fused_infer_attention_score_get_max_workspace kv1024 dtype= torch.float16 numel= 16777216 bytes= 33554432
parameter_probe: _npu_fused_infer_attention_score_get_max_workspace q128 dtype= torch.float16 numel= 20611072 bytes= 41222144
parameter_probe: _npu_fused_infer_attention_score_get_max_workspace dim64 dtype= torch.float16 numel= 16777216 bytes= 33554432
parameter_probe: _npu_fused_infer_attention_score_get_max_workspace heads4 dtype= torch.float16 numel= 17956864 bytes= 35913728
parameter_probe: _npu_fused_infer_attention_score_get_max_workspace kv_heads4 dtype= torch.float16 numel= 18399232 bytes= 36798464
parameter_probe: _npu_fused_infer_attention_score_get_max_workspace bf16 dtype= torch.bfloat16 numel= 17956864 bytes= 35913728
parameter_probe: _npu_fused_infer_attention_score_get_max_workspace lse dtype= torch.float16 numel= 17956864 bytes= 35913728
parameter_probe: _npu_fused_infer_attention_score_get_max_workspace mask dtype= torch.float16 numel= 16777216 bytes= 33554432
parameter_probe: _npu_fused_infer_attention_score_get_max_workspace scale0.5 dtype= torch.float16 numel= 17956864 bytes= 35913728
parameter_probe: _npu_fused_infer_attention_score_get_max_workspace actual_kv128 dtype= torch.float16 numel= 91109376 bytes= 182218752
parameter_probe: _npu_fused_infer_attention_score_get_max_workspace BNSD dtype= torch.float16 numel= 17956864 bytes= 35913728
_npu_fused_infer_attention_score_v2_get_max_workspace inspect.signature: (*args: _P.args, **kwargs: _P.kwargs) -> ~_T
dispatcher: npu::_npu_fused_infer_attention_score_v2_get_max_workspace(Tensor query, Tensor key, Tensor value, *, Tensor? query_rope=None, Tensor? key_rope=None, Tensor? pse_shift=None, Tensor? atten_mask=None, SymInt[]? actual_seq_qlen=None, SymInt[]? actual_seq_kvlen=None, Tensor? block_table=None, Tensor? dequant_scale_query=None, Tensor? dequant_scale_key=None, Tensor? dequant_offset_key=None, Tensor? dequant_scale_value=None, Tensor? dequant_offset_value=None, Tensor? dequant_scale_key_rope=None, Tensor? quant_scale_out=None, Tensor? quant_offset_out=None, Tensor? quant_scale_p=None, Tensor? learnable_sink=None, int num_query_heads=1, int num_key_value_heads=0, float softmax_scale=1., int pre_tokens=2147483647, int next_tokens=2147483647, str input_layout="BSH", int sparse_mode=0, int block_size=0, int query_quant_mode=0, int key_quant_mode=0, int value_quant_mode=0, int inner_precise=0, bool return_softmax_lse=False, int? query_dtype=None, int? key_dtype=None, int? value_dtype=None, int? query_rope_dtype=None, int? key_rope_dtype=None, int? key_shared_prefix_dtype=None, int? value_shared_prefix_dtype=None, int? dequant_scale_query_dtype=None, int? dequant_scale_key_dtype=None, int? dequant_scale_value_dtype=None, int? dequant_scale_key_rope_dtype=None, int? out_dtype=None) -> Tensor
return: Tensor shape= (16777216,) dtype= torch.float16 device= npu:0 numel= 16777216 element_size= 2 bytes= 33554432 data_ptr= 19792501145600
parameter_probe: _npu_fused_infer_attention_score_v2_get_max_workspace kv4096 dtype= torch.float16 numel= 17956864 bytes= 35913728
parameter_probe: _npu_fused_infer_attention_score_v2_get_max_workspace kv1024 dtype= torch.float16 numel= 16777216 bytes= 33554432
parameter_probe: _npu_fused_infer_attention_score_v2_get_max_workspace q128 dtype= torch.float16 numel= 20611072 bytes= 41222144
parameter_probe: _npu_fused_infer_attention_score_v2_get_max_workspace dim64 dtype= torch.float16 numel= 16777216 bytes= 33554432
parameter_probe: _npu_fused_infer_attention_score_v2_get_max_workspace heads4 dtype= torch.float16 numel= 17956864 bytes= 35913728
parameter_probe: _npu_fused_infer_attention_score_v2_get_max_workspace kv_heads4 dtype= torch.float16 numel= 18399232 bytes= 36798464
parameter_probe: _npu_fused_infer_attention_score_v2_get_max_workspace bf16 dtype= torch.bfloat16 numel= 17956864 bytes= 35913728
parameter_probe: _npu_fused_infer_attention_score_v2_get_max_workspace lse dtype= torch.float16 numel= 17956864 bytes= 35913728
parameter_probe: _npu_fused_infer_attention_score_v2_get_max_workspace mask dtype= torch.float16 numel= 16777216 bytes= 33554432
parameter_probe: _npu_fused_infer_attention_score_v2_get_max_workspace scale0.5 dtype= torch.float16 numel= 17956864 bytes= 35913728
parameter_probe: _npu_fused_infer_attention_score_v2_get_max_workspace actual_kv128 dtype= torch.float16 numel= 91109376 bytes= 182218752
parameter_probe: _npu_fused_infer_attention_score_v2_get_max_workspace BNSD dtype= torch.float16 numel= 17956864 bytes= 35913728
```

参数探针以 `q=(1,1,8,128)`、`k/v=(1,4096,2,128)`、FP16、BSND、不带 mask/LSE/actual lengths 为基准，每次只更改列出的维度。两套接口在这组探针中结果一致：

| 改动 | 返回 Tensor 的实际字节数 |
|---|---:|
| 基准 | 35,913,728 |
| KV shape 的 S 改为 1024 | 33,554,432 |
| Q shape 的 S 改为 128 | 41,222,144 |
| head_dim 改为 64 | 33,554,432 |
| num_heads 改为 4 | 35,913,728 |
| num_key_value_heads 改为 4 | 36,798,464 |
| dtype 改为 BF16 | 35,913,728，返回 dtype 同样改为 BF16 |
| 开启 LSE | 35,913,728 |
| 传入同形状 bool mask | 33,554,432 |
| scale 改为 0.5 | 35,913,728 |
| 增加 actual_seq_lengths_kv=[128] / v2 actual_seq_kvlen=[128] | 182,218,752 |
| 改为等价 BNSD shape/layout | 35,913,728 |

可见 Q/K/V shape、head_dim、KV heads、mask 有无、实际长度参数确实能改变结果；dtype 决定返回 Tensor 类型和字节换算。此次只说这些实测配置中观察到的差异，不声称未变化的参数在所有配置里都无影响。量化、rope、共享前缀、block_table、sparse 等扩展参数按完整 schema 存在，但本仓当前 dense composite 未走到对应算子路径，所以没有为它们造接线。

真实调用链是 batch wrapper → registry 的 batch FIA v1 runner → 按请求收集 dense KV → BSND `npu_fused_infer_attention_score`。公共 paged block_tables 在当前实现中参与 gather，未作为 FIA 的 block_table 参数传入。v2 只在本次接口探针中调用，仓库没有 v2 runner，本次也未新增。参考的 [vllm-ascend 调用点](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/attention/attention_v1.py) 同样把最大查询返回的 Tensor 交给 `.out(workspace=...)`；[Ascend op-plugin 的对应实现](https://github.com/Ascend/op-plugin/blob/6935149b5c29e658c6900ca4b53820396ee84fb0/op_plugin/ops/opapi/FusedInferAttentionScoreKernelNpuOpApi.cpp) 可用于交叉核对返回分配与 .out 存储地址用法。设计和字节数以本机实测为准。

**开关和资源职责。** 三种公开 batch wrapper 构造函数新增仅关键字参数 `use_max_workspace: bool = False`；只有 `use_cuda_graph=True` 且显式开启才在 plan 中自动查询。默认 False 保留现有 T9 调用方的小 buffer、按需执行和内存成本；本机常驻 workspace 的实测增幅约 1.79–2 倍，适合由调用方决定。只开 use_cuda_graph 仍是原有 T9 资源/冻结契约，不自动获得 T10 最大容量保证。Host reference 和外部 provider 保持原路由，非图 plan 即使设置 use_max_workspace=True 也不执行最大查询。

首次启用的 plan 分别用真实连续 Q/K/V prototype 查询 LSE 关闭和开启的需求，按字节取所有顺序执行请求的最大值，不求和。prototype 与临时返回 Tensor 都是查询期暂存，返回后不作为另一套持久 workspace 持有。持久存储始终是构造时传入的 `_float_workspace_buffer`，调用方在捕获前一次分配，plan 校验容量，run 把原 tensor 交给 FIA `.out`，无 workspace 换址/自动扩容。int workspace 需求为 0。

相同静态算子配置下，KV seq_len 变化重用已绑定最大需求，不再查询。T9 原来允许的 dtype/heads/维度/mask 有无及 prefill Q lengths 变化会重新查询，但继续校验同一 buffer 的容量；没有增加冻结维度。需求变大而容量不足时，在任何 metadata copy、freeze、plan 发布之前报原 workspace 容量错误；已有 plan/contract/cache key 保持不变。T9 的 batch/backend/q_len_per_req/block table/kernel 冻结维度和错误消息没有改动。

`workspace_size(...)` 的已有 decode/paged prefill 入口现在支持显式 native 最大容量查询，即使 wrapper 是 eager 或 workspace 为空。它返回 `(float_bytes, 0)`，不发布 plan、不写固定 metadata、不冻结 session。它是调用方主动查询，不会使所有 eager plan 支付成本。原有 ragged wrapper 没有 workspace_size 方法，本次没有伪造该接口；ragged 的 graph plan 同样接入最大查询。

调用示意：

```python
# meta/heads/dim/page_size 与后续计划相同；fixed_* 由调用方预先分配。
probe = BatchDecodeWithPagedKVCacheWrapper(torch.empty(0, dtype=torch.uint8, device="npu:0"))
float_bytes, int_bytes = probe.workspace_size(indptr, indices, last_page_len, heads, kv_heads, dim, page_size)
workspace = torch.empty(float_bytes, dtype=torch.uint8, device="npu:0")
wrapper = BatchDecodeWithPagedKVCacheWrapper(
    workspace, use_cuda_graph=True, use_max_workspace=True,
    paged_kv_indptr_buffer=fixed_indptr,
    paged_kv_indices_buffer=fixed_indices,
    paged_kv_last_page_len_buffer=fixed_last_page_len,
)
wrapper.plan(indptr, indices, last_page_len, heads, kv_heads, dim, page_size)
# 后续 plan/run 一直使用 workspace 的原始存储；capture/replay 由调用方负责。
```

**a / b：容量够不够，以及真实内存代价。** 测试从 seq_len=0 的首次 graph plan 开始取得最大容量，再对 1、17、128、1024、8192、32768、131072 逐个重新 plan；后续最大接口被 monkeypatch 为禁止调用，防止通过逐步重新分配或重新查询掩盖错误。每个样本都测 LSE False/True。

按需数字来自真实 NPU allocator 事件：预先分配输入、mask、两个输出，调用 FIA `.out(workspace=None)`（内部走与 eager default 相同的 CANN workspace 查询和分配），记录唯一的 `alloc` 事件。这里测的是实际分配字节数，含 allocator 对齐，不把输出/输入/segment reserve 当作 workspace；用它做容量比较比只取未对齐需求更严格。再给同一 FIA 调用传入 graph 的实际 uint8 buffer，验证新增 workspace alloc 事件为空。

下表在 FP16/BF16 上结果逐项相同，8/2 heads、BSND、Q S=1；LSE 开/关的按需字节数也相同。图列是测试真实分配并绑定的常驻 uint8 buffer，恰好等于首次 plan 的最大字节需求。

| seq_len | D=64 按需字节 | D=128 按需字节 | 图模式最大字节 | D=128 倍数 | D=128 多占字节 |
|---:|---:|---:|---:|---:|---:|
| 1 | 16,777,216 | 16,777,216 | 33,554,432 | 2.000000 | 16,777,216 |
| 17 | 16,777,216 | 16,777,216 | 33,554,432 | 2.000000 | 16,777,216 |
| 128 | 16,777,216 | 16,777,216 | 33,554,432 | 2.000000 | 16,777,216 |
| 1024 | 16,777,216 | 16,777,216 | 33,554,432 | 2.000000 | 16,777,216 |
| 8192 | 16,777,216 | 18,399,232 | 33,554,432 | 1.823687 | 15,155,200 |
| 32768 | 16,777,216 | 18,694,144 | 33,554,432 | 1.794917 | 14,860,288 |
| 131072 | 16,777,216 | 16,777,216 | 33,554,432 | 2.000000 | 16,777,216 |

所有样本均 `最大容量 >= 实际按需分配`，未发现容量不足；按需大小并不随 seq_len 单调增长。D=64 的对应倍数均为 2，多占 16,777,216 字节。表格只比较算子 workspace 的常驻/按需分配，不包含 Q/K/V、输出、mask、首次查询的暂存 tensor 或 allocator 缓存保留量。

本仓 resource_limits.py 明确没有全局固定 seq_len 上限，所以这是一组覆盖小/中/大规模的可复现实测，不能当作任意 SoC/任意参数/无限长度的数学证明。另做 118 组开发探针：heads/KV heads 为 8/2、32/8、64/8、128/8、256/8，Q length 覆盖 1、3、16、128、256、511、512、513、1024、4096，KV lengths 覆盖 128、1024、4096、16384、32768、131072 的分组组合；64/64 与 54/54 均 ENOUGH True。未为通过测试改参数或隐藏不够用的样本。

**c：真实地址和实际使用。** decode、paged prefill、ragged prefill 分别记录首次 plan 前持久 workspace 的 data_ptr、四次 plan 的 data_ptr 和两个真实 `.out` 调用实际接收的 workspace data_ptr，全都相同。测试还验证 seq_len 改变时最大查询仅首次执行 False/True 两次，LSE 两种 run 都不查询。

此外覆盖容量不足的失败原子性、已经成功后的更大需求拒绝发布、外部 set_ 换址在 plan/run 都被拒绝、读大小无副作用、允许的 prefill Q shape 变化重新查询、16 组 paged/ragged × NHD/HND × causal/window/custom/packed mask 的 graph/eager 数值一致性。torch_npu 的 .out 当前使用 storage().data()，所以新增最大路径必须拒绝非连续/非零 offset 的 workspace，防止实际写入地址与绑定的 data_ptr 不一致；这个限制没有施加给非图和 opt-out 路径。

命令与正对照完整输出（同时含 a/b/c 数字）：

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python tools/t10_max_workspace_audit.py
```

```text
[W1023 05:20:33.766149159 unwind.cpp:216] Warning: Unsupported unwinding pattern: addr not in range of known libraries (function unwinderFor)
float16 D=64 seq_len=1: eager_no_lse=16777216 eager_lse=16777216 graph=33554432 ratio=2.000000 delta=16777216 data_ptr=19792499048448 supplied_workspace_allocations=0
float16 D=64 seq_len=17: eager_no_lse=16777216 eager_lse=16777216 graph=33554432 ratio=2.000000 delta=16777216 data_ptr=19792499048448 supplied_workspace_allocations=0
float16 D=64 seq_len=128: eager_no_lse=16777216 eager_lse=16777216 graph=33554432 ratio=2.000000 delta=16777216 data_ptr=19792499048448 supplied_workspace_allocations=0
float16 D=64 seq_len=1024: eager_no_lse=16777216 eager_lse=16777216 graph=33554432 ratio=2.000000 delta=16777216 data_ptr=19792499048448 supplied_workspace_allocations=0
float16 D=64 seq_len=8192: eager_no_lse=16777216 eager_lse=16777216 graph=33554432 ratio=2.000000 delta=16777216 data_ptr=19792499048448 supplied_workspace_allocations=0
float16 D=64 seq_len=32768: eager_no_lse=16777216 eager_lse=16777216 graph=33554432 ratio=2.000000 delta=16777216 data_ptr=19792499048448 supplied_workspace_allocations=0
float16 D=64 seq_len=131072: eager_no_lse=16777216 eager_lse=16777216 graph=33554432 ratio=2.000000 delta=16777216 data_ptr=19792499048448 supplied_workspace_allocations=0
.float16 D=128 seq_len=1: eager_no_lse=16777216 eager_lse=16777216 graph=33554432 ratio=2.000000 delta=16777216 data_ptr=19792499048448 supplied_workspace_allocations=0
float16 D=128 seq_len=17: eager_no_lse=16777216 eager_lse=16777216 graph=33554432 ratio=2.000000 delta=16777216 data_ptr=19792499048448 supplied_workspace_allocations=0
float16 D=128 seq_len=128: eager_no_lse=16777216 eager_lse=16777216 graph=33554432 ratio=2.000000 delta=16777216 data_ptr=19792499048448 supplied_workspace_allocations=0
float16 D=128 seq_len=1024: eager_no_lse=16777216 eager_lse=16777216 graph=33554432 ratio=2.000000 delta=16777216 data_ptr=19792499048448 supplied_workspace_allocations=0
float16 D=128 seq_len=8192: eager_no_lse=18399232 eager_lse=18399232 graph=33554432 ratio=1.823687 delta=15155200 data_ptr=19792499048448 supplied_workspace_allocations=0
float16 D=128 seq_len=32768: eager_no_lse=18694144 eager_lse=18694144 graph=33554432 ratio=1.794917 delta=14860288 data_ptr=19792499048448 supplied_workspace_allocations=0
float16 D=128 seq_len=131072: eager_no_lse=16777216 eager_lse=16777216 graph=33554432 ratio=2.000000 delta=16777216 data_ptr=19792499048448 supplied_workspace_allocations=0
.bfloat16 D=64 seq_len=1: eager_no_lse=16777216 eager_lse=16777216 graph=33554432 ratio=2.000000 delta=16777216 data_ptr=19792499048448 supplied_workspace_allocations=0
bfloat16 D=64 seq_len=17: eager_no_lse=16777216 eager_lse=16777216 graph=33554432 ratio=2.000000 delta=16777216 data_ptr=19792499048448 supplied_workspace_allocations=0
bfloat16 D=64 seq_len=128: eager_no_lse=16777216 eager_lse=16777216 graph=33554432 ratio=2.000000 delta=16777216 data_ptr=19792499048448 supplied_workspace_allocations=0
bfloat16 D=64 seq_len=1024: eager_no_lse=16777216 eager_lse=16777216 graph=33554432 ratio=2.000000 delta=16777216 data_ptr=19792499048448 supplied_workspace_allocations=0
bfloat16 D=64 seq_len=8192: eager_no_lse=16777216 eager_lse=16777216 graph=33554432 ratio=2.000000 delta=16777216 data_ptr=19792499048448 supplied_workspace_allocations=0
bfloat16 D=64 seq_len=32768: eager_no_lse=16777216 eager_lse=16777216 graph=33554432 ratio=2.000000 delta=16777216 data_ptr=19792499048448 supplied_workspace_allocations=0
bfloat16 D=64 seq_len=131072: eager_no_lse=16777216 eager_lse=16777216 graph=33554432 ratio=2.000000 delta=16777216 data_ptr=19792499048448 supplied_workspace_allocations=0
.bfloat16 D=128 seq_len=1: eager_no_lse=16777216 eager_lse=16777216 graph=33554432 ratio=2.000000 delta=16777216 data_ptr=19792499048448 supplied_workspace_allocations=0
bfloat16 D=128 seq_len=17: eager_no_lse=16777216 eager_lse=16777216 graph=33554432 ratio=2.000000 delta=16777216 data_ptr=19792499048448 supplied_workspace_allocations=0
bfloat16 D=128 seq_len=128: eager_no_lse=16777216 eager_lse=16777216 graph=33554432 ratio=2.000000 delta=16777216 data_ptr=19792499048448 supplied_workspace_allocations=0
bfloat16 D=128 seq_len=1024: eager_no_lse=16777216 eager_lse=16777216 graph=33554432 ratio=2.000000 delta=16777216 data_ptr=19792499048448 supplied_workspace_allocations=0
bfloat16 D=128 seq_len=8192: eager_no_lse=18399232 eager_lse=18399232 graph=33554432 ratio=1.823687 delta=15155200 data_ptr=19792499048448 supplied_workspace_allocations=0
bfloat16 D=128 seq_len=32768: eager_no_lse=18694144 eager_lse=18694144 graph=33554432 ratio=1.794917 delta=14860288 data_ptr=19792499048448 supplied_workspace_allocations=0
bfloat16 D=128 seq_len=131072: eager_no_lse=16777216 eager_lse=16777216 graph=33554432 ratio=2.000000 delta=16777216 data_ptr=19792499048448 supplied_workspace_allocations=0
....capacity: float workspace capacity is 1024 bytes, requires 33554432
.decode workspace data_ptr: before=19792740220928 plans=[19792740220928, 19792740220928, 19792740220928, 19792740220928] run.out=[19792740220928, 19792740220928]; queries=[False, True]
.paged_prefill workspace data_ptr: before=19792740220928 plans=[19792740220928, 19792740220928, 19792740220928, 19792740220928] run.out=[19792740220928, 19792740220928]; queries=[False, True]
.ragged_prefill workspace data_ptr: before=19792740220928 plans=[19792740220928, 19792740220928, 19792740220928, 19792740220928] run.out=[19792740220928, 19792740220928]; queries=[False, True]
.....explicit workspace_size: (33554432, 0) active plan/fixed metadata unchanged
.prefill q_len 1->3: [33554432, 35913728] data_ptr: 19792740220928 no additional frozen dimension
..................
=============================== warnings summary ===============================
tests/test_attention_max_workspace.py::test_masked_prefill_uses_maximum_with_same_numerics[NHD-custom-paged]
  /home/dy/projects/flashinfer/tests/test_attention_max_workspace.py:272: UserWarning: Cannot create tensor with interal format while allow_internel_format=False, tensor will be created with base format. (Triggered internally at ../torch_npu/csrc/aten/common/TensorFactories.cpp:437.)
    mask = torch.arange(51, device=npu).reshape(3, 17) % 3 != 0

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
34 passed, 1 warning in 5.45s
```

**负对照。** 沿用 T9 的进程内替换门禁、要求特定测试准确失败、finally 恢复且不改源码的方式。分别移除三项保证：最大查询返回 0；移除 eager 的查询开关；移除固定地址绑定检查。工具检查每轮恰好一个指定测试的 call 失败，skip、收集错误、意外错误集合均不能算负对照通过。

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python tools/t10_max_workspace_audit.py --negative-control
```

关键实际输出：

```text
NEGATIVE CONTROL: maximum replaced with zero
E               assert 0 >= 16777216
FAILED tests/test_attention_max_workspace.py::test_maximum_covers_sequence_lengths[float16-64]
1 failed in 3.84s
PASS negative control: maximum replaced with zero -> 1/1 intended test failed
NEGATIVE CONTROL: eager query gate removed
E       AssertionError: maximum query called without graph opt-in
FAILED tests/test_attention_max_workspace.py::test_eager_and_opt_out_never_query_maximum[eager-opt-in]
1 failed in 0.04s
PASS negative control: eager query gate removed -> 1/1 intended test failed
NEGATIVE CONTROL: fixed workspace binding guard removed
E       Failed: DID NOT RAISE <class 'flashinfer.attention.planner.AttentionStateError'>
FAILED tests/test_attention_max_workspace.py::test_workspace_repoint_is_rejected[plan]
1 failed in 0.02s
PASS negative control: fixed workspace binding guard removed -> 1/1 intended test failed
PASS negative controls: 3/3 guarantees detected; source files unchanged
```

**修改文件与原因。**

| 文件 | 修改原因 |
|---|---|
| flashinfer/attention/public_batch.py | 在 T9 的两阶段 plan 与 workspace contract 上绑定最大容量及静态查询缓存；只读 native workspace_size；容量/地址校验先于发布；run 透传原 workspace。 |
| flashinfer/cann/attention.py | 延迟加载真实最大接口、使用真实 BSND prototype 和字节换算；仅提供 workspace 时切到 v1 .out，原 eager 调用保留。 |
| flashinfer/decode.py、flashinfer/prefill.py | 三种构造函数末尾增加默认 False 的仅关键字 opt-in，说明调用方分配、图模式 gating 和内存原因。 |
| flashinfer/data/api_parity.json、flashinfer/data/attention_api_parity.json | 同步登记 NPU 扩展参数与 native 查询状态，保留 T11 限制；未改变 kernel descriptors。 |
| tools/update_parity_inventory.py | 使本次 Attention workspace 证据可幂等重建，避免下次生成覆盖回旧 T10 待实现文字；单独的注解函数不依赖全量符号解析。 |
| tests/test_attention_max_workspace.py | 34 项可证伪 NPU 验证，容量/成本/指针以及失败原子性、mask 数值、只读查询等。 |
| tests/test_batch_frontend.py | 旧构造函数完整列表增加新参数，并断言仅关键字与 False 默认值；原所有 plan/run/forward 签名断言不变。 |
| tools/t10_max_workspace_audit.py | 可重跑的签名/返回值/参数探针、按需 allocator 测量、正对照与三组负对照。 |
| docs/reports/T10-report.md | 保存本报告的全部证据与 79 行逐项对照；未修改任务文档。 |

**全量、T9 和导入约束。**

修改前、修改后均执行：

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python -m pytest -q
```

修改前原样输出：

```text
FAILED tests/test_attention_metadata_wire.py::AttentionMetadataWireTests::test_config_directory_layout_and_canonical_blob_are_frozen
FAILED tests/test_attention_quant_launch_abi.py::AttentionQuantizedKVLaunchABITests::test_v1_is_frozen_and_v2_has_an_independent_binary_identity
FAILED tests/test_runtime_abi.py::KernelBinaryABITests::test_attention_binary_abi_matches_logical_order_mutation_and_stream
3 failed, 1613 passed, 4 warnings in 22.37s
```

修改后最终原样输出：

```text
FAILED tests/test_attention_metadata_wire.py::AttentionMetadataWireTests::test_config_directory_layout_and_canonical_blob_are_frozen
FAILED tests/test_attention_quant_launch_abi.py::AttentionQuantizedKVLaunchABITests::test_v1_is_frozen_and_v2_has_an_independent_binary_identity
FAILED tests/test_runtime_abi.py::KernelBinaryABITests::test_attention_binary_abi_matches_logical_order_mutation_and_stream
3 failed, 1647 passed, 4 warnings in 22.41s
```

三个失败的测试名及日志中的所有 expected/actual SHA256 值逐项相等：`PASS unchanged ABI freeze failures: 3/3; all expected/actual SHA256 values unchanged`。没有增加 ABI 冻结失败。中间完整回归曾为 `5 failed, 1645 passed, 4 warnings in 23.12s`，新增的两个失败是构造函数签名列表未包含新 opt-in；已显式登记参数并补默认值/仅关键字断言后消失。早期仅 T10 测试曾为 `1 failed, 16 passed in 5.99s`，当时错误假设只要改变 Q length 最大返回值就一定增大；实测同样可能相等，已把测试改为验证真实的两组配置和缓存失效，没有修改生产代码来迎合该假设。

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python tools/t09_graph_plan_audit.py
/root/miniforge3/envs/flashinfer-a5/bin/python tools/t09_graph_plan_audit.py --negative-control
```

```text
21 passed, 1 warning in 3.27s
4 failed in 2.38s
PASS negative control: 4/4 freeze tests failed with guards removed; source files unchanged
```

T9 原测试文件及审计工具本身未修改，execution_identity.py/planner.py 没有 diff；固定维度和错误信息保持原样。

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python -m pytest -q -s tests/test_torch_library.py::test_import_is_pure_and_registration_is_opt_in
```

```text
{"provider_import_attempts": [], "compiler_probes": [], "registration_enabled": false}
.
1 passed in 0.13s
```

这个子进程测试同时禁止 torch、torch_npu、triton、acl 的导入尝试，及 shutil.which/subprocess.Popen，结果全为空，未触发编译器探测。

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python -m flashinfer parity-check
/root/miniforge3/envs/flashinfer-a5/bin/python -m flashinfer parity-report --scope attention
```

结果：`Parity ledger integrity: passed`；总表 304 symbols、Attention 36 symbols；保留原始覆盖表缺失/消费者未验证状态。两本账本的 T10 注解幂等检查为 `PASS T10 parity annotation idempotent; both ledgers synchronized`。

```bash
git diff --check
git diff --exit-code -- docs/tasks/ docs/TASKS.md flashinfer/attention/execution_identity.py flashinfer/attention/planner.py
```

两条命令均 exit 0、无输出。

**79 条 auto 选型逐项对照。**

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python tools/t19_selection_audit.py --write /tmp/t10-selection-before.json
/root/miniforge3/envs/flashinfer-a5/bin/python tools/t19_selection_audit.py --compare /tmp/t10-selection-before.json
```

快照 SHA256：`0c39bb1c5c6f2461ab917708424a9233f8325ac23e971e4dfb75befba2b593ea`。与修改前相比，记录顺序、stack、descriptor fingerprint、signature、hardware、auto selected、hot_path_selected、fallback 顺序和每个 candidate filter 逐项相等。

```text
records: 79
stack: cann records: 40
stack: reference records: 38
stack: triton records: 1
multi_config_op_source_groups: 0 []
```

| record | before auto | after auto |
|---|---|---|
| cann/activation/gelu_and_mul_v1 | cann/activation/gelu_and_mul_v1 | cann/activation/gelu_and_mul_v1 |
| cann/activation/gelu_tanh_and_mul_v1 | cann/activation/gelu_tanh_and_mul_v1 | cann/activation/gelu_tanh_and_mul_v1 |
| cann/activation/silu_and_mul_v1 | cann/activation/silu_and_mul_v1 | cann/activation/silu_and_mul_v1 |
| cann/attention/batch_decode_paged_fia_v1 | cann/attention/batch_decode_paged_fia_v1 | cann/attention/batch_decode_paged_fia_v1 |
| cann/attention/batch_prefill_paged_fia_v1 | cann/attention/batch_prefill_paged_fia_v1 | cann/attention/batch_prefill_paged_fia_v1 |
| cann/attention/batch_prefill_ragged_fia_v1 | cann/attention/batch_prefill_ragged_fia_v1 | cann/attention/batch_prefill_ragged_fia_v1 |
| cann/attention/single_decode_fia_v1 | cann/attention/single_decode_fia_v1 | cann/attention/single_decode_fia_v1 |
| cann/attention/single_prefill_fia_v1 | cann/attention/single_prefill_fia_v1 | cann/attention/single_prefill_fia_v1 |
| cann/cascade/merge_state_attention_update_v1 | cann/cascade/merge_state_attention_update_v1 | cann/cascade/merge_state_attention_update_v1 |
| cann/cascade/merge_states_v1 | cann/cascade/merge_states_v1 | cann/cascade/merge_states_v1 |
| cann/decode/cudnn_batch_decode_fia_v1 | cann/decode/cudnn_batch_decode_fia_v1 | cann/decode/cudnn_batch_decode_fia_v1 |
| cann/fused_moe/cutlass_fused_moe_gmm_v1 | cann/fused_moe/cutlass_fused_moe_gmm_v1 | cann/fused_moe/cutlass_fused_moe_gmm_v1 |
| cann/fused_moe/fp8_block_scale_moe_cascade_v1 | cann/fused_moe/fp8_block_scale_moe_cascade_v1 | cann/fused_moe/fp8_block_scale_moe_cascade_v1 |
| cann/fused_moe/fp8_per_tensor_scale_moe_cascade_v1 | cann/fused_moe/fp8_per_tensor_scale_moe_cascade_v1 | cann/fused_moe/fp8_per_tensor_scale_moe_cascade_v1 |
| cann/gemm/bmm_bf16_v1 | cann/gemm/bmm_bf16_v1 | cann/gemm/bmm_bf16_v1 |
| cann/gemm/bmm_fp8_grouped_v1 | cann/gemm/bmm_fp8_grouped_v1 | cann/gemm/bmm_fp8_grouped_v1 |
| cann/gemm/gemm_fp8_nt_groupwise_block_v1 | cann/gemm/gemm_fp8_nt_groupwise_block_v1 | cann/gemm/gemm_fp8_nt_groupwise_block_v1 |
| cann/gemm/mm_bf16_v1 | cann/gemm/mm_bf16_v1 | cann/gemm/mm_bf16_v1 |
| cann/gemm/segment_gemm_v1 | cann/gemm/segment_gemm_v1 | cann/gemm/segment_gemm_v1 |
| cann/norm/fused_add_rmsnorm_aclnn_v1 | cann/norm/fused_add_rmsnorm_aclnn_v1 | cann/norm/fused_add_rmsnorm_aclnn_v1 |
| cann/norm/gemma_rmsnorm_aclnn_v1 | cann/norm/gemma_rmsnorm_aclnn_v1 | cann/norm/gemma_rmsnorm_aclnn_v1 |
| cann/norm/layernorm_aclnn_v1 | cann/norm/layernorm_aclnn_v1 | cann/norm/layernorm_aclnn_v1 |
| cann/norm/rmsnorm_aclnn_v1 | cann/norm/rmsnorm_aclnn_v1 | cann/norm/rmsnorm_aclnn_v1 |
| cann/page/append_paged_kv_cache_v1 | cann/page/append_paged_kv_cache_v1 | cann/page/append_paged_kv_cache_v1 |
| cann/rope/apply_rope_cos_sin_cache_aclnn_v1 | cann/rope/apply_rope_cos_sin_cache_aclnn_v1 | cann/rope/apply_rope_cos_sin_cache_aclnn_v1 |
| cann/rope/apply_rope_cos_sin_cache_out_v1 | cann/rope/apply_rope_cos_sin_cache_out_v1 | cann/rope/apply_rope_cos_sin_cache_out_v1 |
| cann/rope/apply_rope_inplace_v1 | cann/rope/apply_rope_inplace_v1 | cann/rope/apply_rope_inplace_v1 |
| cann/rope/apply_rope_pos_ids_inplace_v1 | cann/rope/apply_rope_pos_ids_inplace_v1 | cann/rope/apply_rope_pos_ids_inplace_v1 |
| cann/rope/apply_rope_pos_ids_v1 | cann/rope/apply_rope_pos_ids_v1 | cann/rope/apply_rope_pos_ids_v1 |
| cann/rope/apply_rope_v1 | cann/rope/apply_rope_v1 | cann/rope/apply_rope_v1 |
| cann/sampling/chain_speculative_sampling_v1 | cann/sampling/chain_speculative_sampling_v1 | cann/sampling/chain_speculative_sampling_v1 |
| cann/sampling/min_p_sampling_from_probs_v1 | cann/sampling/min_p_sampling_from_probs_v1 | cann/sampling/min_p_sampling_from_probs_v1 |
| cann/sampling/sampling_from_logits_v1 | cann/sampling/sampling_from_logits_v1 | cann/sampling/sampling_from_logits_v1 |
| cann/sampling/sampling_from_probs_v1 | cann/sampling/sampling_from_probs_v1 | cann/sampling/sampling_from_probs_v1 |
| cann/sampling/top_k_mask_logits_v1 | cann/sampling/top_k_mask_logits_v1 | cann/sampling/top_k_mask_logits_v1 |
| cann/sampling/top_k_renorm_probs_v1 | cann/sampling/top_k_renorm_probs_v1 | cann/sampling/top_k_renorm_probs_v1 |
| cann/sampling/top_k_sampling_from_probs_v1 | cann/sampling/top_k_sampling_from_probs_v1 | cann/sampling/top_k_sampling_from_probs_v1 |
| cann/sampling/top_k_top_p_sampling_from_probs_v1 | cann/sampling/top_k_top_p_sampling_from_probs_v1 | cann/sampling/top_k_top_p_sampling_from_probs_v1 |
| cann/sampling/top_p_renorm_probs_v1 | cann/sampling/top_p_renorm_probs_v1 | cann/sampling/top_p_renorm_probs_v1 |
| cann/sampling/top_p_sampling_from_probs_v1 | cann/sampling/top_p_sampling_from_probs_v1 | cann/sampling/top_p_sampling_from_probs_v1 |
| reference/activation/gelu_and_mul_v1 | reference/activation/gelu_and_mul_v1 | reference/activation/gelu_and_mul_v1 |
| reference/activation/gelu_tanh_and_mul_v1 | reference/activation/gelu_tanh_and_mul_v1 | reference/activation/gelu_tanh_and_mul_v1 |
| reference/activation/silu_and_mul_v1 | reference/activation/silu_and_mul_v1 | reference/activation/silu_and_mul_v1 |
| reference/attention/decode_v1 | reference/attention/decode_v1 | reference/attention/decode_v1 |
| reference/attention/prefill_ragged_v1 | reference/attention/prefill_ragged_v1 | reference/attention/prefill_ragged_v1 |
| reference/cascade/merge_state_v1 | cann/cascade/merge_state_attention_update_v1 | cann/cascade/merge_state_attention_update_v1 |
| reference/cascade/merge_states_v1 | reference/cascade/merge_states_v1 | reference/cascade/merge_states_v1 |
| reference/decode/cudnn_batch_decode_v1 | cann/decode/cudnn_batch_decode_fia_v1 | cann/decode/cudnn_batch_decode_fia_v1 |
| reference/fused_moe/cutlass_fused_moe_v1 | cann/fused_moe/cutlass_fused_moe_gmm_v1 | cann/fused_moe/cutlass_fused_moe_gmm_v1 |
| reference/fused_moe/trtllm_fp8_block_scale_moe_v1 | reference/fused_moe/trtllm_fp8_block_scale_moe_v1 | reference/fused_moe/trtllm_fp8_block_scale_moe_v1 |
| reference/fused_moe/trtllm_fp8_per_tensor_scale_moe_v1 | reference/fused_moe/trtllm_fp8_per_tensor_scale_moe_v1 | reference/fused_moe/trtllm_fp8_per_tensor_scale_moe_v1 |
| reference/gemm/bmm_bf16_v1 | cann/gemm/bmm_bf16_v1 | cann/gemm/bmm_bf16_v1 |
| reference/gemm/bmm_fp8_v1 | reference/gemm/bmm_fp8_v1 | reference/gemm/bmm_fp8_v1 |
| reference/gemm/gemm_fp8_nt_groupwise_v1 | reference/gemm/gemm_fp8_nt_groupwise_v1 | reference/gemm/gemm_fp8_nt_groupwise_v1 |
| reference/gemm/mm_bf16_v1 | cann/gemm/mm_bf16_v1 | cann/gemm/mm_bf16_v1 |
| reference/gemm/segment_gemm_v1 | cann/gemm/segment_gemm_v1 | cann/gemm/segment_gemm_v1 |
| reference/norm/fused_add_rmsnorm_v1 | cann/norm/fused_add_rmsnorm_aclnn_v1 | cann/norm/fused_add_rmsnorm_aclnn_v1 |
| reference/norm/gemma_fused_add_rmsnorm_v1 | triton/norm/gemma_fused_add_rmsnorm_v1 | triton/norm/gemma_fused_add_rmsnorm_v1 |
| reference/norm/gemma_rmsnorm_v1 | cann/norm/gemma_rmsnorm_aclnn_v1 | cann/norm/gemma_rmsnorm_aclnn_v1 |
| reference/norm/layernorm_v1 | cann/norm/layernorm_aclnn_v1 | cann/norm/layernorm_aclnn_v1 |
| reference/norm/rmsnorm_v1 | cann/norm/rmsnorm_aclnn_v1 | cann/norm/rmsnorm_aclnn_v1 |
| reference/page/append_paged_kv_cache_v1 | reference/page/append_paged_kv_cache_v1 | reference/page/append_paged_kv_cache_v1 |
| reference/rope/apply_rope_inplace_v1 | cann/rope/apply_rope_inplace_v1 | cann/rope/apply_rope_inplace_v1 |
| reference/rope/apply_rope_pos_ids_inplace_v1 | cann/rope/apply_rope_pos_ids_inplace_v1 | cann/rope/apply_rope_pos_ids_inplace_v1 |
| reference/rope/apply_rope_pos_ids_v1 | cann/rope/apply_rope_pos_ids_v1 | cann/rope/apply_rope_pos_ids_v1 |
| reference/rope/apply_rope_v1 | cann/rope/apply_rope_v1 | cann/rope/apply_rope_v1 |
| reference/rope/apply_rope_with_cos_sin_cache_out_v1 | cann/rope/apply_rope_cos_sin_cache_out_v1 | cann/rope/apply_rope_cos_sin_cache_out_v1 |
| reference/rope/apply_rope_with_cos_sin_cache_v1 | cann/rope/apply_rope_cos_sin_cache_aclnn_v1 | cann/rope/apply_rope_cos_sin_cache_aclnn_v1 |
| reference/sampling/chain_speculative_sampling_v1 | reference/sampling/chain_speculative_sampling_v1 | reference/sampling/chain_speculative_sampling_v1 |
| reference/sampling/min_p_sampling_from_probs_v1 | reference/sampling/min_p_sampling_from_probs_v1 | reference/sampling/min_p_sampling_from_probs_v1 |
| reference/sampling/sampling_from_logits_v1 | reference/sampling/sampling_from_logits_v1 | reference/sampling/sampling_from_logits_v1 |
| reference/sampling/sampling_from_probs_v1 | reference/sampling/sampling_from_probs_v1 | reference/sampling/sampling_from_probs_v1 |
| reference/sampling/top_k_mask_logits_v1 | reference/sampling/top_k_mask_logits_v1 | reference/sampling/top_k_mask_logits_v1 |
| reference/sampling/top_k_renorm_probs_v1 | reference/sampling/top_k_renorm_probs_v1 | reference/sampling/top_k_renorm_probs_v1 |
| reference/sampling/top_k_sampling_from_probs_v1 | reference/sampling/top_k_sampling_from_probs_v1 | reference/sampling/top_k_sampling_from_probs_v1 |
| reference/sampling/top_k_top_p_sampling_from_probs_v1 | reference/sampling/top_k_top_p_sampling_from_probs_v1 | reference/sampling/top_k_top_p_sampling_from_probs_v1 |
| reference/sampling/top_p_renorm_probs_v1 | reference/sampling/top_p_renorm_probs_v1 | reference/sampling/top_p_renorm_probs_v1 |
| reference/sampling/top_p_sampling_from_probs_v1 | reference/sampling/top_p_sampling_from_probs_v1 | reference/sampling/top_p_sampling_from_probs_v1 |
| triton/norm/gemma_fused_add_rmsnorm_v1 | triton/norm/gemma_fused_add_rmsnorm_v1 | triton/norm/gemma_fused_add_rmsnorm_v1 |

```text
PASS 79/79 auto selections unchanged; 79/79 descriptor fingerprints unchanged
PASS hot-path selections, fallback order and per-candidate filters unchanged
```

**遇到而未修改的相邻问题。**

1. 三个历史 ABI 冻结指纹失败保持原样。
2. 本机 API docstring 的 v1 参数与 dispatcher 不完全同步，支持芯片描述也较旧；以实际 schema、Ascend950 返回值为准，没有修改第三方包。
3. 最大接口的实际长度参数能导致很大的返回容量（本例 182,218,752 字节）；当前 dense BSND 路径不传该参数，没有为了“让最大值好看”改变本仓算子参数。未为 v2/量化/rope/分页 FIA 构造不存在的执行路径。
4. T11 的动态参数/tiling 更新未实现；当前 run 的 KV gather、输出/中间 tensor 分配仍是既有 composite 行为。本次只保证显式启用后的固定 workspace 容量和传址，不宣称完整算子/模型已可 replay。本仓没有调用 capture/replay。
5. 全量账本生成器存在独立的既有错误。当前和 HEAD 原脚本都执行到 `ValueError: unresolved ledger symbol: flashinfer.config.dispatch_order`；基线复现命令如下。本次只用幂等 Attention 注解同步两本账本，没有改 config 符号解析。

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python tools/update_parity_inventory.py
git show HEAD:tools/update_parity_inventory.py > /tmp/t10-baseline-update-parity.py
/root/miniforge3/envs/flashinfer-a5/bin/python /tmp/t10-baseline-update-parity.py
```

```text
ValueError: unresolved ledger symbol: flashinfer.config.dispatch_order
```
