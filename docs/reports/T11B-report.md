T11B 交付记录 · 2026-10-23

独立核实的修改前状态是 `records: 79`、`graph_capability: {'not_capturable': 79}`、
`graph_param_updates_nonempty: 0`。这与 11A 的真实 FIA 捕获/重放证据矛盾。
本次只提升 `cann/attention/single_decode_fia_v1`；修改后为 1 条
`NEEDS_PARAM_UPDATE`、78 条 `NOT_CAPTURABLE`，非空更新字段 1 条。

环境为 `/root/miniforge3/envs/flashinfer-a5/bin/python`，torch `2.10.0+cpu`、
torch_npu `2.10.0.post7`、`Ascend950PR_9579`，本机 toolkit 安装信息为
`9.2.0-beta.2`。修改留在工作区，未 commit/push；未修改 `docs/tasks/`、
`docs/TASKS.md`、T9 的冻结实现或 T10 的默认值与启用语义。

实现由 descriptor 的 `graph_capability`、`graph_param_updates` 和新增
`graph_replay: GraphReplaySpec` 驱动。后者包含 `host_kwargs` 策略、逻辑字段到原生
参数的映射、固定参数/张量 metadata/数值范围、绑定的 kernel ID、测试 selector 和
验证环境。本例映射为 `actual_seq_lengths_kv <- kv_seq_lengths`。adapter 交出
descriptor、operation 和正常 kwargs 即可；句柄、事件、全部任务预检及更新循环仍
统一由 11A runtime 实现。序列长度与 host 类型、tensor 地址/shape/stride/dtype/device
仍被锁定，参数范围在录制和每次更新前都检查。

提升声明只覆盖已验证的 Ascend950 BSND `.out` 路径：固定 workspace/out、无 mask，
8/2 heads、dim=64、query 长度 1、KV 容量 1024、FP16/BF16、scale=0.125、返回 LSE。
其它尺寸、LSE 关闭等尚未取得证据的组合在 recording 中拒绝，eager 能力不变。
single-prefill、三个 batch FIA 条目及其它 74 条未取得本任务的真实 capture/replay
证据，保持保守声明；batch composite 也拒绝进入参数 recording。

新算子先以 `NOT_CAPTURABLE` 保留 eager，完成调用方拥有图的真实对照测试后，再
声明上述字段与有限验证范围。`CAPTURE_SAFE` 也必须附证据，策略为 `none`，更新字段
与映射为空。JSON/manifest 可往返配方与证据；非默认字段进入 descriptor 指纹。
使用示例、所有调用边界和下游参考源码见 [NPU 扩展说明](../npu_graph_extensions.md)。

对外入口均在 `flashinfer.npu`：

```python
capability, parameter_names = graph_requirements(selected_kernel_id)
updates = ReplayParameterUpdates()
# 调用方 capture 内：with updates.recording(): model_forward()
# 调用方每次 replay 前：
updates.before_replay([{"kv_seq_lengths": [seq_len]}], stream=update_stream)
# 随后由调用方在同一当前流排队 graph.replay()
sizes = maximum_workspace_size(wrapper, *plan_args, **plan_kwargs)
```

vllm-ascend 可用 kernel ID 查询替换 `issubclass` 判定，把绑定的 `before_replay`
方法接到其回调，在 capture 调用点保留层/请求与任务序号的关联。多个任务逐项提供
逻辑字段，不需按 CANN 参数位置解包。当前证明的是 registry runner 携带固定 workspace
的路径；上游形状的 single-decode 公共函数没有 workspace 参数，batch/paged wrapper
也没有因此取得动态重放保证。没有修改下游仓或跑完整 vLLM 模型，consumer 保留
`unverified`。本仓没有图的创建、capture 或 replay 逻辑。

实际运行的主要命令与原始输出如下。

```text
/root/miniforge3/envs/flashinfer-a5/bin/python -m pytest -q
修改前：3 failed, 1672 passed, 4 warnings in 24.12s
最终：3 failed, 1679 passed, 4 warnings in 23.72s

/root/miniforge3/envs/flashinfer-a5/bin/python -m pytest -q tests/test_replay_param_runtime.py tests/test_graph_capability.py
74 passed in 0.42s

/root/miniforge3/envs/flashinfer-a5/bin/python tools/t11_replay_param_audit.py
8 passed in 4.05s
```

真实测试已改走 descriptor runner → `_record_kernel_call` → public `before_replay`，
开启更新时使用 `atol=rtol=0` 检查输出与 LSE。每组捕获一次、同一图重放 10 次，
`auto_dispatch_capture=False`，同一个句柄与 33554432 字节 workspace，
`forward_calls_during_replay=0`。关闭更新仍记录 ExternalEvent，保证只关掉参数改写。

| FP16 递增数据 | seq_len | eager 首元素 | graph 首元素 | 输出最大误差 | LSE 最大误差 |
|---|---:|---:|---:|---:|---:|
| OFF | 63 | 31 | 511.5 | 480.5 | 4.02272034 |
| ON | 63 | 31 | 31 | 0 | 0 |
| ON | 1 | 0 | 0 | 0 | 0 |
| ON | 17 | 8 | 8 | 0 | 0 |
| ON | 128 | 63.5 | 63.5 | 0 | 0 |
| ON | 513 | 256 | 256 | 0 | 0 |
| ON | 1024 | 511.5 | 511.5 | 0 | 0 |
| ON | 3 | 1 | 1 | 0 | 0 |
| ON | 63 | 31 | 31 | 0 | 0 |
| OFF | 7 | 3 | 31 | 28 | 3.16992474 |

其余三组在所有 ON 长度下的输出和 LSE 最大误差也全部为 0。OFF/seq_len=63 的
BF16 递增数据误差为 `481`，FP16 随机数据为 `0.569366455`，BF16 随机数据为
`0.571044922`。完整日志 `/tmp/t11b-replay-audit.log`。

```text
/root/miniforge3/envs/flashinfer-a5/bin/python tools/t11_replay_param_audit.py --declarations
records: 79 graph_capability: {'not_capturable': 78, 'needs_param_update': 1}
evidence: cann/attention/single_decode_fia_v1 strategy=host_kwargs updates=('actual_seq_lengths_kv',)
tests: tests/test_replay_param_npu.py::test_fia_full_capture_replay_needs_parameter_update
parameter_sources: (('actual_seq_lengths_kv', 'kv_seq_lengths'),)
validated_environment: Ascend950PR_9579; torch 2.10.0+cpu; torch_npu 2.10.0.post7; CANN 9.2
PASS declaration evidence gate: 1 capturable, 78 conservative

/root/miniforge3/envs/flashinfer-a5/bin/python tools/t11_replay_param_audit.py --negative-control
PASS unverified capture_safe declaration rejected: reference/attention/decode_v1: capturable declaration has no graph replay validation evidence
RuntimeError: injected declared-capturable runner fails during capture
1 failed in 4.05s
PASS declared capturable but capture fails -> 1/1 cited hardware test failed; declaration restored
```

上面的 `1 failed` 是工具要求出现的故障注入结果，工具本身退出 0。证据门禁要求
kernel 绑定，不能将其它 kernel 的 spec 直接搬过来；检查测试 selector 真实存在。
设备故障注入让 eager 正常、仅在实际 capture 失败，证据测试会发现。任意 Python
runner 的任意输入无法由静态声明自动证明，因此采取“声明必须附证据、执行条件收窄、
在有 NPU 的 CI 重跑引用测试”的办法；无设备的 skip 不得作为提升声明的新证据。
完整日志 `/tmp/t11b-declarations.log`、`/tmp/t11b-declaration-negative.log`。

选型对照使用修改前保存的原始快照，没有重写基准：

```text
/root/miniforge3/envs/flashinfer-a5/bin/python tools/t19_selection_audit.py --write /tmp/t11b-t19-before.json
records: 79

/root/miniforge3/envs/flashinfer-a5/bin/python tools/t19_selection_audit.py --compare /tmp/t11b-t19-before.json --allow-graph-metadata-change
PASS 79/79 auto selections unchanged; 78/79 descriptor fingerprints unchanged
PASS 1 graph-only fingerprint changes verified against the original baseline
PASS hot-path selections, fallback order and per-candidate filters unchanged
```

唯一变化的 `cann/attention/single_decode_fia_v1` 指纹：

```text
before a3e58ff49ad4b2b46e12c9729dd0563720175a42c1a0e4209ef4322a59e0366d
after  a7061dffac50bb1651e509dec20f608d385c3a124fbe91be5524211366ff2b2a
```

增加的非默认图能力、参数名、配方及证据进入静态身份；恢复这三个字段为旧默认值
后，指纹必须逐字等于原始快照。op、eager capability、排序、fallback、候选过滤结果
及 eager 计算逻辑未变；runner 增加的是录制检查与 descriptor 绑定。严格的原命令
不加 flag 时仍按预期因这一个指纹变化退出 1，
没有削弱默认门禁。79 行对照与指纹检查见 `/tmp/t11b-selection-compare.log`。

账本查询使用 `validate_packaged_manifests()`，遍历 `local.startswith('flashinfer.npu.')`
及 `extra_fields['parameters']['use_max_workspace']`，两本账本都给出以下标记：

```text
flashinfer.npu.graph_requirements api_kind=npu_extension semantic_status=intentionally_different upstream_counterpart=None drop_in=False
flashinfer.npu.ReplayParameterUpdates.before_replay api_kind=npu_extension semantic_status=intentionally_different upstream_counterpart=None drop_in=False
flashinfer.npu.maximum_workspace_size api_kind=npu_extension semantic_status=intentionally_different upstream_counterpart=None drop_in=False
flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(use_max_workspace=False) api_kind=npu_extension semantic_status=intentionally_different drop_in=False
```

另外五个符号为 GraphCapability、GraphReplaySpec、ReplayParameterUpdates 类本身及
它的 recording/tasks；两个 prefill wrapper 的 T10 参数也具有同样的 NPU-only 标记。
查询完整输出 `/tmp/t11b-npu-ledger-query.log`。注解函数单独刷新这些条目可保持登记，
host 测试覆盖两本账本及参数级标记。

```text
/root/miniforge3/envs/flashinfer-a5/bin/python -m flashinfer parity-check
Parity ledger integrity: passed

/root/miniforge3/envs/flashinfer-a5/bin/python -m pytest -q -s tests/test_replay_param_runtime.py::test_private_runtime_and_eager_hook_need_no_optional_imports_or_probes
PASS import flashinfer + NPU extension + graph query + eager hook: optional imports=0 compiler probes=0
1 passed in 0.13s

/root/miniforge3/envs/flashinfer-a5/bin/python tools/t11_replay_param_audit.py --portable-skips
environment: no_torch
8 skipped in 0.01s
environment: no_torch_npu
8 skipped in 0.63s
environment: no_device
8 skipped in 0.01s
```

导入门禁在独立的 `python -S` 子进程阻断 torch/torch_npu/triton，同时禁止
`shutil.which` 和 `subprocess.Popen`。初始化 updater 和登记表查询不访问设备。
完整日志 `/tmp/t11b-import-purity.log`、`/tmp/t11b-portable-skips.log`。

最终失败的三个节点与实际/期望 SHA-256 都与基线相同：

```text
test_attention_metadata_wire.py::AttentionMetadataWireTests::test_config_directory_layout_and_canonical_blob_are_frozen
test_attention_quant_launch_abi.py::AttentionQuantizedKVLaunchABITests::test_v1_is_frozen_and_v2_has_an_independent_binary_identity
test_runtime_abi.py::KernelBinaryABITests::test_attention_binary_abi_matches_logical_order_mutation_and_stream
PASS 3/3 baseline ABI failures and actual/expected fingerprints unchanged
```

完整对照 `/tmp/t11b-abi-compare.log`，基线与最终输出分别为
`/tmp/t11b-baseline-pytest.log`、`/tmp/t11b-final-pytest.log`。
`git diff --check` 通过；禁止修改的任务文档没有 diff。

遇到但没有修改的相邻问题包括上述三个 ABI 冻结指纹失败、依赖捕获时长度的
causal/window/custom mask 预处理、batch/paged composite 的 gather/slice，以及
顶层 single-decode 没有固定 workspace 参数。能力范围未因此被放宽。

账本全量生成器的既有问题也重新复现，当前脚本及 HEAD 原脚本都在写文件前报错：

```text
/root/miniforge3/envs/flashinfer-a5/bin/python tools/update_parity_inventory.py
git show HEAD:tools/update_parity_inventory.py > /tmp/t11b-baseline-parity-inventory.py
/root/miniforge3/envs/flashinfer-a5/bin/python /tmp/t11b-baseline-parity-inventory.py
ValueError: unresolved ledger symbol: flashinfer.config.dispatch_order
```

没有顺手修复这个独立符号解析问题。本次通过幂等的 Attention workspace 注解和
NPU 扩展条目注解更新账本，完整性检查、单条刷新和双账本一致性测试均通过。
