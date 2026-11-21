# 图模式接口与范围

面向推理框架集成者：本页说明现有 NPU 图辅助接口、已声明范围和接入条件。调用方创建并拥有图、固定 buffer 与流；通用辅助接口不承担模型切图或整图重放，受限 single-decode wrapper 为调用方的图提供带更新检查的重放入口。

## 当前范围

源码全局只有 `cann/attention/single_decode_fia_v1` 声明 `needs_param_update`；其余默认实现均声明 `not_capturable`。运行 `python -m flashinfer list-kernels --all` 可查看当前每个实现的图能力。类名带 CUDAGraph、FakeTensor 或“固定尺寸”不构成图证据。

该 single-decode 声明限定 Ascend950、BSND `.out`、固定 workspace/out、无 mask；query 为 `[1,1,8,64]`，K/V 为 `[1,1024,2,64]`、FP16/BF16、8/2 heads、scale=0.125、返回 LSE。更新的逻辑 KV 长度是长度为 1 的 host list，元素范围 1..1024。

这不是通用 single/batch/paged 动态图支持；公共入口是否能提供该声明所需资源也要核对。真实证据入口是 [图验证测试](../tests/framework/test_single_decode_graph_real.py)，完整模型与其他输入不能由此推断。

公共 `flashinfer.decode.SingleDecodeWithKVCacheWrapper` 在捕获外执行 `plan()`、`warmup()`，在调用方 capture 内执行 `recording(graph)` 与恰好一次 `run()`。一张图只绑定一个此类 wrapper；每次重放均调用 `before_replay(kv_seq_len, stream=...)` 后再调用 wrapper 的 `replay()`，不能绕过更新检查直接重放底层图。捕获后不能再次 `plan()` 或替换固定资源。通用 prepared、多 wrapper 协调和动态 batch 图仍是 [设计目标](features/wrapper-and-aclgraph-design.md)。

## NPU 扩展接口

| 接口 | 作用 |
| --- | --- |
| `flashinfer.npu.graph_requirements(kernel_id)` | 查询已选实现的图声明和原生更新参数名，不探测设备 |
| `ReplayParameterUpdates` | 为一张调用方图记录任务，并在 replay 前验证与更新参数 |
| `maximum_workspace_size(wrapper, ...)` | 复用 wrapper 的只读 workspace_size 查询，不发布 plan、不启用图 |

`ReplayParameterUpdates.recording()` 必须位于调用方已经开始的 capture 中。每次 replay 前调用 `before_replay(parameter_updates, stream=...)`，按录制顺序为每个任务提供逻辑参数；同一 kernel 可以录制多次，不能按 ID 去重。

逻辑字段为 `kv_seq_lengths`，不要求调用方知道 CANN 参数位置。预检包括任务字段、host 类型和范围、Tensor 地址/shape/stride/dtype/device 与流。改变绑定或超出范围需要重新捕获，设备更新失败后丢弃对应 updater 与图。

调用方串行访问同一 updater，并在更新所约束的 replay 流排队重放。本仓负责更新所需事件与句柄生命周期，调用方仍负责整个 forward 的 capture/replay 顺序。

## 扩展一个实现

1. 保持默认 NOT_CAPTURABLE，定义最小静态资源和可变参数范围。
2. 通过实际公开/所选 runner 路径验证 eager、capture 和多次 replay 的数值一致。
3. 补地址、shape、范围、漏更新、重绑、stream 与失败恢复的负例。
4. 声明 GraphReplaySpec：kernel 绑定、策略、逻辑到原生字段映射、固定条件、参数范围、测试与环境证据。
5. 在对应目标环境完成真实验收后才提升 graph_capability，不能用模拟调用或能编译替代。

图扩展仍需解决原生 tiling/executor、workspace、同步和资源寿命问题，目标与验收条件见 [规划](roadmap.md)。源码：[NPU 接口](../flashinfer/npu.py)、[参数更新](../flashinfer/runtime/_graph_params.py)、[CANN Attention](../flashinfer/cann/attention/_shared.py)。
