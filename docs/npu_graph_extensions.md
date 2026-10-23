`flashinfer.npu` 是 T10/T11 的 NPU-only 扩展命名空间；固定上游
`2895577720741ddfedff118202c89584996b8875` 没有这些接口。两本对标账本都将
这些符号登记为 `api_kind=npu_extension`、`semantic_status=intentionally_different`、
`upstream_counterpart=null`、`drop_in=false`。没有在 `flashinfer` 顶层再导出。

调用方持有图、固定 buffer 和流，负责整个 forward 的 capture/replay。本仓只在
已有 capture 中记录算子任务、在每次 replay 前更新任务参数；不创建或捕获图。

```python
from flashinfer.npu import (
    GraphCapability, ReplayParameterUpdates, graph_requirements,
    maximum_workspace_size,
)

capability, native_parameter_names = graph_requirements(selected_kernel_id)
if capability == GraphCapability.NOT_CAPTURABLE:
    # 由调用方选择 eager 或图分段策略。
    ...
elif capability == GraphCapability.CAPTURE_SAFE:
    # 不需要参数更新作用域。
    ...
else:
    updates = ReplayParameterUpdates()  # 每个图独享，生命周期和图一致
    # 下面的 torch.npu.graph()/replay() 均属于调用方代码。
    with torch.npu.graph(graph, stream=capture_stream, auto_dispatch_capture=False):
        with updates.recording():
            model_forward()  # 包含调用本仓已验证 runner 的算子

    # 在调用方当前 replay 流上调用；stream 是单独的参数更新流。
    updates.before_replay(
        [{"kv_seq_lengths": [current_length]} for task in updates.tasks],
        stream=update_stream,
    )
    graph.replay()
```

上例按每个录制任务传一份参数，顺序为 forward 调用顺序。多个层使用不同长度时，
调用方应在 capture 调用点与 `updates.tasks` 之间保留层/请求的对应关系，逐项传入，
不能假定一个 kernel ID 只调用一次。`task.kernel_id` 与 `task.descriptor` 提供静态身份，
参数更新机制不依赖 vLLM 的 `forward_context`。可把
`before_replay` 这个绑定方法放进 vllm-ascend 的 replay 前回调；已有
`ParamProvider.resolve()` 的逻辑数据也可以交给这个回调。参数更新任务只由本仓的
recording 作用域记录一次，调用方不需要为同一个任务再注册一套 UpdatableGraph 更新。

参考核对的下游源码是
[UpdatableGraph](https://github.com/vllm-project/vllm-ascend/blob/c4f64438034a85ac68a93a788bbe1b2df84d6eb5/vllm_ascend/compilation/updatable_graph.py)、
[use_updatable_graph](https://github.com/vllm-project/vllm-ascend/blob/c4f64438034a85ac68a93a788bbe1b2df84d6eb5/vllm_ascend/utils.py#L1913)
以及 attention_v1 的 FIA/PA ParamProvider。这里提供可接入接口和本仓真实 runner 验证，
没有修改下游仓或运行完整 vLLM 模型；账本的 consumer 使用状态保留 `unverified`。

目前只有 `cann/attention/single_decode_fia_v1` 声明 `NEEDS_PARAM_UPDATE`。
`graph_requirements()` 返回它的原生更新字段名，供诊断；调用方实际传入逻辑字段
`kv_seq_lengths`，无需知道 FIA 的 CANN 参数名或位置。descriptor 的 `graph_replay` 为：

```python
graph_capability = GraphCapability.NEEDS_PARAM_UPDATE
graph_param_updates = ("actual_seq_lengths_kv",)
graph_replay = GraphReplaySpec(
    validated_kernel_id="cann/attention/single_decode_fia_v1",
    strategy="host_kwargs",
    parameter_sources=(("actual_seq_lengths_kv", "kv_seq_lengths"),),
    # 其余字段见 flashinfer/cann/attention.py：
    # required_arguments、fixed_arguments、tensor_metadata、parameter_ranges，
    # 以及 validation_tests 和 validated_environment。
)
```

这个声明有严格范围：Ascend950 上的 BSND `.out`、固定 workspace/out、无 mask，
query `(1,1,8,64)`、key/value `(1,1024,2,64)`、FP16/BF16、8/2 heads、
scale `0.125`、返回 LSE；逻辑 KV 长度为固定长度为 1 的 host list，元素在 `[1,1024]`。
真实测试覆盖 `63,1,17,128,513,1024,3,63`，含重复长度以及关闭更新的前后对照。
调用方可用 registry descriptor 的 runner 传入 `decode=True, return_lse=True,
workspace=workspace`。顶层上游形状的 single-decode 接口没有 workspace 参数，本次
没有扩展那个签名，也没有宣称 batch/paged wrapper 已能动态重放。

`host_kwargs` 由 11A 统一实现：录制时保存完整命名参数、张量绑定、句柄和事件；
replay 前把逻辑字段映射到原生参数，检查所有任务的字段、host 类型/序列长度、范围、
张量地址/shape/stride/dtype/device，然后才执行设备更新。时序保持：

```text
capture: event.wait -> event.reset -> group_begin -> operation -> group_end
update:  update_stream.wait_stream(current_replay_stream)
         update_begin -> operation(new kwargs) -> update_end -> event.record
caller:  graph.replay()
```

调用方必须串行访问同一个 updater，并在调用 `before_replay()` 的当前流上排队 replay。
预检失败不更新任何任务；设备更新失败后丢弃 updater 和对应图。改变 tensor 地址、
dtype、布局、host list 长度或超出已验证参数范围需要重新捕获。

接入一个新算子时，保留默认 `NOT_CAPTURABLE`，先写由测试/调用方拥有图的真实
capture/replay 对照。验证后声明 `graph_capability`、`graph_param_updates` 和
`GraphReplaySpec`：逻辑字段到原生命名参数的映射、统一策略、录制条件/更新范围、
绑定的 kernel ID、可运行测试 selector 和验证环境。adapter 只需把 descriptor、
普通 operation 和 kwargs 交给 `_record_kernel_call()`，不编写句柄、事件或更新循环。
`CAPTURE_SAFE` 的声明也必须有证据，使用 `strategy="none"`、空更新字段与映射。

`python tools/t11_replay_param_audit.py --declarations` 检查所有非默认声明必须有
kernel 绑定证据和真实存在的测试；全量 CPU 测试也包含这个门禁。默认审计运行现有
图安全测试以及所有新声明引用的其它测试。`--negative-control` 一方面伪造无证据声明，
另一方面给真实 descriptor 注入“eager 正常、capture 失败”的 runner，要求证据测试失败。
这不构成对任意算子、任意输入的自动证明；无 NPU 时设备测试必须跳过，不能用 skip
作为提升能力声明的证据。新增或放宽声明仍需要对应环境中真实通过的结果及评审。

T10 的显式最大 workspace 查询可以写成：

```python
sizes = maximum_workspace_size(wrapper, *plan_args, **plan_kwargs)
```

它复用既有 `wrapper.workspace_size()` 的只读查询，返回两个字节数，不发布 plan。
`use_max_workspace=False` 的默认值及启用条件不变；原有构造参数作为兼容入口保留，
在账本参数级另标为 NPU-only。已有上游 `workspace_size()` 符号仍按真实来源对标，
不把它错误登记成上游不存在的名字。

79 条 auto 选型保持不变。只有上面的 single-decode descriptor 因增加非默认图声明、
更新配方与证据改变指纹。原始选型基准保持不动；`t19_selection_audit.py` 默认仍严格
拒绝指纹变化，显式 `--allow-graph-metadata-change` 才逐项验证去掉图字段后仍与原指纹
一致，并继续严格检查 selected、fallback、hot-path 与所有候选过滤报告。
