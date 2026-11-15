# Tuning：测量、查询与记录

面向集成者和后端开发者：本页说明现有离线测量、记录失效与选择机制，帮助正确解释调优结果。

## 当前行为

调优是显式离线操作。import、dispatch 和普通算子调用不采样、不搜索新算法。路由先过滤能力与策略，再查询有效记录；没有有效显著 winner 时使用确定性排序，不宣称总能选到最快实现。

`FLASHINFER_NPU_ONLINE_TUNING` 只有 config 读取/显示，未接入在线采样。发布目录的 manifest 当前不提供测量 shard；本地或应用显式准备的有效证据可以参与查询。

通用 `autotune` 上下文、tactic 搜索及 prepared/图模式测量属于 [AutoTuner 设计](features/autotuner-design.md)，不是当前接口。

## 候选与采集

BenchmarkCandidate 指定 Descriptor 与 prepare 函数。`prepare(iterations)` 在计时外执行，计时内运行其返回的 callable 并同步；编译、分配和正确性检查需要由采集方提前完成，callable 内仍执行的工作会计入结果。每个样本应准备独立输入，原地算子不能反复消费已经改变的同一输入后当作相同比较。

`collect_tuning_record` 使用调用方 synchronize，计时包含 Python 启动与同步的调用方可见延迟，不是纯设备时间。记录保留原始样本、执行顺序、机器观察、候选和软件/产物身份。

默认 BenchmarkConfig 为 warmup=50、rounds=31、iterations=20、轮间隔 0.01 秒、最小相对差 3%、noise_multiplier=3、family_alpha=0.01。比较结合稳健离散度、成对显著性和不同执行顺序；无显著差异时保存 winner=null，而不是强行选冠军。

## 已有 FA 调优入口

CATLASS AOT 与实验 Triton FA 在独立 registry 中比较，默认生产目录不改变。已有脚本先确认数值和 AOT，再采集精确 shape/dtype 数据：

```bash
python -B tools/tune_flash_attention.py --output-dir build/fa-tuning --dtype float16 bfloat16 --shape 128,128,1 --shape 512,512,8
export FLASHINFER_NPU_TUNING_DB=build/fa-tuning/tuning.sqlite3
```

输出目录必须新建。该脚本默认 warmup=10、iterations=200，覆盖上述 BenchmarkConfig 的对应默认值。默认 timing-boundary=api；可选择 runner 边界，但不同边界的数值不能混比。使用这些记录时仍要进入相同实验 registry 作用域，并保持候选、能力和执行身份相同。要求出现两个后端并不能制造 winner，也不是性能优势证据。

## 存储与有效性

| 情况 | 行为 |
| --- | --- |
| 有效本地记录与有效随包记录同时匹配 | 本地优先；日期只在同一层内比较 |
| 最新有效本地 winner=null | 保留无显著差异结论，不用旧 winner 或随包 winner 覆盖 |
| 新记录损坏或身份不匹配 | 拒绝该记录，继续检查较早有效记录或另一层 |
| 胜者被当前 backend/JIT/custom/reference 策略过滤 | 不复活候选，回到合法候选的确定性排序 |
| 硬件或参与比较的任一候选变化 | 比较完整硬件、Descriptor、Capability 与适用的执行身份，失配不复用 |

数据库路径为 `FLASHINFER_NPU_TUNING_DB`，默认取 `$XDG_CACHE_HOME/flashinfer/tuning.sqlite3`。空值关闭本地与随包查询；只想用随包数据时可给一个不存在的本地路径。读取不创建数据库，写入是显式操作。`FLASHINFER_NPU_TUNING_DIR` 当前不控制这条存储路径。

硬件的 SoC、核数和 UB 必须完整，unknown/零值不能建立可信匹配。bucket 来自实际声明 shape_ranges 的交集，其他轴、dtype、布局和语义保持精确；区间可复用不代表区间内每个点都已测过。

v1 查询校验记录中的比较者；新增候选本身不会使旧记录失效。执行身份由可选的 `runner.tuning_identity` 提供，未统一强制覆盖所有后端；发布导出校验 Descriptor/Capability，但不调用该钩子重算执行身份。采集方还需补完整策略与 admission 筛选；采集器本身只检查 capability/target。

同进程写入会刷新查询与决策缓存。其他进程写入或原地替换发布数据后使用 `reload_tuning_records()` 或新进程；软件/源码未如实进入声明身份的变更不能靠查表自动识别。

## 发布与治理

发布数据通过 `flashinfer.tuning._release.export_release_data` 显式导出，要求完整硬件、当前 registry 身份、有效原始测量和受控大小，输出到新目录。它不能证明调用方确实完成真实测量，来源审核仍需独立进行。

未来的在线探索、统一软件身份和模型级收益验证不属于当前离线查询的承诺，参见 [规划](roadmap.md)。代码：[benchmark](../flashinfer/tuning/benchmark.py)、[存储与查询](../flashinfer/tuning/store.py)、[执行身份](../flashinfer/tuning/identity.py)、[发布](../flashinfer/tuning/_release.py)。
