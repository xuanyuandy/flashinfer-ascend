# T1 自动验收记录

基线 `b8ba29b`；环境 `/root/miniforge3/envs/flashinfer-a5`；2026-10-21。

- 104 个 Python 模块导入：0 失败（`steps/T1-imports.log`）。
- 853 个测试可收集：0 错误（`steps/T1-collect-02.log`）。
- 全量测试：734 通过、119 失败（`T1-baseline.xml`、`T1-failures.json`）。
- schema/registry/ABI/artifact/corpus/attention schema/dispatch：68 通过、1 失败。
- A5 冒烟：23 项数值检查通过（`steps/T1-a5-smoke.log`）。

## 契约恢复

`T1-contract-map.json` 保存全部使用点、字段及不可变性。除原清单十项外，恢复了使用方实际依赖的 WorkloadSpec、DeviceCapability、C POD、参数/错误 ABI，补齐现有 JIT 导出、manifest/plan 和 CLI 配置导出。保留现代 runner/capability 路径与历史 manifest 路径，未改变现代 dispatch 排序。

冻结测试确认 QuantSpec 与 CFieldABI 均包含 schema_version=1；补齐后原 corpus 和五个 POD 固定指纹全部恢复，不修改断言。剩余 binary/error ABI 历史指纹仍不匹配，未通过硬编码映射或放宽测试掩盖。

## 暴露的问题

1. 单请求/分页/ragged 前端在仓库中确实缺实现。为完成测试收集，四个公开缺失名字暂以明确抛 NotImplementedError 的声明暴露；账本仍为 missing，不能算 T2 已实现。T2.2 必须替换这些声明。
2. single decode/prefill 的现有签名与上游不同，约 117 项前端、provider 与 workspace 测试暴露该问题，归 T2.2/T4/T9 逐项处理。
3. T1 遗留：binary/error ABI 的历史序列化指纹无法仅由已恢复字段完全还原。POD 布局/pack/unpack、logical mutation/order、artifact 校验均已有通过证据。
4. 原通用账本把未接入的 Attention surface 标为 missing，却同时声称有 host reference，校验失败。归 T5，不在 T1 中伪改为 functional。

## 留痕

`events.jsonl` 按时间记录操作；`steps/*.patch` 保存每组修改；`steps/*.log` 保存命令输出；`preexisting.patch` 和 untracked 清单隔离前序 A5 改动。导入/收集门禁与失败清单已交付，按用户持续执行授权进入 P1；用户复核尚未发生。

## P1 后的契约复查

恢复 `QuantSpec` 分组的 `group_size` 必填校验、manifest 要求的 capability binding 版本，以及 `WorkloadSpec.schema_version=1`。
最后一项由冻结 wire 反推验证：只在诊断副本中替换已知错误的 binary header 后，补齐 workload 版本即可得到原始冻结 wire 的完整 SHA-256；未修改运行期指纹或测试断言。
证据为 [`T1-workload-version-recovery.json`](T1-workload-version-recovery.json)。

当前工作区全量回归 997 通过、3 失败（[`steps/P1-full-after-workload.log`](steps/P1-full-after-workload.log)）。
仍失败的是 binary ABI v1、binary ABI v2 和由其 header 派生的 wire 指纹；error ABI 单独检查也尚未匹配。
导入 119 个模块零失败，收集 1000 项零错误。最初 119 项失败清单保留为历史基线；最新状态见 P0/P1 总报告。
