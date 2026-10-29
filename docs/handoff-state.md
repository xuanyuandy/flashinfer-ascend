# 交接状态：当前进展与待办

> 更新于 2026-10-26，HEAD `8cfbe72`。
> 协作约定见仓库根目录 `AGENTS.md`（代理开工必读）。

## 0. 2026-10-28 调用方式与测试策略增量

2026-10-29 dtype 扩展验证：公开 v2 recipe 增加可选的完整 `dtype_signatures`，旧配方省略时身份保持兼容；FA 同一 kernel_id/op_package 编译 FP16、BF16 两个设备特化，混合 dtype 拒绝。已完成 202 项受影响 UT、12 组 FA 真机数值、隔离 wheel 双 dtype Torch 调用及 selftest；按用户速度要求未重跑全量和完整 real_compile 矩阵，见 [FA 双 dtype 验证](reports/fa-dual-dtype-2026-10-29.md)。

优先参考 [调用模式与公共 UT](operator-call-patterns.md) 和 [完整接口对比](operator-coverage.md)。上游并非全部采用 wrapper/plan/run；本仓公共 UT 按直接函数、有状态 wrapper、run-only 等对应生命周期执行。

当前实查为 **81 descriptor / 45 op，其中 43 个绑定 NPU runner、2 个仅 reference**。下文历史块中的“43 op”混用了口径，不能作为当前登记表总数。`BatchAttention` 已导出但默认 NPU provider 未接通，普通 batch CANN wrapper 与之分开统计。

日常执行 `python3 -B tools/run_regression.py --profile quick`，编译改动补充受影响 `compile`，完整验收执行一次 `full`；full 已包含 real_compile，不机械重复。下文全量数字保留历史时点，本轮实际检查见 [审计报告](reports/operator-audit-2026-10-28.md)。

## 1. 已验证基线（真机 8× Ascend950PR / CANN 9.2.0-beta.2）

```
全量             1978 passed, 0 failed
-m real_compile    80 passed, 0 skipped
登记表             81 descriptor / 43 op
                   backend 分布 ascendc 2 / CANN 40 / REFERENCE 38 / TRITON 1
生产自研算子        2 条（matmul_bias_silu native、flash_attention catlass）
可入图             1 / 81，CAPTURE_SAFE 为 0
能力域             9 个有 op，8 个为空或仅有壳
```

上游对照基线 `da992ed99`（42 个直接子包 / 2656 个 `.py`）。

## 2. 本轮已完成

| 项 | 结果 |
|---|---|
| ABI 冻结门禁 | 从 3 红转全绿。根因是单一对象 `ATTENTION_KERNEL_ERROR_ABI` 漂移经完整 dict 嵌入传播；确认无活产物依赖旧值后重新冻结，**未声称新旧语义等价**。见 `reports/abi-refreeze-2026-10-25.md` |
| AOT 运行期发现 | 接通 discovery → match → bind runner → 受控 JIT 回落；四类错误分流，corrupt 在 JIT 开启时仍拒绝。见 `aot-runtime-discovery.md` |
| Ascend C 契约 v2 | 放开 AIC+AIV、动态 workspace、动态 tiling、变参 ABI；v1 受限档原条件不变。见 `reports/ascendc-contract-v2.md` |
| 首条生产算子 | `matmul_bias_silu`，登记表从 `[]` 到 1 条，纵向闭环真机验证 |
| 图模式公共入口 | `SingleDecodeWithKVCacheWrapper`，单一配置口径。见 `reports/graph-mode-delivery.md` |
| catlass FA 迁移 | 第 2 条生产算子，受限 envelope。见 `reports/custom-operator-onboarding.md` |
| 打包与装后自检 | `FLASHINFER_AOT_ALL=1` 一条命令编全量并打进 wheel；`python -m flashinfer selftest` 不依赖源码树 |
| 设计文档 | `framework_design.md` 改为设计+现状+路线图，按交付主线重排，面向外部专家 |
| 设备源码目录（2026-10-27 增量） | 统一为 `csrc/kernels/ascendc/`，探针与生产源码按算子分目录；schema 位于 `flashinfer/data/kernels/`，见 [目录归一化验证](reports/kernel-layout-2026-10-27.md) |

## 3. 待办（按优先级，详见 `framework_design.md` 附录 B）

### P0

1. **生产算子入图**。两条生产自研算子**都不可入图**；目前唯一可入图路线是
   `cann/attention/single_decode_fia_v1`，且公共 wrapper 写死单一配置
   （Q `(8,64)`、K/V `(1024,2,64)`、heads=8/2、scale=0.125、`kv_seq_len` 1..1024）。
   自评规模**大**。
2. **安装后生产矩阵验收**。`selftest` 已覆盖基本路径，但真实发布环境、
   多 SoC、无编译器镜像的矩阵未验证。
3. **CANN 运行时版本兼容校验**。当前只校验 target/spec，**不比较当前 CANN 版本**；
   构建时记录的版本不等于部署时 ABI 兼容。

### P1

4. **接口差异补齐**。已支持域内仍缺的具名接口见 `framework_design.md` §7.5
   （25 项已逐一核实，如 RoPE 缺 Llama 3.1 全部缩放变体、page 缺 MLA append）。
5. **空能力域**。MLA、稀疏 attention、通信（HCCL 路径）属**尚未补齐**；
   MNNVL / NVSHMEM / CUDA IPC 属**有意不对标**。两者不可混为一谈。
6. **tuning**。采集/store/query 已有，发布库为空，软件身份不足，无公开 autotuner。

### 已知缺口（不在上述优先级内，但不得遗忘）

- **数据内容驱动的 host tiling**：契约表达不了（`shape_constraints` 绑的是 shape 符号）。
  FA 迁移**未**证明 v2 能表达这类 tiling——定长限制来自 catlass 例子本身
  （`fai.cpp:127` 即拒绝 `isVariedLen > 0`），两笔账要分开记。
- **多 package identity**：单进程只能激活一个。
- **下游集成**：没有 vLLM-Ascend / SGLang 实际接入证据。
- **性能**：从未做过同口径测量，文档中无任何性能声称。

## 4. 结构性债务

- `docs/ascendc_build_design.md` 的 `[L7]` 用绝对路径 `/home/dy/projects/...`，换机器即断。
- `tests/` 不随 wheel 分发。`selftest` 覆盖了装后验证，
  但完整用例集仍需源码树——这是**有意取舍**，若要改需权衡包体积与 pytest 运行时依赖。

## 5. 历史任务 prompt

本轮 12 个任务的完整 prompt 保存在 **`/mnt/fi/task-prompts/`**（仓库外）。
可作为后续任务的写法参考，尤其是其中的验收条件与防糊弄约束。

## 6. 对接手方的提醒

本轮的工作方式是：**提出任务 → 执行 → 独立复核**。
复核环节抓到过若干问题，包括一次把"更新文档"做成整篇重写
（1499 insertions / 1327 deletions，代码块从 56 掉到 42），
以及若干"自述通过但实际未验证"的情况。

**自述不等于验证。** 交付后请实际重跑关键命令，而不是采信报告里的数字。
