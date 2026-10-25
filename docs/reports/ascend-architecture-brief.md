# 昇腾 flashinfer 架构简报

> 基线：本仓 `infer @ fbca115abc` · 上游 `da992ed99` · 2026-10-24
>
> **这是精简入口。**详细证据保留在同目录三份报告里，本文不重复它们的表格：
> `competitor-survey.md`（上游机制基准 45 条）、
> `ascend-parity-analysis.md`（逐项对标与四轴线）、
> `ascend-architecture-review.md`（完整评审）。

**一句话结论：单点能力已被真机验证、工程契约相当完整，但生产交付尚未闭环。**
既不是纯骨架，也不能定性为已对齐上游或生产可用。

成熟度标记：`[V]` 有限定范围的运行证据 · `[P]` 部分接通 · `[G]` 明确缺口

---

## 1. 这个仓库负责什么

```
  上层：模型与推理框架                             ← 不在本仓
  vLLM-Ascend · SGLang · 自研 serving
        │
        │  调用算子 API（对标 FlashInfer 的名字与语义）
        ▼
  ┌─ 本仓职责 ──────────────────────────────────┐
  │  算子 API 与语义对标                 [P]     │
  │  后端选择 dispatch（registry 79 条） [V]     │
  │  构建 / 缓存 / 加载（JIT · AOT）     [V/G]   │
  │  入图资源契约（aclgraph）            [G]     │
  └─────────────────────────────────────────────┘
        │
        ▼
  底座：torch_npu · CANN/ACL · Ascend C · bisheng  ← 不在本仓
        triton-ascend · CATLASS（可选）
```

**明确不做**：CUDA 二进制兼容、逐行翻译 CUDA kernel、训练与 autograd、
图的切分与生命周期（FULL/PIECEWISE、分桶由调用方负责）。

API 名称与上游相同**不代表所有参数组合等价**——这是读这个仓库时最容易踩的误解。

---

## 2. 一次调用怎么走

```
  调用方 Tensor / out / backend 请求
        ▼
  API 校验与输出准备
        ▼
  InputSignature + HardwareSignature
        ▼
  dispatch：capability / target / policy 过滤 → tuning 记录或默认排序
        ▼
  descriptor.kernel_id → 已绑定 runner
        ▼
  CANN 算子 → NPU → 写入 out
```

默认排序 `external > cann > aot > jit > reference`。

**关键事实**：`aot` / `jit` 只是 **metadata tier**，排序时**不会去检查产物是否存在**。
没有已绑定 runner 时 API 直接报错，不会自动去找、去编译。

---

## 3. 与上游的架构对比

两边分层形状几乎一致，**差别集中在一条边上**。先看上游：

```
  上游 FlashInfer（CUDA）

  Python API
        ▼
  backend= 选择 / autotuner
        ▼
  ┌──────────────────────────────────────┐
  │  AOT 命中  ◄── 自动发现 ──► JIT 兜底  │   ← 这条连线存在
  │  wheel 内产物              NVCC/NVRTC │
  └──────────────────────────────────────┘
        ▼
  cubin / .so
        ▼
  torch.library 注册面（该 baseline 实为 identity decorator）
        ▼
  CUDA Graph：wrapper 广泛支持 use_cuda_graph
```

再看本仓，**同样的五层，第三层断开**：

```
  本仓 flashinfer（昇腾）

  Python API                                      [P] 语义子集
        ▼
  dispatch：registry 79 条 descriptor             [V]
        ▼
  ┌──────────────────────────────────────┐
  │  AOT 产物      ╳  无自动发现          │   ← 这条连线缺失  [G]
  │  （能产出、能显式加载）    JIT bisheng │
  └──────────────────────────────────────┘
        ▼
  op-package / host .so                           [V] 真机验证
        ▼
  torch.library 注册面（真注册，非 identity）      [V]
        ▼
  aclgraph：79 条中仅 1 条可入图                   [G]
```

逐层对照：

| 层 | 上游 | 本仓 | 性质 |
|---|---|---|---|
| 设备编译 | NVCC / NVRTC → cubin | bisheng → op-package | **必然差异**，已重建等价链路 |
| 构建身份 | URI + Ninja 新鲜度检查 | `JitSpec.fingerprint` 内容寻址 | **工程选择**，本仓更利于复现与审计 |
| AOT→运行期 | 自动发现 → 命中 → 回落 JIT | **仅显式目录加载** | **尚未补齐**，不是硬件决定 |
| torch.library | 该 baseline 是 identity decorator | 真实注册，含 mutation/fake 契约 | 本仓**局部更完整** |
| 图模式 | CUDA Graph，wrapper 广泛 | aclgraph，1/79 | **尚未补齐** |

两点要说清楚：

- **`torch.library` 这一层本仓更完整。**上游 `flashinfer-ori: flashinfer/utils.py:400-440` 的
  `register_custom_op` / `register_fake_op` 在 if/else **两个分支都** `return lambda x: x`，
  即该 baseline 上"注册"是 no-op。本仓是真注册。局部更完整不等于整体能力超越。
- **AOT 那条断边不能用硬件差异解释。**产物链已经重建完成且跑通，缺的是集成。

---

## 4. 对比示例：部署一个自研算子后第一次调用

这是最能说明差距的一条路径。**同一个场景，两边走法不同：**

```
  上游：pip install 带 AOT 产物的 wheel，然后直接调 API

   ① API 调用
   ② 查 AOT 产物（wheel 内 jit-cache / cubin provider）
   ③ 命中 ──────────────────► ④ 直接执行          ✅ 无需编译器
      │
      └─ 未命中 ─► JIT NVCC 编译 ─► 执行           ✅ 自动兜底
```

```
  本仓：pip install 带 op-package 的 wheel，然后直接调 API

   ① API 调用
   ② dispatch 按 tier 排序（aot > jit）
   ③ ✗ tier 只是元数据排序，不去定位产物
   ④ ✗ 没有已绑定 runner
   ⑤ ► 报错

  要真正用上那个产物，当前必须由使用者自己做：
      load_op_package(显式目录)  →  自己写 Tensor/stream/ACL adapter
      →  给 descriptor 绑定 runner  →  注册  →  之后 API 才能调到
```

**断点在第 ③ 步**：产物在包里，API 不会去找它。

这不是"AOT 没实现"——`load_op_package()` 是真实可用的加载器，
会做 build directory 校验、format/SoC 校验、设 `ASCEND_CUSTOM_OPP_PATH`、`ctypes.CDLL` 取 ACLNN 函数，
而且已被通用 JIT loader 消费，并在 NPU 上实跑通过。

**准确结论**：**能产出、能显式加载、已真机验证；缺的是「运行期自动发现 + AOT 优先命中 + 失败回落 JIT」这段集成。**
写成"AOT 完全没实现"是低估，写成"AOT 已支持"是高估。

---

## 5. 四条轴线各一句话

| 轴线 | 能力边界 |
|---|---|
| **JIT** `[V/G]` | 显式构建、内容缓存、原子发布与加载已可运行且有真机验证；生产 dispatch 自动编译与统一缓存治理尚未接通。 |
| **AOT** `[V/G]` | 产出可以、显式加载可以且已在 NPU 实跑；缺运行期自动发现、优先命中与回落 JIT。 |
| **入图** `[G]` | torch.library 有真实窄入口；aclgraph 仅 1 条受限 single-decode descriptor，batch 三段未接成通用 replay。 |
| **自定义算子** `[P]` | recipe/CMake/显式加载/torch.library 已串通固定 16×16 GEMM 探针；CATLASS 可选，任意新算子仍需补 ABI adapter、runner 与交付接线。 |

**入图的 1/79 是 descriptor 口径**（登记表的能力声明），不是"算子实测通过率"。
79 条分布：CANN 40 / REFERENCE 38 / TRITON 1；入图能力 NOT_CAPTURABLE 78 / NEEDS_PARAM_UPDATE 1 / CAPTURE_SAFE 0。
唯一可入图者 `cann/attention/single_decode_fia_v1`。

---

## 6. 风险（按交付影响面）

| | 风险 | 完成标准 |
|---|---|---|
| **P0** | **ABI 冻结当前为红**：三项 fingerprint 断言失败（`size_bytes` 仍通过，只有指纹漂移） | 先解释漂移原因再决定修复或版本迁移。**禁止只更新 golden 消红灯** |
| **P0** | **自动 AOT 集成缺失**：有产物却不能被公共 API 自动消费 | 安装后验证 discovery → 匹配 → load → runner；缺失/损坏/禁 JIT 分支均可诊断 |
| **P1** | 生产 Ascend C 登记为 0，Attention providers 默认空 | 交付一个生产算子的完整闭环，并给实际 workload 的数值与性能证据 |
| **P1** | 图能力仅 1/79，batch 三段未连通 | 先补 single-decode 公共资源接口，再做 batch 变参与生命周期验证 |

P2 级（多 SoC 交叉编译未等同实机、CANN 版本兼容未闭环、CANN composite 同步开销）见完整评审 §9。

---

## 7. 结论与下一步

保留现有分层、内容寻址与显式能力契约，按顺序投入：

1. **恢复 ABI 冻结门禁**，明确 golden 漂移的兼容性解释与版本决定。
2. **用一个生产自研算子**打通登记 → 自动 AOT 消费 → 受控 JIT 回落的纵向闭环。
3. **把窄 single-decode 入图能力交付到公共入口**，再扩 batch 覆盖。

扩大算子清单应以上述闭环为前提，而不是先增加更多未连接的接口。

---

## 附：本轮实测口径

| 实测 | 结果 |
|---|---|
| `-m real_compile` | **28 passed · 0 skipped · 37.0s**（真机 8× Ascend950PR） |
| 全量套件 | **1777 passed · 3 failed · 44.9s** |
| 模块导入 | `import flashinfer` 成功，83 公开符号；138 子模块 0 失败 |

解释器须用 `/root/miniforge3/envs/flashinfer-a5/bin/python`（默认 `python3` 无 torch）。
3 条失败全为冻结 C-ABI/wire 指纹断言。
`docs/TASKS.md` 中「T1 阻塞 66 个模块导入」的说法**已过时**，不应继续引用。

不能外推：A2/A3/310P 实机、aarch64 host、异构多包、全包图能力与相对性能均未获同口径验证。
