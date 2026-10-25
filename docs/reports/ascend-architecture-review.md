# 昇腾 flashinfer 架构评审（R3，终轮）

> 基线：本仓 `infer` @ `fbca115abc`；上游 CUDA 仓 @ `da992ed99`；2026-10-24。
> 本文评审当前实现；`docs/framework_design.md` 用于理解设计意图，不作为能力证明。
> R1 指同目录 `competitor-survey.md`；R2 指 `ascend-parity-analysis.md`。
> 本轮只新增本文；真机结果采用用户提供的独立补测，未在 R3 重跑设备测试。

## 1. 定位、范围与总判断

**本项目是对标 FlashInfer Python 接口与语义的昇腾推理算子库，当前已验证单点执行能力，尚未完成成体系的生产交付。**

主要执行依托 CANN 适配和 Triton，Ascend C 的编译、缓存、显式加载已有真机探针。
项目不是 CUDA 二进制兼容层，也不是完整 serving 引擎；API 名称相同不代表所有参数组合等价。
本评审判断分层、生命周期和交付边界是否合理，不给出未经同口径测量的性能排名。

值得保留的是内容寻址、原子发布、显式 ABI、能力筛选和真实 torch.library 边界。
主要问题集中在三处：**生产自研算子登记、AOT 运行期集成、入图覆盖面**。
此外，ABI 冻结契约当前有三项失败，必须作为交付门禁处理。

| 术语 | 本文含义及评审边界 |
|---|---|
| descriptor / runner | 前者声明候选能力与身份，后者实际执行；登记数量不等于独立算子数或实测数量。 |
| JitSpec | 不可变构建规格；其 fingerprint 才是统一 JIT store 的真实身份。 |
| recipe | Ascend C 源码、依赖、目标、ABI 与产物约束；不是注册后自动上线的插件。 |
| AOT / JIT | 提前或按需准备产物的方式，与 native/CATLASS/Triton 等实现选择正交。 |
| op-package | CANN vendor 布局的 host 库、设备二进制和元数据集合。 |
| torch.compile / aclgraph | 分别解决程序追踪和设备任务捕获重放；前者 fullgraph 通过不证明后者安全。 |
| plan / replay update | 前者准备并约束执行身份；后者在重放前更新已录任务允许变化的参数。 |

成熟度采用三个标记：`[V]` 有限定范围的运行证据，`[P]` 部分接通，`[G]` 明确缺口。
标记描述当前层的边界，不代表全量 dtype、shape、SoC、版本及并发组合均获验证。
机制证据基准见 R1 §5；本仓逐项判定见 R2 §3，本文不重新扩张其能力等级。

## 2. 架构总览：执行主线与构建支线已分层，自动交付连接仍缺失

**分离调度、实现和构建是合理的；当前公共执行主线可运行，但不会自动把一个 artifact 变成可用 runner。**

静态视图按代码实际职责排列，右侧为成熟度；图中缺口不是隐含调用。

```text
+-------------------------------------------------------------------+
| API: norm / GEMM / attention wrappers                 [P] 语义子集 |
+-------------------------------------------------------------------+
| runtime: signature -> registry -> dispatch             [V] 选择契约 |
| Attention: metadata / plan / workspace / provider      [P] 独立体系 |
+------------------------------+------------------------------------+
| 已绑定 runner                | 显式 JitSpec / recipe              |
| CANN / Triton / reference     | build -> verify -> publish -> load |
| [V] 有真机路径，覆盖各异      | [V] host / NPU 探针与缓存契约      |
+------------------------------+------------------------------------+
| CANN 适配 / 原生 Triton       | Ascend C native / 可选 CATLASS     |
| [V] 实际算子                 | [P] 固定 GEMM 探针                |
+------------------------------+------------------------------------+
| torch_npu / ACL Runtime / Ascend AI Core              [V] 真机底座 |
+-------------------------------------------------------------------+
  横切能力：torch.library [P]；aclgraph [P]；tuning 查询 [P]
  [G] 随包 AOT 自动发现 -> runner 绑定 -> 失败回落 JIT
```

下面以一次 eager `rmsnorm` 调用说明数据流；batch Attention 的准备步骤另列在图后。

```text
调用方 Tensor / out / backend 请求
  -> API 校验与输出准备
  -> InputSignature + HardwareSignature
  -> dispatch: capability / target / policy 过滤
  -> 有效 tuning 记录，或默认排序
  -> descriptor.kernel_id -> 已绑定 runner
  -> CANN 算子 -> NPU -> 写入 out -> 返回调用方

显式自研支线：recipe/spec -> build_and_load -> verified module
                                   -> 用户 adapter/runner -> 注册后供 API 调用
AOT 支线：构建/export -> 显式目录 -> load_op_package -> adapter
```

默认排序为 `external > cann > aot > jit > reference`；reference 和 custom 的自动准入有配置约束。
`aot/jit` 是 metadata tier；排序并不检查随包产物是否存在，`fallback_chain` 也不是执行失败重试引擎。
无 runner 时 API 报错，不能将图中的显式支线读成 dispatch 自带的冷启动功能。

batch Attention 的 `plan` 保存 metadata、选择和资源身份；`run` 仍有逐请求 gather、FIA 调用和输出拼接。
通用 `Plan.workspace_bytes` 的零占位与 Attention 自己的 workspace contract 并存，尚无统一设备 scheduler。
CANN composite 中的 CPU/list 转换、临时张量和探针桥的同步，使“run 只有异步 launch”仍是目标。

| 可见集合 | 当前数量与意义 |
|---|---|
| 默认 runtime descriptors | 79：CANN 40、REFERENCE 38、TRITON 1；生产 Ascend C 登记为 0。 |
| 随包 kernel manifest | `flashinfer/data/kernels/registry.json` 为 0 条；CLI 只读此表，不能据其空值否定 runtime。 |
| 默认 Attention package providers | runtime specs 与 capability profiles 均为空；外部注入契约已有。 |

分层判断依据见 `ascend-parity-analysis.md` §2、§8；三个集合必须分别治理和展示。

## 3. 证据口径与冲突裁决

**R1/R2 的主干结论相容；最新真机补测扩充了 R2 的验证范围，并未补上其指出的集成缺口。**

用户提供的环境已确认有 8× Ascend950PR，每卡 HBM 131072MB，NPU 可用且设备数为 8。
CANN 为 9.2.0-beta.2、bisheng 存在；解释器是 `/root/miniforge3/envs/flashinfer-a5/bin/python`。
该环境含 torch 2.10.0+cpu、torch_npu 2.10.0.post7 和 Triton；torch 的 `+cpu` 后缀不否定 NPU 能力。
默认 Python 缺 torch 不能作为“功能未实现”依据；本轮不重新猜测硬件或安装依赖。

| 待裁决陈述 | 本评审采用的事实与判断 |
|---|---|
| “real_compile 只有 11 个，且全是 gcc” | 错误线索已被 R2 §3.1、§10.1 推翻：15 个标记函数，分布 11/2/1/1，参数化后补测 28 passed。 |
| “R2 未做 NPU 编译/数值，所以尚无真机能力” | 推论不成立；R2 陈述的是自身执行范围。用户补测覆盖 bisheng 两种产物、Triton 数值与 NPU fullgraph，全部通过。 |
| “运行期 AOT 加载不存在” | 绝对说法错误，R2 已纠正；显式 loader 存在且被通用 loader 调用，缺的是自动发现及策略集成。 |
| 139 modules 与 138 submodules | 计数口径不同：`tools/check_imports.py` 包含根包，pkgutil 子模块统计不含根包；两者均为零失败。 |
| TASKS 的“66 模块导入阻塞” | 已过时；补测 import 成功、83 个公开符号、138 子模块全部可导入，不能沿用旧阻塞。 |
| “存在 ABI 冻结契约，故兼容性已保证” | 不成立；全量补测 1777 passed、3 failed，三项均为冻结指纹断言；size_bytes 通过不能消除指纹漂移风险。 |

用户补测命令 `python -B -m pytest -p no:cacheprovider -m real_compile -v -rs`：
**28 passed、1752 deselected，0 skipped、0 failed，37.01s**；此处 `python` 指上述专用解释器。
全量套件补测为 **3 failed、1777 passed，44.90s**；不是全量通过，也不是 1777 个 NPU kernel 均被验证。
native `[host_shared]` / `[op_package]`、6 个 Triton 对 FP64/reference 数值 case、NPU Inductor fullgraph 是其中的决定性证据。
最新补测不应外推为 CATLASS 四种组合重新验证；其较早运行证据仍按 R2 §7 的历史口径引用。

R1 只静态调研上游，不能据此宣布本仓性能追平；旧 parity 账本仍基于 `289557772`，也不是本轮覆盖率。
设计中的自动 AOT、独立 artifact 分发和无分配热路径与实现不一致，属于待交付承诺，见 R2 §8。

## 4. 与上游的关键差异：必然差异 / 有意不对标 / 尚未补齐

**芯片生态决定实现底座，项目范围决定不做什么；自动交付和图覆盖的缺失则需要工程补齐，不能借硬件差异解释。**

### 4.1 必然差异：设备代码和运行时不能照搬

| 差异 | 合理性与代价 |
|---|---|
| CUDA/NVCC/cubin 对应 Ascend C/bisheng/NPU ELF 与 CANN package | 目标 ISA、编译器和设备装载格式不同，必须重建产物链；完整性、失效和诊断要求仍应保留。 |
| CUDA stream/graph 对应 ACL/aclgraph | 任务捕获及 host 参数固化机制不同，必须通过已支持的 task update 协议更新；不能假设 Python plan 在 replay 中重跑。 |
| SM/CUDA 专属优化与 SoC/AI Core/UB 约束不同 | CUTLASS、PDL 等实现不能直接移植；应保持数学语义并显式声明性能或参数差异。 |

这些是技术约束；具体采用 ctypes 还是 typed FFI、按 hash 还是 URI 组织缓存，不是硬件强制。
依据见 R1 §3.1–§3.4；R2 §4–§7。

### 4.2 有意不对标：明确范围选择，而非缺口改名

| 明确非目标 | 评审意见及边界 |
|---|---|
| 不模拟 CUDA runtime，不承诺 CUDA 二进制兼容，不逐行翻译 CUDA kernel | 合理，应对标接口语义和生命周期；用户仍需迁移 backend 参数及设备资源管理。 |
| 首期不覆盖上游全部能力域，不提供在线训练/autograd | 聚焦推理合理；R1 已含训练与模型专用扩展，不能将其全集当成本仓当前交付承诺。 |
| 模型 FULL/PIECEWISE、batch 分桶和图生命周期交给调用方 | 符合算子库职责；本仓仍须提供真实的算子 capture 能力、资源约束和参数更新接口。 |

上述选择明确来自 `docs/framework_design.md` §1.3、§3.4；证据边界见 R2 §3、§6。
缓存内容寻址、recipe 快照和 CANN 优先是可评审的工程选择：可增强复现或快速获得执行能力，但仍有治理和性能代价。
目前主 wheel 合包与设计的独立分发目标不一致，不能替项目追认成“有意放弃 provider”。

### 4.3 尚未补齐：通用需求依然成立

| 缺口 | 为什么影响架构成立 |
|---|---|
| 生产 Ascend C 登记与通用 ABI adapter | 探针能编译，尚不能让普通生产算子按统一入口被选择并执行。 |
| AOT 自动发现、匹配、优先命中及失败回落 JIT | wheel 携带产物与 API 使用产物脱节；tier 顺序无法完成部署承诺。 |
| batch 图执行、统一图前准备及更多算子 capture 证据 | 固定 buffer/plan 已有，但缺设备执行与 task update 的完整连接。 |
| 批量预热、软件身份完整的 tuning、多目标运行治理 | 上游已有相应机制；本仓缺口会影响冷启动、升级有效性与扩展交付。 |

不要求照搬上游所有 wheel 角色或编译框架；要求补齐等价的运行行为与可验收契约。
依据见 `competitor-survey.md` §3、§5；`ascend-parity-analysis.md` §3、§9。

## 5. 轴线一：JIT

**能力边界：显式构建、内容缓存、原子发布和加载已可运行且有真机验证；生产 dispatch 自动编译与统一缓存治理尚未接通。**

### 5.1 身份设计合理，但声明完整性由调用者承担

真实 key 是 `JitSpec.to_dict()` 的 canonical JSON SHA-256，即 `JitSpec.fingerprint`。
它绑定 generator、目标、环境、特化、输入、源摘要、flags、entry 和产物路径；Ascend C 再绑定 recipe、构建文件和依赖。
相比只用 shape/SoC 的 key，这能区分同输入下不同源码、编译选项和 ABI，适合可审计构建。
代价是身份闭包必须准确：源或依赖改变后要重新声明 spec，不能期待旧 spec 自动追踪任意 include。
命中校验的是已发布 bytes；miss 时源码与声明 digest 不符会拒绝编译。

顶层 `build_cache_key` / `toolchain_fingerprint` 是 legacy helper，不是真实构建身份。
`build_cache_key` 非测试调用为零；测试还将其 monkeypatch 为 `forbidden`，断言真实路径不使用它。
继续公开二者容易误导扩展者；应明确弃用或文档迁移，不能把这一 API 债务误判为真实缓存身份不足。

### 5.2 冷路径与热路径必须分开

```text
declare spec -> store.lookup + 完整性校验
  hit  -> verified loader -> module/runner
  miss -> policy 检查 -> per-spec flock -> 锁内复查
       -> staging compile -> 校验 -> immutable generation
       -> 原子替换 symlink -> verified loader
```

失败保留诊断，已发布 generation 不被原地覆盖；并发首用由锁内复查去重。
DISABLED/CACHE_ONLY 阻止 miss 编译，但允许有效本地 JIT cache；这不是上游 NVCC 的 AOT-only 语义。
loader 有进程内 LRU；重复 `build_and_load` 仍涉及磁盘验证，热执行应持有 module/runner。

### 5.3 两种编译路线有实现，生产接法不统一

| 路线 | 已接通与代价 |
|---|---|
| Ascend C | discover/resolve → 输入快照 → CMake → bisheng → host_shared 或 op_package；native 两种产物的真机测试通过。 |
| 统一 Triton | 冻结 source/request → worker `triton.compile` → npubin/IR/CPU launcher → 校验恢复；依赖 Triton 3.6.* ABI，有历史 cache-only 运行证据。 |
| 生产 Triton norm | 直接使用原生 `@triton.jit`，最新 NPU 数值测试通过；这些测试不能代替统一 store 的覆盖验证。 |
| Attention generator | 能形成 variant/spec 契约，尚未形成可编译 Attention 源码闭环；公共 batch 不接受该扩展路径。 |

Ascend C 快照并清理隐式编译环境有利于复现；CMake 当前固定 `-j2`，缺统一批量预热和编译资源策略。
统一 Triton 与原生 Triton 并存本身可接受，但 cache、禁编译策略和升级失效必须明确归属。

### 5.4 与上游比较及评审意见

上游主 NVCC 路线用 URI/版本/架构定位，再用 Ninja source/header/command 检查新鲜度。
本仓改用声明式内容寻址，取舍在于较强的显式复现与更严格的输入声明责任，不能笼统比较“谁的 key 更完整”。
上游 CuTe/Cake/NVRTC 本来也有独立缓存路线；不应以虚构的“上游全库统一 key”要求本仓。
优先接通一个生产算子的 spec→runner 生命周期，并明确图前预热；继续增加 generator 外壳收益较低。
没有统一 active-capture 首次 JIT 门禁，也不能把“可手动预构建”表述为“自动保证捕获安全”。

证据见 `ascend-parity-analysis.md` §4；上游见 `competitor-survey.md` §3.1。
真实编译通过范围及 legacy helper 的强化证据按本文 §3、用户补测口径采用。

## 6. 轴线二：AOT

**能力边界：产出可以、显式加载可以且已在 NPU 实跑；缺的是运行期自动发现、AOT 优先命中及失败回落 JIT 的集成。**

### 6.1 打包粒度清晰，但当前只是探针集合

AOT 复用 JIT 的 recipe、store、锁、构建和验证，导出已校验 generation，避免两套编译语义分叉。
当前 CLI 硬编码 `matmul_probe`，粒度为 recipe × target，固定 FP32、M=N=K=16、row-major。
7 条 recipe 中 5 条为 op_package、2 条为 host_shared；AOT CLI 只接受 op_package。
这不是生产 attention/norm/GEMM 的 dtype/head-dim/shape 预编译矩阵。

默认 wheel 携带 Python 和本仓构建输入；AOT config 构建或 PREBUILT 复制把产物附在同一主 wheel。
多 SoC 产物累加在包内，尚无上游 shim/provider 的独立分发与版本发现体系。
下面是生成/安装布局模式，非源码树中已存在的产物：

```text
flashinfer/data/op_packages/<spec-sha>/
  manifest.json              # spec / files 的 hash+size / record
  ascendc-recipe.json         # 输入、工具链与命令 receipt
  op-package.json            # vendor / API 库 / compute unit / recipe
  op-package.run             # 携带，loader 不执行安装器
  packages/vendors/<vendor>/
    op_api/lib/libcust_opapi.so
    op_impl/ai_core/tbe/kernel/<compute-unit>/...  # NPU ELF / CANN JSON
```

### 6.2 显式运行期加载确实接通

`load_op_package(directory, target_soc=...)` 先验证 build directory、format/SoC 和 package 内容。
之后设置 `ASCEND_CUSTOM_OPP_PATH`，经 `ctypes.CDLL` 取得 ACLNN GetWorkspaceSize 与 launch 函数。
通用 JIT loader 的 `ASCENDC_OP_PACKAGE` 分支调用它，故它既不是孤儿函数，也不是未来接口。
Tensor adapter 准备参数、stream/workspace 并执行；最新 `[op_package]` 测试真实构建后以 `--package <dir>` 加载。
旧 `ArtifactManifest.verify()` 无调用不代表 loader 不校验：实际验证来自 store 与 op-package 两层。

### 6.3 自动发现到回落的连接未实现

全仓非文档 `op_packages` 仅命中 `setup.py` 的构建期写入，没有 runtime scanner。
`build_and_load` 只查自身 store；公共 API 消费已绑定 runner，dispatch tier 不负责定位 release。
显式目录缺失、损坏、目标不符或 dlopen 失败会报错，没有自动重编译重试。
因此安装 AOT wheel 后直接调用 API，不能承诺自动使用其中的产物。

loader 当前还限制单进程只激活一个 package identity；同进程加载第二个不同包会被拒绝。
它校验请求 target 与声明一致，却不重新比较当前 CANN 版本；构建记录版本不等于部署 ABI 兼容。
上游有按 Tensor device 选择 AOT module 的机制，本仓 hardware device 0 快照不能替代它。

### 6.4 评审意见与完成标准

复用编译/验证链是正确方向；下一步应复用现有 loader，补最小的发现索引、匹配、runner 绑定和失败策略。
需要区分 missing、corrupt、incompatible、load error，并明确哪些情况允许 JIT、哪些必须拒绝。
验收应在安装后的环境测试：命中时无编译器也运行；缺失/损坏时分别验证允许及禁止 JIT 的行为。
该集成必须有生产算子作为消费者，不能以探针显式目录演练代替公共 API 的部署验收。
是否进一步拆 provider wheel 可随体积和目标矩阵决定；自动发现这一运行需求不能继续留空。

证据见 `ascend-parity-analysis.md` §5；上游分发参照 `competitor-survey.md` §3.2、§4.3。

## 7. 轴线三：入图场景

**能力边界：torch.library 有真实窄入口，aclgraph 仅一条受限 single-decode descriptor；batch 的固定资源与 plan 冻结尚未接成通用 replay。**

### 7.1 三段机制分别成立，组合范围必须诚实

```text
capture 外：分配固定 metadata/workspace -> plan/预热 -> 冻结执行身份
capture 内：run + ReplayParameterUpdates.recording() -> 记录 task/绑定资源
每次 replay 前：更新固定数据 -> before_replay(逻辑参数) -> graph.replay()
```

上图表达目标生命周期；当前 batch 实现前一阶段的资源约束，single-decode 低层 runner 实现 task 更新。
二者不是同一条已交付的通用 batch 路径；recording 下 CANN batch composite 明确拒绝执行。

| 阶段 | 当前机制、目的与代价 |
|---|---|
| 固定 buffer | 保存 CSR/mask/block table 的地址与 metadata，校验容量后原位 copy_；避免换 Tensor 导致捕获地址失效。 |
| plan 冻结 | 首次成功 plan 固定 batch/backend/q_len/kernel 等身份，后续变化要求新 wrapper；限制动态图形状自由度。 |
| 重放前更新 | 校验录制任务和存储绑定，通过 task_update 与 event/stream 同步更新 host 参数；调用方须按协议提供参数。 |

paged prefill 仅在显式提供 `max_total_num_rows` 时检查 Q 容量；不能套用上游“首 plan 自动冻结上限”的结论。
`before_replay` 接收按已录任务顺序排列的参数映射序列，将 `kv_seq_lengths` 映射为 `actual_seq_lengths_kv`。
它在 update stream 调用已录 operation 的更新协议，再由用户 replay；不是重跑完整 Python forward 或 dispatch。
图的创建、FULL/PIECEWISE、分桶和生命周期由调用方负责，这一职责边界合理。

### 7.2 真实可入图覆盖：1/79 是 descriptor 口径

默认 79 条中：**NOT_CAPTURABLE 78、NEEDS_PARAM_UPDATE 1、CAPTURE_SAFE 0**。
唯一条目为 `cann/attention/single_decode_fia_v1`，op 为 `attention.single_decode`。
这不是“1/79 个算子实测通过率”，而是登记表的能力声明，不能从 norm fullgraph 通过增加计数。

| 唯一 aclgraph 路径的约束 | 已有证据与限制 |
|---|---|
| 环境 | Ascend950PR_9579、torch 2.10.0+cpu、torch_npu 2.10.0.post7、CANN 9.2 的既有验证。 |
| 设备输入 | BSND；Q=(1,1,8,64)，K/V=(1,1024,2,64)；FP16/BF16；heads=8/2，scale=0.125，LSE=True。 |
| 动态范围 | 只更新 KV 长度 1..1024；无 causal/window/custom mask，要求固定 workspace。 |
| 存储与入口 | 低层 FIA out 在 capture 时创建并由 recorder 持有；公共 single_decode 缺固定 workspace 参数，不能直接等同该路线。 |

历史测试有一次 capture、十次 replay、output/LSE 一致及禁更新反例；证据不能外推任意 shape/版本。
带 `use_cuda_graph` 的 decode/prefill wrapper 和专用 CUDAGraph 类名只代表资源接口，不改变 NOT_CAPTURABLE 状态。
MoE piecewise、模型级切图和下游完整集成没有同等范围证据。

### 7.3 torch.library 注册面与 aclgraph 正交

`enable_torch_library()` 显式注册 `rmsnorm`、`fused_add_rmsnorm`，包含 mutation 和 fake 契约。
用户 `AscendCOp` 可另行提供 schema/device_impl/meta_impl；已有固定 GEMM 探针桥，非全库自动注册。
最新 native 两种产物及 norm NPU fullgraph 测试通过，证明这些不透明 Tensor op 可被编译器消费。
探针桥仍有 stream 同步和资源释放，因此 fullgraph 零 graph break 不能升级为 aclgraph 安全。
上游通用 helper 在该基线为 identity decorator，本仓 helper 真注册；这项局部实现更完整不等于整体能力超越。

### 7.4 评审意见

保留按 descriptor 声明证据的门禁，先将低层 single-decode 接入明确的公共资源接口。
再完成一个 batch 场景的稳定调度、workspace、参数更新、容量负例与生命周期验证，之后才扩能力声明。
统一图前编译/预热门禁仍缺；现有测试手动 warmup 不能替代生产入口保证。

证据见 `ascend-parity-analysis.md` §6；上游对照见 `competitor-survey.md` §3.3、§3.4。

## 8. 轴线四：自定义算子接入

**能力边界：recipe/CMake/显式加载/torch.library 已串通固定 GEMM 探针；CATLASS 可选，任意新算子仍需补 ABI adapter、runner 和交付接线。**

### 8.1 recipe 是构建契约，CATLASS 是实现依赖

recipe 绑定源码/头摘要、依赖 revision、target、flags、entry、launch ABI、tiling/workspace 和输出形式。
CMake 按 `host_shared` / `op_package` 分流：前者构建 C launcher 共享库，后者生成 ACLNN/proto/tiling 与设备包。
native 与 CATLASS 共用编译器、store 和 loader；CATLASS 不是另一个 runtime、backend 或 AOT 机制。
native 不要求 CATLASS；选择 CATLASS recipe 才需要固定 Git checkout 和匹配头文件，wheel 不自带这些头。
依赖的实际 header 内容进入身份有利于复现，但增加接入者的环境准备和版本管理责任。

当前 native 实现是固定 16×16 FP32 标量 GEMM，主要验证编译和 ABI；不能代表高性能 GEMM。
CATLASS 探针使用 Ascend950 模板构件，也仍固定 16×16；不能凭库名推导通用 GEMM 性能或形状覆盖。
schema 限 AIC、workspace=0、tiling schema=1 等；op_package 只接受单 device translation unit，host_shared 支持多源。
动态 tiling、非零 scratch、AIV/AIC 混合算子需要扩展契约和 driver，当前不能只新增 JSON 即接入。

### 8.2 多平台证据按输出、实现和运行分别判断

| 输出与实现 | 当前目标及证据强度 |
|---|---|
| host_shared，native/CATLASS | 限 Ascend950/dav-3510；A5 有历史实跑，最新补测重新覆盖 native。 |
| op_package，native | A5 有实跑及最新补测；A2/910B、A3/910_93、310P 有历史交叉编译和产物校验，未证实对应设备运行。 |
| op_package，CATLASS | 本仓仅 A5 recipe 有历史实跑；非 A5 在编译前被拒绝。 |
| host CPU 平台与多包进程 | aarch64 host 未验证；SoC 映射不证明跨 host 可用，也不消除单 package identity 限制。 |

每个 target 独立构建和导出；target alias 不是 fat binary，也不是 Tensor device 自动分派。
因此“多 SoC 可编译”与“多平台可运行”必须分列发布。

### 8.3 外部用户端到端接入步骤

下面是接入一个真实新算子的完成序列；“现成”指可复用设施，仍须编写自身源码和执行契约。

| 步骤 | 接入者交付物与边界 |
|---|---|
| 1. 定义语义 | 数学 reference、dtype/shape/layout、mutation、stream、workspace/tiling；先检查当前 recipe 限制。 |
| 2. 实现 device | native 或 CATLASS translation unit；选匹配目标的 Arch/Policy，不能只替换 SoC 字符串。 |
| 3. 实现 host | host_shared 的 C launcher，或 op_package 的 OpDef/infer/tiling；两套调用 ABI 分开处理。 |
| 4. 声明 recipe | 输入 digest、依赖 revision、flags、entry/ABI、target/output；config 选择 recipe，不生成新实现。 |
| 5. 显式构建 | discover/resolve → descriptor + environment/specialization → declare spec → build_and_load；注册本身不触发编译。 |
| 6. 接成 callable | 编写 Tensor/stream/ACL adapter；现成 OpPackageModule 固定三个 aclTensor 参数，GEMM 桥拒绝其他 kernel，必须适配新 ABI。 |
| 7. 接入调度 | 给 descriptor 绑定真实 runner/capability，注册 custom kernel；auto 需 opt-in，显式 kernel_id 单独校验。 |
| 8. 接入编译器 | 定义 AscendCOp 的 schema、mutation、device_impl/meta_impl，再显式 enable；fake 只推导 metadata。 |
| 9. 分别验证 | 真机 eager 数值/负例、FakeTensor/opcheck/fullgraph；需要 aclgraph 时再验证稳定资源、更新和反例。 |
| 10. 分发验收 | 扩 AOT family 枚举或复用 export，安装包后按显式目录加载并禁编译运行；自动发现仍依赖 §6 的集成。 |

该步骤能暴露工作量边界：任意输入/attr 数目的算子不是换 recipe 名称就能沿用 GEMM adapter。
外部 Attention single 有额外参数/callable 注入协议，但 variant 源编译未闭环，public batch 拒绝 jit_args 等扩展。

### 8.4 评审意见

构建契约与可选模板依赖的分离合理，当前推广瓶颈是执行 ABI 的通用性和生产消费者，而非再增加一种实现标签。
建议用一个有实际收益的生产算子贯穿上述十步，形成可复制的接入范例，再按算子需求扩 schema。
CATLASS 应按工作负载收益选择；既不设为全库强依赖，也不承诺失败后自动换 native。
数值、性能、交付和图能力分别验收，避免“编译成功”成为四者的替代指标。

证据见 `ascend-parity-analysis.md` §7；上游扩展路径见 `competitor-survey.md` §4.5。

## 9. 风险与缺口：按交付影响面排序

**最高优先级是恢复契约可信度并打通生产交付；扩大算子清单应以这一闭环为前提。**

| 顺序 | 风险、判断依据 | 建议与完成标准 |
|---|---|---|
| 1 / P0 | ABI 冻结当前为红：全量补测三项 fingerprint 失败，影响跨组件契约可信度。 | 定位 canonical/wire 变化，与消费者判定兼容恢复或正式版本迁移；禁止只更新 golden 消除红灯。 |
| 2 / P0 | 自动 AOT 集成缺失：有产物却不能由公共 API 自动消费，破坏无编译器部署预期。 | 生产算子安装后验证 discovery→匹配→load→runner；缺失、损坏、禁 JIT 分支均可诊断并有明确策略。 |
| 3 / P1 | 生产 Ascend C 登记为零，Attention providers 默认空；探针不能支撑自研覆盖承诺。 | 交付一个生产 recipe/adapter/runner/registry/meta/artifact 闭环，并给实际 workload 数值与性能证据。 |
| 4 / P1 | 图能力仅 1/79，batch 三段未连通；名称和 flag 易造成错误集成预期。 | 明确公共入口的 capture 边界，先补 single-decode 资源接口，再完成 batch 变参、容量和生命周期验证。 |
| 5 / P1 | 生产 Triton 绕过统一 store，tuning 软件身份不完整，发布 tuning 库为空。 | 明确缓存归属和禁编译策略，绑定源码/编译栈身份，采集代表 workload；验证升级后拒绝旧 winner。 |
| 6 / P2 | 多 SoC 主要交叉编译、device 0 快照、单 package 进程限制；CANN 运行版本兼容未闭环。 | 按 Tensor device 建身份/选择，定义多包及版本策略，在目标设备与 host 平台分别验收。 |
| 7 / P2 | CANN composite 有同步/分配/循环，旧任务及 parity 口径滞后。 | 测热路径瓶颈后优化；诊断分别报告 runtime/artifact/provider，更新兼容基线，禁止以导入率代替功能覆盖。 |

三项 ABI 失败分别位于 `tests/test_attention_metadata_wire.py`、`tests/test_attention_quant_launch_abi.py`、
`tests/test_runtime_abi.py`；后者实际 `21d0e29e…` 对期望 `cb528512…`。
`size_bytes` 仍通过说明已有尺寸检查未报警，不能证明字段语义、序列化身份或消费者兼容性没有变化。
现有证据也不足以断言已发生设备数据损坏；应先解释漂移原因，再决定修复和版本策略。
风险细证据见 `ascend-parity-analysis.md` §9、§10；新增全量结果的来源见本文 §3。

## 10. 结论与下一步投入

**当前成熟度是“单点能力已被真机验证、工程契约相当完整、生产交付尚未闭环”，不是纯骨架，也不能定性为已对齐上游或生产可用。**

建议保留现有分层、内容寻址和显式能力契约，按以下顺序投入，而非先扩展更多未连接接口：

1. 恢复 ABI 冻结门禁，明确 golden 漂移的兼容性解释与版本决定。
2. 用一个生产自研算子完成登记、自动 AOT 消费和受控 JIT 回落，形成安装后可验收的纵向闭环。
3. 将窄 single-decode 入图能力交付到公共入口，再扩 batch 覆盖，配套工作负载性能与生命周期证据。

本轮没有实现上述缺口，也未重跑 NPU/全量测试或构建 wheel；这是文档评审范围，不是环境阻塞。
A2/A3/310P 实机、aarch64 host、异构多包、全包图能力及相对性能仍未获同口径验证。
工作区初始已含未跟踪的 `.claude/`、R1、R2；验收按保留原状且仅新增本文计算，不通过暂存或删除掩盖基线。
