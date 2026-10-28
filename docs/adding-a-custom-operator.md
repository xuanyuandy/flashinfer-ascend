# 新增一个自定义算子：操作指南

> 面向要往本仓接入自研 Ascend C 算子的开发者。
> 本文讲**怎么做**与**会踩什么坑**；完整的交付件对照表见
> `docs/reports/custom-operator-onboarding.md` §4（native v2 / catlass v2 双列，33 项），
> 契约字段语义见 `docs/reports/ascendc-contract-v2.md`。
>
> 已有两个可照抄的范本：
> - **native v2**：`matmul_bias_silu`（`silu(A@B+bias)`，BF16）
> - **catlass v2**：`flash_attention`（迁自 catlass `49_ascend950_flash_attention_infer`）

## 1. 先明确接入契约与实现

### 1.1 对外只提供统一 v2 契约

`schema_version=1/2` 是本仓 `BuildRecipe` 的序列化与能力校验版本，不是 Ascend C 的两种编译方式。所核对的 flashinfer-ori 接入链路没有这套 Ascend C recipe 档位；它通过 generator 参数、模板源码和 FFI 导出定义 CUDA 编译单元。

所有公开配方（生产与探针）统一按 v2 接入；`BuildRecipe`、配方读取/解析、配置选择与新发布入口会拒绝 v1。shape 驱动的 host 动态 tiling 是已有能力。旧 v1 仅用于内部 artifact 解析和冻结测试，不是新作者可选的接入方式。

内部兼容 reader 继续保持旧 v1 护栏，排查旧产物时仍可见以下诊断：

```
tiling_schema_version != 1 或 workspace_bytes != 0 或 task_type != "aic_only"
  → SchemaError("Ascend C v1 supports explicit fixed tiling, AIC-only, no workspace")
```

统一契约显式描述以下能力：

- 用到 AIV（向量核），或 AIC/AIV 混合 + 跨核同步
- 需要非零 workspace（哪怕是固定大小）
- tiling 要在 host 按 shape 算，而不是编译期写死
- 参数不是恰好三个 aclTensor

固定 shape、三个 Tensor、零 workspace、AIC-only 同样可用 v2 表达，无需另选一个档位。五条 ACLNN 探针已按该方式迁移；两条裸指针探针及旧源码闭包移入 `tests/fixtures/legacy_ascendc/`，不再作为公开构建输入。内部 v1 的序列化、校验及既有产物身份保持原样。

版本号应按所属对象分别理解：recipe 的 `schema_version=2` 描述接入字段；`tiling_schema_version=2` 描述 host/device 共享的 tiling 布局；冻结声明外层与 `JitSpec` 的 schema 仍可为 1；`ascendc.op_package.v1` 是产物适配格式。这些数字不同不表示存在多套编译器。

FA 已使用 `host_dynamic`：`FiFlashAttentionTiling` 从实际 Q/K/V 的 shape 取得 S、T、heads，调用 tiler 并设置 block 数与 workspace。同一份二进制可覆盖其声明域内的多组 shape。它仍限 dense batch=1、D=128、S/T 为 128..512 的 128 倍数；不读取变长 length Tensor 的内容，不能外推为数据内容驱动 tiling 或任意变长 FA。MatmulBiasSiLU 也使用 shape 驱动 callback。

### 1.2 实现来源：native 还是 catlass

两者**共用**编译器、store、loader 与 op-package 流程，不是两套 runtime。差别只有一处：

- `native`：源码全在本仓，`dependencies: []`
- `catlass`：要在 recipe 里固定 catlass 的 Git revision 与头目录

```json
"dependencies": [{"name": "catlass", "revision": "41bf90da655bba3c66d0acd7e00abe33960ecfd6", "include_dir": "include"}]
```

**wheel 不自带 catlass 头树。**接入方要自己准备 checkout，并通过
`FLASHINFER_CATLASS_ROOT` 指向它；缺失时构建解析阶段会报
`catlass dependency headers are missing`。

### 1.3 统一使用 op_package

**公开构建只支持 v2 `op_package`。** `host_shared` 仅存在于内部旧协议测试设施；公开配置会明确拒绝。
想让算子能被 AOT 运行期自动发现并在无编译器环境命中，也必须是 `op_package`。

## 2. 按这个顺序做

顺序是有讲究的——先把**不支持的输入拒绝干净**，再谈扩大范围。

```
① 定义数学语义与 envelope
     明确 dtype / shape / layout / mask / batch 支持范围，以及数值容差
     ↓
② 写 kernel TU 与 host TU
     csrc/kernels/ascendc/<name>/      设备侧
     csrc/op_package/<name>.cpp        OpDef / InferShape / tiling 回调 / workspace 回调
     ↓
③ 写 recipe
     csrc/recipes/<name>.json          照抄 matmul_bias_silu.json 的字段
     ↓
④ 冻结 spec 并构建
     tools/build_<name>.py             declare_kernel() → export_op_package()
     ↓
⑤ 进登记表 + 接公共入口
     flashinfer/data/kernels/registry.json
     flashinfer/data/kernels/ascendc_specs.json
     flashinfer/ascendc.py             lazy 绑定
     公共 wrapper（命名与同域现有接口一致）
     ↓
⑥ host 侧测试（不需要真机）
     recipe 校验 / dispatch 门禁 / 非法配置拒绝 / 依赖缺失
     ↓
⑦ 真机测试（@pytest.mark.real_compile）
     数值对 golden + §4 的七个验收场景
```

## 3. 这几轮实际踩到的坑

需要 Torch 算子入口时，参考 [Torch/ACLNN 接入说明](aclnn-torch-adapter.md) 与 `flashinfer/aclnn/_ops.py`：用具名 schema、`PrivateUse1` 实现和 FakeTensor 元数据函数复用公共 facade。完整 wheel 的 `package_tensor_runner` 再调用 `csrc/aclnn_torch_adapter/adapter.cpp`，完成 Tensor 描述符、workspace、stream 与 `OpCommand` 生命周期；不要另写一个绕过已验证 loader 的裸 `dlopen` 路径。该适配器当前覆盖完整档的有序 Tensor ABI，新增 scalar/attribute 等参数仍须单独扩展契约。

按踩中频率排，不是理论风险。

| 坑 | 症状 | 怎么避 |
|---|---|---|
| **改了源码没重算摘要** | 构建报 digest 不符，或命中了旧产物 | recipe 的 `sources`/`headers`/`host_sources` 都带 SHA256，**改任何源文件都要重算并重新冻结 spec** |
| **改了 descriptor 没重新冻结** | 产物身份对不上 | kernel fingerprint **包含能力与 ABI 元数据**，改 descriptor 同样要重冻 |
| **解释器用错** | `ModuleNotFoundError: torch` 或大批测试 skip | 必须用 `/root/miniforge3/envs/flashinfer-a5/bin/python`；默认 `python3` 和 `/usr/local/python3.12.14` **都没装 torch** |
| **把 skip 当 pass** | 真机用例静默跳过还以为通过 | 跑前自检 `torch.npu.is_available()` 应为 `True 8`；`-m real_compile` 应 **0 skipped** |
| **登记表计数断言被撞** | 新增生产条目后若干测试失败 | **这是设计如此**。断言钉死了精确集合，新增 kernel 必须被显式确认。更新时继续钉死精确集合，**不要放宽成 `>= 1` 或"包含"** |
| **虚报入图能力** | `graph_capability` 写了 `CAPTURE_SAFE` 却没验证 | 没做过捕获/重放验证就写 `NOT_CAPTURABLE`。类名带 `CUDAGraph`、参数带 `use_cuda_graph` **都不构成证据** |
| **单进程多包** | 第二个不同 op-package 加载被拒 | `load_op_package` 目前限单进程一个 package identity，已知缺口 |
| **CANN 路径含点** | vendor 注册找不到 JSON | CANN 9.2 的 `NnopbaseReadJsonConfig` 按路径**第一个 `.`** 推导 JSON 名（`.objects`、`~/.cache` 都会中招）。loader 已用 `/proc` 目录 fd 别名绕开，接入方不必自己处理 |
| **传 `--timeout`** | pytest 报 unrecognized arguments | `pytest-timeout` **未安装** |
| **把 tiling 的数据依赖当 shape** | 契约表达不了 | `shape_constraints` 绑定的是 **shape 符号**。tiling 若依赖张量**内容**（如 `actual_seq_lengths`），当前契约**无法表达**——要么缩范围，要么另扩一档契约并写独立校验。**不要硬编码常量绕过** |

## 4. 验收门槛

七个场景是硬门槛，可直接照 `tests/test_flash_attention_real.py` 的结构套用：

| # | 场景 | 期望 |
|---|---|---|
| 1 | 数值对 golden | 多组 shape，给出误差与容差 |
| 2 | AOT 命中 + 禁 JIT + 编译器不可见 | `source=aot`、`compiler_calls=0`，数值正确 |
| 3 | 产物缺失 + 允许 JIT | 回落真实编译，数值正确 |
| 4 | 产物缺失 + 禁 JIT | 明确报错，`fallback=false` |
| 5 | 产物损坏（JIT 仍开启） | **拒绝**，`compiler_calls=0`。损坏**不得**被当成缺失而静默重编译 |
| 6 | 超出 envelope | 在 **dispatch 阶段**拒绝，不得静默算错 |
| 7 | 外部依赖缺失（catlass 场景） | 构建解析阶段明确报错 |

另外两条全局要求：

```bash
python -B -m pytest -p no:cacheprovider -q                 # 0 failed
python -B -m pytest -p no:cacheprovider -q -m real_compile # 0 failed, 0 skipped
```

## 5. 扩大 envelope 时

envelope 不是文档里的一句话，它同时落在**四个地方**，扩范围必须四处同步：

1. `csrc/recipes/<name>.json` 的 `shape_constraints`
2. descriptor 的 `capability`（`dtypes` / `soc_models` / `shape_ranges`）
3. host 侧 tiling / OpDef 的实际支持
4. dispatch 门禁与对应的拒绝测试

**漏掉任何一处，结果都是"声明支持但算错"或"能算却被拒"。**
扩完要补对应的数值用例，不能只放开约束。

## 6. 相关文档

| 文档 | 用途 |
|---|---|
| `docs/reports/ascendc-contract-v2.md` | v1→v2 契约 diff、字段语义、通用 ABI 参数约定 |
| `docs/reports/custom-operator-onboarding.md` | 33 项交付件对照表、FA 迁移实录、七场景实际输出 |
| `docs/aot-runtime-discovery.md` | AOT 发现/匹配/绑定/回落机制 |
| `docs/reports/call-stack-layering.md` | wrapper→generator→JIT/AOT→调用 ABI 分层，与上游对比 |
| `docs/reports/graph-mode-delivery.md` | 图模式接口与资源归属（接入可入图算子时看） |
| `docs/ascendc-compile-options.md` | 当前生产编译选项、驱动/SDK 的参数边界，以及 vLLM-Ascend 自定义算子接入对照 |

## 7. 当前尚未支持的

这些不是"你没做对"，是契约本身还没有：

- **数据内容驱动的 host tiling**（依赖张量里的值，如变长 `actual_seq_lengths`）
- **多 package identity**（单进程只能激活一个）
- **CANN 运行时版本兼容校验**（只校验 target/spec，不比较当前 CANN 版本）
- **生产算子的 aclgraph 入图**（目前仅 single-decode 一条，且口径很窄）

需要其中任何一项，先扩契约并写独立校验，再接算子——
参照 `_validate_v2()` 的写法，**放开限制的同时必须补上对应约束**。
