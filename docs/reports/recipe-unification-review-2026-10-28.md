# Ascend C 接入契约归一评审（2026-10-28）

后续已按用户确认实施对外 v2 迁移，当前代码与实测范围见 [迁移验收记录](v2-migration-2026-10-28.md)。本文保留为迁移前的评审时点证据。

**可以归一，建议把新算子接入收敛到当前完整契约。** 本仓的 v1/v2 是早期探针向真实算子演进时留下的 recipe 能力档位，两个生产算子已经都使用 v2。竞品中确有名字带 V2 的算子，但这与要求所有算子在“受限/完整接入档位”之间选择是两回事。

本轮完成源码核对、配方清单和迁移影响分析，未实施契约/ABI 迁移。工作区保留上一轮尚未提交的 ACLNN 接入改动；本次没有改动那些运行时代码。

## 1. 竞品的 V1/V2 指什么

核对本地源码基线：flashinfer-ori `62b5ae4e52e14d841e1b0455dc3cfa7053181741`，vLLM-Ascend `96a69d793457917d5d8fdb6b3b4e18f6f372297f`。下列引用文件均与各自 HEAD 字节一致，摘要见证据目录 `references.json`。

| 对象 | 核对到的实际做法 | 与本仓的区别 |
| --- | --- | --- |
| vLLM-Ascend 算子 API | `csrc/torch_binding.cpp:1991` 同时检测 `aclnnDequantSwigluQuantV2` 及其 GetWorkspaceSize；特定条件下调用旧 API，V2 调用增加 dst_type、round_mode、swiglu_mode 等参数 | 这是具体 API 的签名/语义演进，旧名没有 V1 后缀；并非所有自定义算子都选择一个全局接入 profile |
| vLLM-Ascend adapter | `csrc/aclnn_torch_adapter/op_api_common.h:391` 转换参数，`:705` 的统一 `EXEC_NPU_CMD` 拼接两段式 API 名、准备 workspace 并提交 OpCommand | 同一个 adapter 服务不同 API；所核对实现没有本仓式 BuildRecipe v1/v2 能力档位 |
| FlashInfer 上游 generator | flashinfer-ori: `flashinfer/jit/attention/modules.py:530` 按 backend、dtype、head_dim、variant 等形成编译输入；存在 fa2/fa3 等后端选型 | 这是实现/硬件后端选择，不是受限/完整 recipe 的选择 |
| FlashInfer 上游构建与导出 | flashinfer-ori: `flashinfer/jit/core.py:292`、`:372` 提供共同 build/load 生命周期；`csrc/single_prefill_jit_binding.cu:27` 用 typed FFI 导出 run | 所核对链路未采用本仓这两档 Ascend C 接入契约 |
| 上游元数据格式版本 | flashinfer-ori: `flashinfer/jit/flash_kda.py:325` 仍检查生成模块表的 `schema_version=3` | 竞品也需要元数据兼容版本；不能据此声称竞品“完全没有版本号” |

因此，归一的对象应当是**算子作者使用的接入契约及校验/调用路径**。具体算子 API 可以继续独立演进，文件格式也应继续保留版本号。

## 2. 本仓实际存量

逐条解析 `csrc/recipes/*.json`，共 9 条 recipe：

| 算子 / 用途 | 条数 | recipe schema | 产物 | 现有特征 |
| --- | --- | --- | --- | --- |
| MatmulBiasSiLU 生产算子 | 1 | 2 | op_package | BF16、4 Tensor、shape 驱动 host tiling、动态 workspace、AIC+AIV |
| CATLASS FlashAttention 生产算子 | 1 | 2 | op_package | FP16、4 Tensor、shape 驱动 host tiling、动态 workspace、AIC+AIV |
| Ascend950 native/CATLASS matmul_probe | 2 | 1 | host_shared | 固定 FP32 16×16×16，直接 host launch ABI，stream + 三个设备指针 |
| Ascend950 native/CATLASS matmul_probe | 2 | 1 | op_package | 三 Tensor、固定支持域、零 workspace、AIC-only |
| 其他 SoC 的 native matmul_probe | 3 | 1 | op_package | ascend910b1、ascend910_9391、ascend310p1 的探针声明 |

生产登记中的 Ascend C 条目恰好是前两条，没有 matmul_probe。**当前不存在一份生产 FA v1 和另一份 FA v2 的重复实现**；v1 全部属于探针。其他 SoC 配方的存在不代表本机已完成那些设备的数值验证。

主要代码依据：

- `flashinfer/jit/recipe.py:183`：v1 强制固定 tiling/零 workspace/AIC-only，拒绝显式 Tensor 契约。
- `flashinfer/jit/recipe.py:197`：完整契约要求 op_package、host callback、有序参数及严格 shape/资源校验。
- `flashinfer/jit/recipe.py:234`：v1 隐式返回 `(a,b,c)`，v2 读取 `tensor_parameters`。
- `flashinfer/jit/recipe.py:284`：序列化 v1 时去掉新增字段，以保持原指纹。
- `flashinfer/jit/op_package.py:127`、`:149`：部分二进制/参数/动态 tiling 校验仍与 schema 版本耦合。
- `flashinfer/jit/torch_library.py:48`、`:120`：探针有独立适配路径；当前生产 Tensor runner 是另一条通用路径。
- `flashinfer/jit/recipe.py:326`、`flashinfer/aot/build.py:156`：旧配置不写 recipe_id 时仍选择探针，默认配置还可选择 host_shared；普通 wheel 已另外默认构建生产清单。

## 3. 建议的统一目标

**一个面向新算子的完整声明和接入流程，能力由显式字段描述。** 当前完整 v2 已能作为这个基础，无需为归一再增加一个 v3 能力档位。

| 需要表达的事情 | 统一后的表达 |
| --- | --- |
| 几个参数、顺序、dtype、shape | `tensor_parameters` + `shape_constraints`；三个 Tensor 同样合法，无需固定为四个 |
| 固定支持域 | 声明固定维度或严格范围；不以“v1”代替 shape 约束 |
| shape 驱动动态 tiling | 统一的真实 host callback；FA 已支持，无需另选一个接入版本 |
| 固定 shape 算子的 tiling | 同样经 host callback，限定输入后返回常量 tiling；这不意味着扩大 shape 支持域 |
| workspace | 固定非负值或 dynamic + callback；零 workspace 合法，不应形成单独档位 |
| 核类型/同步 | AIC-only 或 AIC+AIV 及对应严格同步规则；不通过版本号决定能力 |
| 编译与调用 | 同一 recipe 驱动、store、loader、经过验证的 ACLNN API 和 Tensor adapter |

新建/发布入口应明确要求完整契约，并在代码中拒绝新增受限档生产算子；不能只在文档写“请用 v2”。旧产物解析属于兼容边界，不应继续作为作者可选的新接入方式。

以下数字各管各自的兼容边界，不必改成同一个值：

- `BuildRecipe.schema_version`：声明格式及兼容解析。归一当前接入契约不等于永远取消格式演进。
- `tiling_schema_version`：某个算子的 host/device 数据布局；升级必须核对真实结构，不能只改声明。
- `launch_abi` 中的 `.v1/.v2`：各算子的调用身份，不应随 recipe 格式号机械改名。
- `ascendc.op_package.v1`、op-package/JitSpec/原生桥 manifest 的 schema：产物/元数据格式，与能力档位分开。现有生产 v2 本来就使用 op_package.v1。

原 Python adapter 与新 C++ adapter 的显式启用策略也是另一条兼容维度；两者当前都能服务生产 v2。此次归一不应顺带改变默认执行路径。

## 4. 分阶段迁移

| 阶段 | 具体工作 | 完成标准 / 边界 |
| --- | --- | --- |
| 统一新接入入口 | 新建/发布只使用完整声明；文档展示一套字段、校验和示例；旧 v1 从作者选型中移出 | 新算子无需决定 v1/v2；旧读取仍保持原身份，不放宽 v1 校验 |
| 收敛兼容边界 | 将旧解析、序列化和探针适配标为内部兼容；减少生产流程按 schema 分支，改为按已验证参数/任务/资源契约处理 | 原 v1 字节、指纹和拒绝行为保持；不能把读到的 v1 自动改写为 v2 后拿去匹配旧二进制 |
| 迁移 5 条 op-package 探针 | 显式声明 `(a,b,c)` FP32 固定 16×16 shape，保持 AIC-only、零 workspace；实现/命名 host callback，核对 tiling 结构与 SDK metadata，重建产物和身份 | 先验证本机 Ascend950 native/CATLASS；其他目标仍按实际编译/设备证据报告，不因声明迁移而记为已验证 |
| 处理 2 条 host_shared 探针 | 保留为独立底层编译/launch 测试设施，或显式退役并以 ACLNN 探针补足原覆盖 | 不再让生产算子作者选择 raw-pointer launch；不能直接删掉原 compile/load、stream、native/CATLASS 覆盖来换“归一” |
| 移除旧 reader | 盘点旧 release/config/调用方，并明确兼容退场后删除 v1 解析和旧适配 | 这是最终删除版本分支的独立迁移，不能把“不再新建 v1”写成“已经完全没有 v1” |

推荐先完成第一阶段，随后迁移探针并按明确的退场条件完成删除；避免长期把两档能力作为公共接入接口维护。若选择一次性移除全部 v1，必须同时明确放弃哪些旧格式/ABI/config 兼容，并保留对应能力的替代测试。

`host_shared` 是实际差异：`csrc/include/flashinfer_ascend/launch_abi.h:7` 的导出参数为 stream 和三个裸设备指针；ACLNN 第一段接收 Tensor 描述符并返回 executor，第二段再接收 workspace/stream。当前完整契约还显式拒绝 host_shared，因此它不是修改 `schema_version=2` 就能迁移的对象。

## 5. 影响范围与必须保留的门禁

预计涉及 `flashinfer/jit/recipe.py`、`op_package.py`、`torch_library.py`、配置/AOT 入口、探针 host/device/tiling 源码、配方和对应测试。无需为归一新增生产 registry 条目；两条生产算子的 dtype、shape、mask、workspace、图能力不应顺带扩大。

**工具链身份也必须纳入迁移。** `flashinfer/jit/ascendc.py:40` 将 `ascendc.py` 与 `op_package.py` 的文件摘要计入 driver identity，再计入 toolchain fingerprint。因此即便生产设备源码不变，只要调整 loader 的版本分支，也可能导致冻结生产声明与当前工具链不匹配；不能承诺“只是重构 Python，生产 spec 必然不变”。

本轮做了纯内存模拟：仅假设为 `op_package.py` 追加一个换行，toolchain 指纹就从 `226cecda…d250` 变为 `25590ae7…7b96`。没有实际修改该文件。这证明了身份闭包的影响，不是新工具链编译/运行验证。实际迁移应按改动闭包审核、重新冻结受影响身份并重建；不能关闭指纹校验或盲目复用旧缓存。

既有冻结测试明确要求保持旧协议：`tests/test_ascendc_contract_v2.py:50` 验证 v1 序列化、指纹与 `(a,b,c)` 完全一致；`:65` 验证 v1 不得继承完整档能力。归一时这些断言应保留为兼容测试，或在明确退役范围后以同等强度的新断言替换；不得删断言、放宽为“包含/至少”或跳过失败。

实际实施时，改写冻结 ABI/spec 或删除兼容护栏前需按 `AGENTS.md` §4 明确确认迁移范围。本次是评审，尚未执行这些动作。

验收应覆盖：旧产物按约定保留或明确拒绝；新声明/重建身份；Tensor/shape/格式/重叠拒绝；固定和动态 workspace；host tiling；AOT 禁编译命中；冷 JIT；损坏拒绝；native/CATLASS；wheel 三种模式、隔离安装与 selftest；随后串行全量和 real_compile，并保留已有图资源/plan/run 测试。

## 6. 本轮证据与未验证事项

设备检查在用户 `dy_test` 环境、加载 `../../tools/miniconda3/Ascend/ascend-toolkit/set_env.sh` 后执行，实际为 **True 8**。通过实际解析 recipe 和 registry，断言 7 条 v1 / 2 条 v2、生产 2 条且无 probe；另核对参考文件与 HEAD、模拟 driver 身份变化。证据位于 `../../var/flashinfer/recipe-unification-20261028-k4aghxhq/`。

实际核对命令包括：

```bash
rg -n 'schema_version|launch_abi|matmul_probe' flashinfer/jit/recipe.py flashinfer/jit/torch_library.py
rg -n 'V1|V2|schema_version' ../vllm-ascend/csrc/aclnn_torch_adapter
```

配方清单、参考摘要、身份模拟和设备结果分别保存在 `recipe-inventory.json`、`references.json`、`identity-simulation.json`、`device-check.log`。最终核查同时确认本轮没有改变已有运行时代码与冻结数据。

实际执行 `python3 -B <证据目录>/verify-review.py`，输出 `result=PASS, v1=7, v2=2, production=2, runtime_changed=false`、退出码 0；同时通过本地链接、CRLF、62 个围栏及空白检查。该脚本是本轮静态审计，不是归一代码的运行测试。

本轮没有重跑全量和 real_compile。上一轮 ACLNN 接入的实际数字为 **1985 passed** 和 **82 passed、1903 deselected**，均 0 failed / errors / skipped、退出码 0，见 [ACLNN 接入验收](aclnn-integration-2026-10-28.md)；这些不是契约归一后的测试结果。

尚未验证：上述迁移的实际补丁及新构建、旧产物/外部调用方的真实存量、其他 SoC 的设备数值、竞品代码的本机编译执行、归一后的图 capture/replay 和性能。任意 scalar/attribute、optional Tensor/Tensor list 的通用接入仍未完成，归一当前 Tensor 契约不将这些算成已有能力。未声称完成迁移或性能改善。
