# FlashInfer 文档指南

`docs/` 只保存仓库需要长期维护的设计与使用说明，不保存开发过程记录。

## 文档范围

- 总体分层、模块职责和依赖方向；
- FlashInfer Attention API 对标范围与兼容策略；
- `plan()` / `run()` 生命周期、自动算子选择和 provider 接入契约；
- metadata、workspace、tensor、量化 KV、JIT、artifact 与 launcher ABI；
- capability、correctness、trace、发布和扩展所必须满足的规范；
- 仓库当前明确支持、尚未支持和禁止隐式推断的能力边界。

## 不进入文档的内容

- 检查点编号和开发进度流水账；
- 某一次本地或远端测试的通过数量、控制台输出和耗时；
- 临时验证目录、机器地址、账号或环境操作记录；
- wheel 文件名、哈希和一次性安装结果；
- 会随测试集合变化而失效的 case 数量、coverage 数量或通过率。

这些信息分别属于测试代码、CI 输出、版本历史和受控的验证记录，不是仓库设计文档。
文档可以定义测试与证据必须遵守的规则，但不复制某次执行结果。

## 阅读顺序

新增算子先读 [接入指南](adding-a-custom-operator.md)，按作者文件、注册、构建和验收的顺序执行。
文件位置以 [算子组织规范](operator-layout.md) 为准，类型化字段、生成 JSON 与 wheel 资源边界见
[作者声明与资源交付](authoring-and-resources.md)。Ascend C、CANN、Triton 各自维护实现与 runner，
公共 `operators/` 维护逻辑语义；多目标 Ascend C 共享接口与 ABI 一致的 runner。

运行时数据模型、实现选择、AOT/JIT 产物管理、执行边界和 FA 接入规范见
[运行时与编译系统技术设计](runtime-and-compilation-design.md)。
[架构介绍提纲](expert-architecture-overview.md) 用于整理介绍思路，不作为正式设计规范。

运行 A5 上已有的 CANN/Triton 算子及了解当前源码布局，先阅读
[`a5.md`](a5.md)。

本地日常测试、CI / 提交前全量测试及新增编译类测试的标记规则，见
[`testing.md`](testing.md)。

1. [`architecture.md`](architecture.md)：总体目标、分层和架构决策。
2. [`framework_design.md`](framework_design.md)：新仓框架设计——`flashinfer.xx`
   接口对标、CANN / Ascend C / Triton 多 backend 调度、AOT/JIT 双模式与
   kernel 缓存（cache key 含 `ASCEND_RT_VISIBLE_DEVICES`）、tuning 机制与
   自定义 kernel 接入。
3. [`attention/framework.md`](attention/framework.md)：Attention 范围、模式和核心语义。
4. [`attention/frontend_contract.md`](attention/frontend_contract.md)：公开 API 与参数归属。
5. [`attention/plan_run_dispatch_design.md`](attention/plan_run_dispatch_design.md)：wrapper
   持有的 `plan()` / `run()` 生命周期与自动 provider 选择。
6. [`attention/quantization.md`](attention/quantization.md)：量化 KV Cache 语义。
7. [`attention/workspace.md`](attention/workspace.md)：workspace 查询、容量和所有权。
8. [`attention/capability_profile.md`](attention/capability_profile.md) 与
   [`support_matrix.md`](support_matrix.md)：外部算子包进入可执行路由前必须满足的能力声明。
9. [`attention/jit_framework.md`](attention/jit_framework.md)：JIT 目录、缓存身份与加载边界。

其余文件细化 trace、数值、tensor、artifact、ABI、launch 和 execution identity 等独立契约。

## 动态事实来源

- Attention API 对标状态：`tools/metadata/attention_api_parity.json`；
- 外部 provider 能力资料：`tools/metadata/attention_capabilities.json`；实现能力以各后端声明为准；
- 公共接口与实现能力：`flashinfer/operators/<domain>/<op>/operator.py` 与各后端 `implementation.py`；
- recipe/Descriptor/spec：由作者声明生成至 `build/generated/`，构建时组装入 wheel 的 `flashinfer/data/`；
- kernel/artifact 身份：对应的冻结 spec 与经过校验的版本化 manifest；
- conformance case 与覆盖统计：版本化 corpus 和 coverage policy；
- 测试是否通过：CI 或当前测试命令的输出。

设计文档解释这些事实源的含义和约束，但不手工维护它们的动态统计值。

## Matmul 与路由讲解

- [算子接入指南与交付清单](adding-a-custom-operator.md)：共同接入契约、三后端注册与构建交付、按阶段验收。
- [Matmul 接口、登记和源码调用链](matmul-backends.md)：三后端共同契约、选型、编译和执行边界。
- [Dispatcher 数据结构参考](generated/dispatcher-data-models.md)：从源码生成的类型、字段和协作关系。
- [四页架构讲解](presentations/dispatcher.pptx) 与 [PDF 预览](presentations/dispatcher.pdf)：仅维护当前版本。

文档生成入口为 `tools/build_dispatcher_slides.py` 和 `tools/inspect_dispatch_data_model.py`。
文字提取、路由演示与产物检查结果写入 `build/dispatcher-docs/`；历史布局、测试流水和重复预览不进入文档树。

过期的 `TASKS.md`、`tasks/` 和交接状态已从当前文档树清理，历史版本通过 Git 查看。
带基线提交的 `reports/` 保留为时点证据，不作为当前接口与支持范围的事实源。
