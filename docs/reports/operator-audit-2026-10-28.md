# 调用生命周期、算子清单与开发回归验收（2026-10-28）

本轮优先确认 flashinfer-ori 的真实调用方式，并按对应生命周期整改公共数值 UT；随后完成测试目录静态扫描、当前本仓登记表核对与昇腾官方候选对比。没有新增生产算子或改动原有算子实现。
工作基线为 `infer` / `b742c5aa84916cc2a0be2528784cf1bbc93464b5` 加前一轮尚未提交的架构重构；此前工作树被保留。上游为干净的 `62b5ae4e52e14d841e1b0455dc3cfa7053181741`。

## 环境与设备

从仓库根目录计，解释器为 `../../tools/conda-envs/dy_test/bin/python3`，加载用户指定的 `../../tools/miniconda3/Ascend/ascend-toolkit/set_env.sh`，CATLASS 为 `../catlass-ref`。
实际自检：**True 8**；Torch `2.10.0+cpu`、TorchNPU `2.10.0.post7`。`AGENTS.md` 中原服务器绝对路径没有被误用为当前环境。

证据目录为 `../../var/flashinfer/operator-audit-20261028-183232/`，包含初始工作树记录、设备输出、AST 原始数据、官方检索结果、命令、完整测试日志、JUnit 和实际退出码。
压缩包为同级 `operator-audit-20261028-183232-evidence.tar.gz`，校验值另存 `.sha256`；原始失败/负例记录不覆盖最终通过日志。

## 主要结论与改动位置

| 文件:行号 | 本轮改动 |
| --- | --- |
| `tests/test_attention_cann.py:58`、`:113` | 公共 prefill/decode wrapper 经创建、未 plan 拒绝、plan、run 和新输入复用；prefill 另验证重新 plan，保留原数值容差及 out/LSE 检查 |
| `tests/test_gemm_segment.py:117` | Segment wrapper 创建后直接 run；增加新输入/out 复用，不虚构上游不存在的 plan |
| `tests/test_batch_attention_frontend.py:50` | 顶层 `flashinfer.BatchAttention(device="cpu")` 的 plan/run/新 V 复用，明确只是 Host oracle |
| `tools/run_regression.py:18`、`:30` | quick/compile/full 明确选集；full 包含 compile，保存原始退出码，skip 或未执行用例不返回成功 |
| `tools/audit_operator_surface.py:51`、`:450` | 不导入上游的 AST 扫描，恢复导出别名、有限 lazy export、继承、签名及 wrapper 方法；动态/未解析引用单列 |
| `tools/build_operator_comparison.py:128` | 将接口、规范声明、本仓 op、官方候选组分开连接，保留 backend/experimental 视图与本仓实际能力快照 |
| `docs/framework_design.md:268`、`:1120` | 增量同步调用方式、UT/回归策略及 81 descriptor / 45 op 的正确口径；CRLF 和全部 62 个代码围栏保留 |
| `AGENTS.md`、`README.md`、`docs/handoff-state.md`、`docs/support_matrix.md` | 开发默认快速检查，移除机械重复编译要求；区分 optional provider-contract 与已执行 CANN/Ascend C 路线 |

上游并非所有接口采用 wrapper/plan/run：存在直接函数、直接 batch 函数、plan/run 类、run-only 类、begin_forward/forward 旧接口，以及准备函数与执行函数分离的形式。调用方式及对应源码见 [专门说明](../operator-call-patterns.md)。

完整扫描 1416 个 Python 测试目录文件、15311 个静态 test 定义。公共视图保留 1175 个函数/类拼写、959 个规范声明；其他可解析测试入口 4759 个，包含下层、实验、方法及工具。值/常量 2044 个、未解析引用 1671 个、非字面量动态查询 88 处均保留，不能把这些数相加称为独立数学算子数量。
主表、全部明细、摘要与来源见 [接口全集与对比](../operator-coverage.md) 和 [生成数据说明](../generated/operator-coverage/README.md)。

官方检索结果经人工归并为候选组件与限制，未把“查到算子名”当作逐参数等价。例如 RGDR 9.2 与 9.1 的 Ascend950 支持不同；Sparse/MLA 的索引和物理布局不能按名字合并；HCCL 数学通信原语不兼容 CUDA IPC/NVSHMEM 的资源协议。官方来源、文档版本和候选限制保存在 `docs/operator-comparison-rules.json`。

## 实际命令与结果

以下命令实际执行；`$WORK` 指上述证据目录，初始化 CANN/conda 后使用 `python3`。

```bash
python3 -B tools/run_regression.py --profile quick --log-dir "$WORK/lifecycle-quick" tests/test_attention_cann.py tests/test_gemm_segment.py tests/test_batch_attention_frontend.py tests/test_batch_frontend.py
python3 -B tools/run_regression.py --profile quick --log-dir "$WORK/quick-all"
python3 -B tools/run_regression.py --profile quick --log-dir "$WORK/scanner-reviewed" tests/test_operator_surface_audit.py
python3 -B tools/run_regression.py --profile quick --log-dir "$WORK/quick-final"
python3 -B tools/audit_operator_surface.py --upstream ../flashinfer-ori --output-dir "$WORK/upstream"
python3 -B -m tools.build_operator_comparison --inventory "$WORK/upstream/test-api-inventory.json" --upstream ../flashinfer-ori --local-registry "$WORK/local-registry.json" --output-dir docs/generated/operator-coverage
```

| 检查 | 实际结果 |
| --- | --- |
| 初批公共生命周期 UT | 66 passed、0 failed/errors/skipped，pytest 26.68 秒、含启动共 30.048 秒，退出码 0 |
| 首轮全部 quick | 1916 passed、90 deselected、0 failed/errors/skipped，pytest 47.64 秒、含启动共 51.834 秒，退出码 0 |
| 扫描器最终专项 | 5 passed、0 failed/errors/skipped，退出码 0；覆盖别名/继承、反射/实例属性/作用域、有限 lazy 导出、根函数遮蔽模块和参数列绑定 |
| 最终全部 quick | **1918 passed、90 deselected、0 failed/errors/skipped、4 warnings**，pytest **28.89 秒**、含启动共 **32.835 秒**，退出码 **0** |
| skip 拒绝负控制 | 人工构造 1 skipped：pytest 原始退出码 0、回归入口退出码 1，证明不把 skip 变绿；不计入正常 UT 成功数字 |
| 扫描与生成复现 | 再次扫描得到逐字节相同原始数据；13 个生成数据文件重复输出相同；核对全部 1416 份测试源码及 1146 份引用实现摘要 |
| 生产与文档静态审计 | 前一轮已验收的 47 个生产源码/配方/产物声明路径保持；设计文档 CRLF、62 个围栏及其代码块内容保持，182 个本地文档链接和 Git 空白检查通过 |
| 本轮 full / real_compile | **未重跑**。本轮只改 UT、工具与文档，生产实现保持；90 个编译用例明确 deselected，不能据 quick 声称完成冷编译验收 |
| 上一轮完整验收，仅作既有证据 | full 2003 passed，real_compile 90 passed / 1913 deselected，两轮均 0 failed/errors/skipped、退出码 0；不是本轮新增 5 个测试后的 full 数字 |

上一轮完整日志的 testcase 时间为 931.155 秒，其中 real_compile 90 项占 900.844 秒，其余 1913 项占 30.311 秒；又紧接着跑 real_compile 重复了编译相关选集。
本轮保留全部既有测试与断言，日常 quick 只将昂贵编译选集明确 deselect；发布/跨层修改运行 full 一次，编译专项按改动范围选择。实际墙钟耗时会受环境影响，不保证固定耗时或比例。

## 修正的既有结论

1. 先前“81 descriptor / 43 op”的登记表总数不准确。当前实际是 **45 个 op**：43 个至少绑定 NPU runner、2 个仅 reference。43 只能用于前者口径，不能再作为全部 op 总数。
2. 不能说本仓没有 BatchAttention。它有公共 Host/框架接口，但默认 NPU provider 没有生产闭环；不能与三个已运行 CANN batch wrapper 混为一项。
3. 不能把全部上游算子归纳为 wrapper/plan/run，Segment GEMM 甚至没有 plan。UT 现已按相应接口生命周期整改。
4. “CANN 没有某个接口”需要版本和完整语义证据；新版 RGDR 的存在不代表本仓已实现 GDN/KDA，也不能沿用旧产品表断言 Ascend950 永远不支持。
5. MoE 的 EP 接入分支确实存在，但依赖外部包且多 rank 未验证；这不等于完整上游 moe_ep 已支持。

历史报告保持时点记录。本轮在当前清单、设计文档和交接入口中显式给出修正，没有通过重写旧报告隐藏先前结论。

## 未验证与边界

- 未执行上游 CUDA 测试、multi-GPU/EP 通信或每个官方候选接口。静态扫描中的 test 定义、引用和 import 不是数值通过证据。
- 未验证所有 dtype/shape/layout/mask、FP4/MX 物理布局、完整模型、自动微分、生产 Ascend C 图模式或跨 Torch/CANN 版本。
- 未解析任意运行时构造的名称及所有装饰器/extension 附加属性；原始、动态和未解析清单明确保留，候选不是逐参数兼容保证。
- 本轮没有冷 JIT/AOT、wheel 重建或装后 selftest 新结果；前一轮产物/安装结果仍是前一轮证据。没有性能优势结论。

本轮修改尚未提交。完整审计清单与校验结果保存在证据目录，源码与文档可直接审阅。
