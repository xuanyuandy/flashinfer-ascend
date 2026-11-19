# 测试与支持声明

面向开发者和发布维护者：本页规定怎样证明算子的语义、路由、交付和图能力，避免把静态声明或 skip 写成运行成功。

## 证据分开记录

| 验证面 | 需要的证据 |
| --- | --- |
| API/元数据 | 参数与输出、非法输入、FakeTensor、alias/mutation、接口生命周期 |
| 路由 | 同 op 候选、显式覆盖、自动选择、拒绝理由、未选 runner 不执行 |
| 数值 | 独立 CPU golden，逐 dtype/shape/语义组合及明确容差 |
| 编译与缓存 | 真实编译、源/ABI/环境身份、冷/热路径、跨进程复用与失效 |
| AOT | 禁编译命中、缺包策略、损坏与身份冲突拒绝、workspace/stream/队列 |
| 安装 | 干净环境、源码目录外导入检查、包内资源与 selftest |
| 图 | eager/capture/replay 对照、参数更新、地址/shape/流越界负例 |
| 性能 | 同口径输入、正确性先行、计时边界、原始样本和噪声分析 |

没有设备、依赖或产物导致的 skip 均不计为成功。缺少关键依赖必须明确报告；不能靠放宽校验或精度阈值使测试变绿。

## 回归入口

```bash
python -B tools/run_regression.py --profile quick
python -B tools/run_regression.py --profile compile tests/packaging/test_operator_packages_real.py
python -B tools/run_regression.py --profile full
```

这些命令按改动范围选用，不是要求每次连续执行全部。quick 排除 real_compile，但仍可能用 NPU；compile 只运行所选编译用例；full 包含全部 real_compile，之后无需紧接着重复同一矩阵。入口保留日志、JUnit、pytest 原始退出码；有 skip 或没有执行用例时返回失败。

测试按 [算子精度、框架特性与打包交付](../../tests/README.md) 分组。回归入口默认 `--soc auto`，full 表示适用于本机芯片的完整范围；有明确硬件前提的异芯片用例单独记为 deselected，不代表已验证。`--soc all` 保留跨芯片范围；直接调用 pytest 也默认收集全芯片用例。

仅改文档执行静态核对和受影响检查；编译、缓存、原生桥或设备源码改动补受影响 compile，发布和跨层重构执行 full。环境可见性先检查 Torch/TorchNPU，不能把错误解释器的依赖缺失当成功能缺失。

## 独立 golden 与容差

CPU 参考计算要独立于目标 runner，覆盖真实量化、裁剪、mask、空专家、可选输入及输出组合。需要 FP64 对照时先复制到 CPU 再计算；不要把设备不支持的 dtype 转换当作高精度参考。

容差写在对应算子和 dtype 的测试中，不把某个 Norm 的阈值套到所有算子。验证必须让被测特性实际生效，例如裁剪阈值应产生可观测变化。固定输入与种子便于复现，但单个样例不代表全部合法范围。

## 通用算子验收

schema 算子可提供 `cases.py` 的 sample、CASE_PARAMETERS、TOLERANCES 和可选 verify_effect。`tests/packaging/test_operator_packages_real.py` 收集样例，调用公开接口并检查输出、out 身份、原输入存储、非默认流及队列策略；缺包和损坏用例也复用公共工具。

```bash
python -B -m flashinfer.operators.testing sparse_flash_attention matrix
```

装后验证从干净环境和源码目录外运行：

```bash
python -B -m flashinfer selftest --kernel-id ascendc/attention/sparse_flash_attention/native/Ascend910B
python -B -m flashinfer.operators.testing sparse_flash_attention installed
```

只打包一个 AOT 子集时只验对应范围。源码模式默认 selftest 检查全部生产登记；安装包按 wheel 元数据检查实际本机 SoC 的条目并打印范围。910B 包的八项需要 A2/A3 分别执行四项，其他 SoC 不计为 PASS；显式 ID 锁定该实现，异 SoC 在数值执行前拒绝。必须解释目标设备不匹配与缺包；单设备的部分通过不能写成整个矩阵通过。

## 精确预期与兼容

生产集合由 `tests/production_inventory.py` 独立维护。新增 kernel 必须审核完整集合和数量，继续使用精确相等，不改成包含或下界。ABI 指纹和序列化护栏同样独立于生成器，规则见 [ABI 兼容](abi-compatibility.md)。

图声明和调优记录不能用模拟计时、类名、固定 buffer 或源代码能编译来代替真机证据，分别参见 [图模式](../npu_graph_extensions.md) 与 [Tuning](../tuning.md)。
