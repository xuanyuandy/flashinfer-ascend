# 测试组织与运行

测试覆盖算子数值正确性、框架行为和构建交付三个层面。算子用例通过公开 API 与独立参考结果比较，共享 fixture 统一管理设备、内部格式和 AOT 产物准备。

| 目录 | 职责 |
| --- | --- |
| `operators/` | 算子数值、dtype/shape/量化组合、必要的 out/inplace 结果检查 |
| `framework/` | 路由与能力、公共接口契约、计划生命周期、JIT/缓存/图/调优协议 |
| `packaging/` | AOT/ACLNN 产物、损坏与缺包拒绝、wheel/sdist 与安装后验证 |
| `support/` | 共享测试辅助、独立参考脚本，以及需要进程隔离的交付验证 |
| `framework/data/production_capabilities.json` | 独立冻结的能力基线，防止既有支持范围意外变化 |
| `packaging/data/legacy_ascendc/` | 旧格式兼容与拒绝测试的固定 ABI 样本和源码闭包 |

CATLASS 独立参考脚本位于 `support/catlass_golden.py`，按原始源码 SHA256 校验。固定 ABI 样本和能力基线分别由构建交付与框架测试目录维护。

`production_inventory.py` 独立维护审核后的精确生产集合。`framework/test_replay_param_*.py` 是图参数更新的协议与真机测试入口，其路径同时用于图能力声明和证据审计。

根目录 `conftest.py` 是 pytest 自动发现的插件加载入口，公共逻辑位于 `support/pytest_plugin.py`：SoC 选择、设备初始化、路由缓存隔离和共享单元测试夹具。`operators/conftest.py` 将设备、格式及 AOT 准备限定在算子精度测试范围。`framework/test_replay_param_runtime.py` 检查图参数更新的状态、资源与异常清理；`framework/test_replay_param_npu.py` 在声明匹配的 A5 设备上验证实际 capture/update/replay 与 eager 结果。

## 算子精度

`operators/test_sparse_flash_attention.py` 与 `operators/test_grouped_matmul_swiglu_quant.py` 直接执行：

```python
args, options, expected = cases.sample(torch, torch_npu, **configuration)
actual = flashinfer.sparse_flash_attention(*args, **options)
for value, reference, (atol, rtol) in zip(actual, expected, cases.TOLERANCES):
    torch.testing.assert_close(value.cpu().double(), reference.double(), atol=atol, rtol=rtol)
```

输入、独立 CPU 参考计算与容差位于对应算子的 `cases.py`。精度阈值按算子和 dtype 维护，构建准备位于 `operators/conftest.py`，自动设备包构建选择本机 SoC 的生产 AOT 实现。

设备 fixture 为 NZ 样例启用内部格式，并在用例结束后恢复配置；GMM 显式校验输入的 ND/NZ 格式。wheel 安装来源通过独立虚拟环境中的导入路径断言验证。

`packaging/` 独立验证多进程隔离、禁编译执行、冷缓存、队列策略及安装来源等交付契约。

## 编译后复用产物

环境准备、指定算子编译和原生桥构建见 [原生算子构建流程](../README.md#原生算子编译与精度验证)。已有 `build/sfa-aot/` 与 `build/aclnn/` 时，在仓库根目录执行：

```bash
FLASHINFER_OPERATOR_TEST_RELEASE="$PWD/build/sfa-aot" \
FLASHINFER_OPERATOR_TEST_ADAPTER="$PWD/build/aclnn" \
python -B tools/run_regression.py --profile full tests/operators/test_sparse_flash_attention.py
```

`FLASHINFER_OPERATOR_TEST_RELEASE` 指向设备包导出根目录，`FLASHINFER_OPERATOR_TEST_ADAPTER` 指向包含 `libflashinfer_aclnn.so` 和 `manifest.json` 的原生桥目录。指定两者后公共 fixture 复用产物，设置运行时搜索路径并禁止 JIT。未指定的部分由 fixture 自动构建；自动设备包构建选择本机 SoC 的全部生产 AOT 实现。

SFA 子集产物对应单个测试文件；运行整个 `tests/operators/` 需提供当前 SoC 所需的完整产物，或省略复用变量由 fixture 准备。SFA/GMM 精度测试标记为 `real_compile`，使用 `full` 或 `compile` 档；回归入口按本机芯片选择范围并报告其他目标的未执行数量。wheel 安装后使用 `python -B -m flashinfer selftest` 验证，运行位置与导入核对见 [安装产物验证](../README.md#wheel-构建与安装验证)。

## 运行范围

```bash
python -B tools/run_regression.py --profile quick
python -B tools/run_regression.py --profile full
python -B tools/run_regression.py --profile full tests/operators
python -B -m pytest -p no:cacheprovider --collect-only --soc all
```

回归入口默认 `--soc auto`，读取本机芯片。`requires_soc` 只用于有明确硬件前提的真机测试；异芯片用例在收集时 deselected，记录数量，不能写成已验证。普通 pytest 默认 `--soc=all` 保留全芯片收集；指定 `--soc=auto` 才按本机选择。

quick 排除 `real_compile`，仍可能执行 NPU；full 包含所选芯片的编译与数值用例。一次本机 full 不证明其他芯片通过。没有设备、依赖或适用产物必须显式失败；回归入口仍将 skip 或零用例视为失败。

数值容差、精确生产集合、ABI 冻结值和非法输入拒绝条件由独立断言维护；预期变更遵循 [兼容性维护规则](../docs/development/abi-compatibility.md)。
