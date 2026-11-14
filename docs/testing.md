# 测试运行与编译分组

在仓库根目录使用已配置依赖的 Python 环境运行 pytest；A5 环境准备见
[`a5.md`](a5.md)。以下 `python` 指该环境的解释器。

执行前确认实际设备；缺设备、缺依赖或 skip 均不计为成功。回归入口会保存日志、JUnit、
pytest 原始退出码及最终状态，有 skip 或未执行用例时返回失败。

```bash
python -B -c 'import torch,torch_npu;print(torch.npu.is_available(),torch.npu.device_count())'
```

## 本地日常快速检查

```bash
python -B tools/run_regression.py --profile quick
```

`real_compile` 表示测试（含 fixture）会调用真实编译器，或构建、落盘、
加载真实编译产物。例如 gcc 生成 host 共享库，以及 Triton / Ascend C
生成设备产物。JIT 测试即使本次命中缓存，也保留该标记，因为独立运行或
缓存未命中时仍会编译。

快速组保留纯声明、元数据、模拟编译失败等测试，以及调用预编译 CANN
算子的数值测试；NPU 数值测试仍需要相应设备和依赖。分组用于控制编译
开销，不保证每个保留用例都耗时很短。

## CI / 提交前全量检查

```bash
python -B tools/run_regression.py --profile full
```

直接运行 `pytest tests/` 默认执行全部测试，包括 `real_compile`；配置中
没有默认排除条件。CI 和提交前使用完整环境执行全量，以保留编译、缓存、
装载和数值覆盖。
`full` 已包含 `real_compile`，通过后无需紧接着重复相同编译矩阵。

需要单独验证编译组时运行：

```bash
python -B tools/run_regression.py --profile compile
```

## 新增编译类测试

类型化作者目录的元数据与路由检查由 `tests/test_operator_authoring.py`、
`tests/test_author_api.py`、`tests/test_operator_layout.py` 等覆盖。
新增原生算子提供 `cases.py` 的独立 golden、参数矩阵和容差，
`tests/test_operator_packages_real.py` 与 `flashinfer.operators.testing` 复用通用 AOT/out/stream/安装验收。
精确生产清单仍由维护者审核，不由生成器改写。只改文档或清单生成工具时执行静态检查及受影响 UT，
无需重复冷构建与 wheel 安装；编译/设备源码改动补受影响 compile，跨层重构执行 full。

新增用例若会直接或通过 fixture 调用真实编译器、生成或加载真实编译产物，
在测试函数或方法上添加：

```python
@pytest.mark.real_compile
def test_compiled_kernel(tmp_path):
    ...
```

参数化测试的所有实例会继承该标记。仅在整个类或模块的测试都符合定义时，
才使用类标记或 `pytestmark = pytest.mark.real_compile`；pytest 不支持通过
给 fixture 打标记来自动标记依赖它的测试。

新增或调整分组前先用 `python -m pytest tests/ --durations=40 -q` 实测，
结合测试、fixture 和实际编译入口确认行为。耗时受缓存和设备初始化影响；
必要时将 `TRITON_CACHE_DIR` 指向仓库外的空临时目录核实冷缓存行为。
纯声明、注入替身、手写模拟 artifact 字节及编译前校验失败用例留在快速组。
标记和筛选不应改变断言、参数化范围或原有跳过条件。
