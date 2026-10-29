# Torch / ACLNN 算子接入

完整 wheel 为 MatmulBiasSiLU 和 CATLASS FlashAttention 提供可显式启用的 C++ Torch/ACLNN 适配库。参考实现为 `../vllm-ascend/csrc/aclnn_torch_adapter/op_api_common.h`：将 Tensor 转成 ACLNN 描述符，执行两段式 API，准备 workspace，并交给 TorchNPU 的 `OpCommand` 和当前 NPU stream。

## 调用方式

原有 `flashinfer.matmul_bias_silu(...)` 和 `flashinfer.single_prefill_with_kv_cache_catlass(...)` 调用方式及默认 Python Tensor/ACLNN 路径保留。在程序初始化阶段显式注册一次，即可使用 `torch.ops`；启用后原公共函数也走原生适配库：

```python
import torch
import torch_npu
from flashinfer.aclnn import enable_aclnn

enable_aclnn()
a = torch.randn(16, 48, dtype=torch.bfloat16, device="npu")
b = torch.randn(48, 32, dtype=torch.bfloat16, device="npu")
bias = torch.randn(32, dtype=torch.bfloat16, device="npu")
result = torch.ops.flashinfer.matmul_bias_silu(a, b, bias)
out = torch.empty_like(result)
torch.ops.flashinfer.matmul_bias_silu_out(a, b, bias, out)
```

FlashAttention 与 MatmulBiasSiLU 可在同一进程按任意顺序调用，示例见 [wheel 编译与直接调用](../README-wheel-conda.md)。完整 AOT wheel 首次使用时按可信声明一次性校验并激活当前随包的生产 vendor 集合，之后可复用两个包。每个 vendor/op type/API 只允许一个不可变 generation；同名替换或追加新 vendor 要求新进程，注册 schema 不会提前加载包。缺失的同组项不触发 JIT；多个冷 JIT 包须预先构建，再通过 `load_op_packages` 一次性激活。`enable_aclnn()` 可重复调用，导入/注册不加载本仓适配库及算子包、不编译、不申请设备内存；真实 NPU 调用时才加载。Torch 自身的 backend 自动导入行为取决于其运行环境。

| Torch 入口 | 返回与参数 |
| --- | --- |
| `torch.ops.flashinfer.matmul_bias_silu(a, b, bias)` | 返回 BF16 Tensor |
| `torch.ops.flashinfer.matmul_bias_silu_out(a, b, bias, out)` | 写入 `out`，返回 `None` |
| `torch.ops.flashinfer.single_prefill_with_kv_cache_catlass(q, k, v, causal=False, kv_layout="NHD", custom_mask=None)` | 返回 FP16 Tensor |
| `torch.ops.flashinfer.single_prefill_with_kv_cache_catlass_out(q, k, v, out, causal=False, kv_layout="NHD", custom_mask=None)` | 写入 `out`，返回 `None` |

MatmulBiasSiLU 支持连续 BF16，M/N/K 为 16..256 的 16 倍数；FlashAttention 支持连续 FP16 NHD、batch=1、D=128、heads=1..8，S/T 为 128..512 的 128 倍数、非 causal 且无 mask。越界输入、输出重叠、跨设备和非 ND 格式继续拒绝。没有 CPU 数值回退；FakeTensor 实现只校验元数据并推导形状，不执行算子。

## 实际调用链与职责

```text
torch.ops.flashinfer.* (PrivateUse1) 或原 Python 公共接口
  → 原公共 facade / dispatcher：有限支持域、后端选择、out 约束
  → LazyKernelRunner：冻结 spec、AOT 发现或受控 JIT、完整性校验
  → package_tensor_runner：已验证 module + recipe 的 Tensor ABI 校验
  → 首次绑定并复用 libflashinfer_aclnn.so / torch.ops.flashinfer_aclnn._execute
  → TorchNPU OpCommand::RunOpApi callback（按框架配置入队或直接执行）
  → Tensor → aclTensor（storage 基址、offset、shape、stride、dtype）
  → aclnnOpGetWorkspaceSize → host dynamic tiling / executor
  → workspace 分配、recordStream → aclnnOp(workspace, executor, stream)
  → TorchNPU 完成同步后返回 caller，释放描述符及 workspace
```

内部注册只接受已经通过原 loader 验证的 API 函数地址，公共 Torch schema 不接收任意动态库路径或裸指针。适配器不重新查找不受验证的 vendor 库，不绕过损坏拒绝、禁编译策略、缓存身份或 package identity 门禁。有序 Tensor 参数来自完整 recipe，使用首次准备的 libffi 签名调用对应的第一段 ABI。TorchNPU 启用任务队列时提交 `EXECUTE_OPAPI`；框架关闭队列或使用同步 launch stream 时按其直接执行路径处理。本仓不修改全局队列策略。同步 eager 边界通过已导出的 `NPUStream::stream(true)` 在 callback 外准备 stream；callback 内不自行同步，`RunOpApi(sync=true)` 返回后才清理资源。

与参考实现的差异：本仓使用 TorchNPU 的格式查询 API，不复制私有 `NPUStorageImpl` 布局；保持原有同步 eager 生命周期，描述符、Tensor、workspace 到设备完成后才释放。当前没有 executor 缓存，也没有任意 scalar/attribute/optional Tensor/Tensor list 转换，不能声称通用接入这些参数。原生桥只使用 TorchNPU 已导出的调用/stream 接口，不复制其任务队列。

## 构建与兼容

普通 `python3 setup.py bdist_wheel` 和 CONFIG 构建会编译适配库并打入 `flashinfer/data/aclnn/`，同时携带适配库 manifest。构建依赖匹配的 Torch/TorchNPU 头文件和库、CANN、C++17 编译器、libffi 开发头文件及库；Debian/Ubuntu 的开发包名为 `libffi-dev`。host 适配层使用 `-O2 -g0 -fPIC -shared`，C++ ABI 按 Torch 构建值设置；设备 recipe 的编译选项不变，详见 [编译配置](ascendc-compile-options.md)。

适配库无 CPython 扩展 ABI，wheel 沿用平台 tag；它依赖 Torch/TorchNPU C++ ABI，加载时必须匹配 manifest 中的两个版本，并校验库 SHA-256。运行依赖 libffi 与 CANN runtime，安装完整 wheel 后不再编译适配库。

适配库 manifest 的 `schema_version=1` 只描述该 host 二进制元数据格式，与算子 `BuildRecipe.schema_version=1/2` 的兼容/完整契约无关。

`FLASHINFER_AOT_ALL=0` 不编译、不携带适配库；`FLASHINFER_AOT_PREBUILT` 保留无编译器打包模式，只导入已验证 op-package。所有模式在未显式启用原生适配时沿用原 Python Tensor/ACLNN 路径，保留 `OpPackageModule.get_workspace_size` 的既有观测边界。调用 `enable_aclnn()` 后缺少适配库会明确失败；选中适配库后目录损坏、版本不匹配或库加载失败不会静默回退。

源码开发时可显式构建适配库：

```bash
python3 -B -m flashinfer.aclnn.build --output-dir flashinfer/data/aclnn
```

也可用 `FLASHINFER_ACLNN_ADAPTER_PATH` 指向另一目录进行隔离验证；该目录应含 `libflashinfer_aclnn.so` 和 `manifest.json`。构建日志留在输出目录，wheel 构建时移至 `build/aclnn-native-build.log`。

## 验证边界

新增 [主机测试](../tests/test_aclnn_adapter.py) 验证延迟导入、重复注册、FakeTensor、CPU 拒绝、缺失和损坏拒绝；[真机测试](../tests/test_aclnn_adapter_real.py) 冷构建适配库及两条生产 AOT 后，在禁编译策略下执行原 API、Torch functional/out、非默认 stream 与非零 storage offset。

现有 CANN、Triton、plan/run、图资源契约继续由原测试覆盖。生产自研算子的 capture/replay、异步返回、自动微分、完整模型接入、跨 Torch/CANN 版本和其他 SoC 未验证；FakeTensor 注册不提升 `graph_capability`，两条自研路径仍为 `NOT_CAPTURABLE`。实际回归与安装结果见 [初始接入验收](reports/aclnn-integration-2026-10-28.md) 和 [队列及架构重构验收](reports/architecture-refactor-2026-10-28.md)。
