# 开始使用

面向终端用户和集成者：本页说明如何安装本仓产物、确认导入位置，并在已声明的设备范围内调用算子。

## 运行环境

包声明 Python `>=3.9`。NPU 调用需要匹配的 Torch、TorchNPU 和 CANN；Triton 实现还需要匹配的 Triton Ascend 工具链。`environment-a5.yml` 提供一个明确钉定版本的 A5 环境模板，不代表所有产品或版本组合都已验证。

先加载本机 CANN 环境，再使用装有依赖的解释器：

```bash
python -B -c 'import torch,torch_npu;print(torch.npu.is_available(),torch.npu.device_count())'
```

设备不可见时先修复环境。不要把无设备导致的 skip 当作验收通过。当前各算子的目标 SoC、dtype 和布局见 [支持矩阵](backends/overview.md)。

## 安装与导入

使用与芯片、主机平台和运行栈匹配的 wheel；A2/A3 共用 `flashinfer_ascend910b`，A5 选择 `flashinfer_ascend950`。导入名均为 `flashinfer`，同一环境只安装一种；pip 不自动探测 NPU。以下示例对应 A2/aarch64：

```bash
python -m pip install --force-reinstall ./flashinfer_ascend910b-0.1.0.dev0-py3-none-linux_aarch64.whl
python -B -c 'import flashinfer;print(flashinfer.__file__)'
python -B -m flashinfer --help
python -B -m flashinfer show-config
```

验证 wheel 时从源码目录外运行，并确认 `flashinfer.__file__` 位于目标虚拟环境。已有 editable 安装可能抢先解析源码；使用干净环境，不以 `--system-site-packages` 共享环境。需要复用依赖路径时在安装 wheel 后配置纯路径文件，再次核对导入位置。

## 一个 Tensor 调用

`mm_bf16` 的 CANN 声明覆盖 Ascend910B/Ascend910C/Ascend950；实际 dtype/布局仍须符合该接口约定：

```python
import torch
import torch_npu
import flashinfer

a = torch.ones((16, 16), dtype=torch.bfloat16, device="npu")
b = torch.ones((16, 16), dtype=torch.bfloat16, device="npu")
y = flashinfer.mm_bf16(a, b, backend="cann")
print(y.shape, y.dtype)
```

普通 `matmul` 的三个内建实现当前都限定 Ascend950 和其 BF16 支持域；不要从 `torch.mm` 在其他产品可用，推断本仓 `gemm.matmul` 的登记范围也相同。细节见 [GEMM](operators/gemm.md)。

## 选择与诊断

`backend` 和 `kernel_id` 只用于声明这些参数的接口。部分上游兼容门面保留原签名，通过全局配置选择；普通 batch Attention 的 Tensor 路径固定 CANN。覆盖差异和拒绝原因见 [路由说明](concepts/backend-routing.md)。

```python
from flashinfer.runtime import get_global_registry

for item in get_global_registry().descriptors(op="gemm.matmul"):
    print(item.kernel_id, item.backend.value, item.compile_mode.value)
```

`python -m flashinfer list-kernels` 当前只列 Ascend C 生产声明；完整目录用 registry 查询。Reference 需要显式选择，不是执行失败时的自动 CPU 兜底。

## AOT 与装后检查

AOT-only 算子需要匹配的预编译包和原生桥。公共函数首次调用自动注册 Torch/ACLNN 接口，无需启用开关。源码模式不提供二进制；缺包、损坏或目标不匹配不能通过手改指纹绕过。

A2 上可对已安装的 SFA 单项执行：

```bash
python -B -m flashinfer selftest --kernel-id ascendc/attention/sparse_flash_attention/native/Ascend910B
```

不指定 ID 时，安装包 selftest 按实际 SoC 选择本机条目：910B 包含 A2/A3 各自的 GMM/SwiGLU/量化、SFA、DsaSparseProlog、CATLASS prefill，共八项；单机检查对应四项并打印范围。950 包含 Matmul、MatmulBiasSiLU、CATLASS prefill。其他 SoC 不计为 PASS，A3 真机数值未验证；各目标需分别验收，所选范围必须零失败、零跳过。编译与 PREBUILT 打包见 [交付说明](ascendc_aot.md)，完整验收规则见 [测试规范](development/testing.md)。
