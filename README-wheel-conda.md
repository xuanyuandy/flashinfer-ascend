# 重新编译 wheel，在新 conda 环境中通过 torch 测试算子

本文按当前 Ascend950PR / CANN 9.2.0、`dy_test` 的 Python 3.12 环境编写。日常使用按下面的快捷流程执行；需要新建环境或保存完整编译日志时，再使用后文的分步说明。

## 常用流程：编译、安装、直接运行

激活 `dy_test` 并加载 CANN 后，在仓库根目录执行：

```bash
source ../../tools/miniconda3/etc/profile.d/conda.sh
conda activate dy_test
source ../../tools/miniconda3/Ascend/ascend-toolkit/set_env.sh
python3 setup.py bdist_wheel
python3 -m pip install --force-reinstall --no-deps \
  dist/flashinfer-0.1.0.dev0-py3-none-linux_x86_64.whl
(cd tests && python3 flash_attention_acceptance.py)
(cd tests && python3 matmul_bias_silu.py)
```

普通 `bdist_wheel` 默认执行生产登记表中五条 Ascend C 算子的 AOT 编译，并把 host 库、设备二进制、manifest 和 Torch/ACLNN C++ 适配库一起打进 wheel。构建会自动查找 `3rdparty/catlass` 或相邻的 `../catlass-ref`，仍校验固定 revision 和头文件摘要。CANN 调用路径使用已安装的 CANN 运行时；Triton 路径沿用其现有编译方式。

作者文件与构建资源见 [接入指南](docs/adding-a-custom-operator.md)：公共定义在 `operators/`，
各后端维护 `implementation.py/runner.py`；生成 JSON 在 `build/generated/`，打包时组装入 `flashinfer/data/`。
下文脚本演示 Ascend950 的三个实例；Ascend910B 的 GMM/SFA 使用对应的
[通用验收](docs/authoring-and-resources.md)，设备不匹配不计通过。

新增 `flashinfer.matmul(a, b, out=out, backend=...)` 使用同一接口路由到 CANN、native Ascend C AOT 或 Triton JIT；共同输入、内存与编译策略见 [Matmul 多后端说明](docs/matmul-backends.md)。native Matmul 的 AOT 包缺失时明确失败。

源码包选项为 `FLASHINFER_AOT_ALL=0`；缺少编译依赖时，默认 AOT 构建会报错。

公开 Ascend C recipe、配置和新发布入口现只接受 v2；旧 `host_shared` 配置会被拒绝，v1 仅保留内部读取兼容。迁移更新了冻结构建身份，升级源码后请重新编译并安装本轮 wheel；验证范围见 [v2 迁移验收](docs/reports/v2-migration-2026-10-28.md)。

## 已安装完整 wheel：直接运行脚本

激活安装了 wheel 的环境并加载 CANN 后，在仓库根目录执行：

```bash
(cd tests && python3 flash_attention.py)
(cd tests && python3 matmul_bias_silu.py)
```

两个脚本都会自行生成 torch Tensor，调用已安装的算子，并与 CPU 结果比较。
FA 脚本无参数时依次测试 FP16、BF16，经 TorchNPU 原生桥调用 Torch functional 与 Python out 入口；输出 dtype 与输入相同，混合 dtype 会拒绝。
`python3 flash_attention_acceptance.py` 和 `python3 matmul_bias_silu_acceptance.py`
在 `tests/` 下无参数执行也会运行对应示例；带参数时保留原有回归验收入口。

| 测试入口 | 实现 | 本文测试输入 |
| --- | --- | --- |
| `flashinfer.matmul_bias_silu(a, b, bias)` | native Ascend C，wheel 内 AOT | BF16，`a[16,48]`、`b[48,32]`、`bias[32]` |
| `flashinfer.single_prefill_with_kv_cache_catlass(q, k, v)` | catlass Ascend C，wheel 内 AOT | FP16/BF16，Q/K/V/out 同 dtype，`q[128,2,128]`、`k/v[256,2,128]` |
| `torch.ops.flashinfer.rmsnorm(out, x, w, ...)` | CANN RMSNorm，经 torch.library 注册 | FP32，`x[4,128]`、`w[128]` |

前两个公共函数直接接收和返回 `torch.Tensor`，自动注册并转发到同名 Torch 算子。完整 wheel 统一通过 C++/OpCommand → ACLNN 适配库执行；直接使用 `torch.ops` 时先注册一次，`enable_aclnn()` 不再切换执行模式。这两个示例可同进程调用，也可分别执行下面的测试命令：

```python
import torch
import torch_npu
from flashinfer.aclnn import enable_aclnn

enable_aclnn()
q = torch.randn(128, 2, 128, device="npu", dtype=torch.float16)
k = torch.randn(256, 2, 128, device="npu", dtype=torch.float16)
v = torch.randn_like(k)
out = torch.ops.flashinfer.single_prefill_with_kv_cache_catlass(q, k, v)
print(out.shape, out.dtype)
```

另一条入口是 `torch.ops.flashinfer.matmul_bias_silu(a, b, bias)`，输入使用表中的 BF16 形状。两者均有 `_out` 版本，参数 `out` 原地写入、返回 `None`。注册、FakeTensor、完整调用链和构建依赖见 [Torch/ACLNN 接入说明](docs/aclnn-torch-adapter.md)。

## 1. 设置构建环境并检查设备

以下路径相对于仓库根目录；迁移机器时先修改这几个变量。

```bash
FI_REPO="$(pwd -P)"
FI_CONDA_BASE="$(realpath ../../tools/miniconda3)"
FI_BUILD_PYTHON="$(realpath ../../tools/conda-envs/dy_test/bin/python)"

source "$FI_CONDA_BASE/Ascend/ascend-toolkit/set_env.sh"
export FLASHINFER_CATLASS_ROOT="$(realpath ../catlass-ref)"
export PATH="$(dirname "$FI_BUILD_PYTHON"):/usr/bin:/bin:$PATH"

test -f "$FI_REPO/setup.py"
test "$(git -C "$FLASHINFER_CATLASS_ROOT" rev-parse HEAD)" = \
  41bf90da655bba3c66d0acd7e00abe33960ecfd6
"$FI_BUILD_PYTHON" -B -c \
  "import torch,torch_npu;print(torch.__version__, torch_npu.__version__);print(torch.npu.is_available(), torch.npu.device_count())"
```

本机版本为 torch `2.10.0+cpu`、torch_npu `2.10.0.post7`，设备自检应得到 `True 8`。看不到 NPU 时先修复环境。

构建会严格检查工具链身份。可提前检查当前工具链是否与全部生产 spec 一致：

```bash
"$FI_BUILD_PYTHON" -B - <<'PY'
import json
from pathlib import Path
from flashinfer.jit import AscendCToolchain

actual = AscendCToolchain.discover().fingerprint
records = json.loads(Path("flashinfer/data/kernels/ascendc_specs.json").read_text())["kernels"]
for record in records:
    assert record["spec"]["environment_fingerprint"] == actual, record["kernel_id"]
print("TOOLCHAIN_MATCH", actual)
PY
```

若不匹配，先确认 CANN、`cmake`、`c++` 和构建 Python；重新冻结生产 spec 需要单独确认，本文不自动执行 `--freeze`。

## 2. 用全新缓存重新编译并打包

默认构建或显式 `FLASHINFER_AOT_ALL=1` 都会编译当前登记的五条生产 Ascend C 算子并收入同一个 wheel。下面每次创建新输出目录和新缓存，便于确认这次确实重新编译。

```bash
cd "$FI_REPO"
mkdir -p "$FI_REPO/build"
FI_WHEEL_RUN="$(mktemp -d "$FI_REPO/build/wheel-manual-XXXXXXXX")"
unset FLASHINFER_AOT_CONFIG FLASHINFER_AOT_PREBUILT FLASHINFER_AOT_SOCS

FI_BUILD_RC=0
FLASHINFER_AOT_ALL=1 \
FLASHINFER_NPU_JIT_DISABLE=0 \
FLASHINFER_AOT_CACHE_DIR="$FI_WHEEL_RUN/aot-cache" \
  "$FI_BUILD_PYTHON" -B setup.py -q bdist_wheel -d "$FI_WHEEL_RUN/dist" \
  > "$FI_WHEEL_RUN/wheel-build.log" 2>&1 || FI_BUILD_RC=$?
printf '%s\n' "$FI_BUILD_RC" > "$FI_WHEEL_RUN/wheel-build.exitcode"
tail -n 30 "$FI_WHEEL_RUN/wheel-build.log"
test "$FI_BUILD_RC" -eq 0

FI_WHEELS=("$FI_WHEEL_RUN"/dist/*.whl)
test "${#FI_WHEELS[@]}" -eq 1
export FI_WHEEL="${FI_WHEELS[0]}"
test -f "$FI_WHEEL"
echo "wheel: $FI_WHEEL"
```

日志中应有三个 `disk_cache_hit: false`，汇总为 `requested=3, built=3, failed=0, skipped=0`。`wheel-build.exitcode` 必须为 `0`。

再检查 wheel 确实包含三个 AOT 包：

```bash
"$FI_BUILD_PYTHON" -B - <<'PY'
import os
import zipfile

with zipfile.ZipFile(os.environ["FI_WHEEL"]) as wheel:
    manifests = [name for name in wheel.namelist()
                 if name.startswith("flashinfer/data/op_packages/")
                 and name.endswith("/manifest.json")]
    assert len(manifests) == 3, manifests
    print("AOT packages:", len(manifests))
    print(*manifests, sep="\n")
PY
```

## 3. 新建 conda 环境，安装运行依赖与 wheel

示例环境名可以修改。请使用一个新名字；不要复用已有 editable 安装的开发环境。这里独立安装 torch/torch_npu，不共享父环境的 site-packages。

```bash
source "$FI_CONDA_BASE/etc/profile.d/conda.sh"
FI_ENV_NAME=flashinfer-wheel-test
conda create --name "$FI_ENV_NAME" python=3.12 pip -y
conda activate "$FI_ENV_NAME"
source "$FI_CONDA_BASE/Ascend/ascend-toolkit/set_env.sh"

# 移到源码目录外，后续所有调用都从这里运行。
FI_RUN_DIR="$(mktemp -d -t flashinfer-wheel-test-XXXXXXXX)"
cd "$FI_RUN_DIR"

python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.10.0+cpu'
python -m pip install 'numpy==1.26.4' 'torch-npu==2.10.0.post7'
python -m pip install --force-reinstall --no-deps "$FI_WHEEL"

export FLASHINFER_NPU_JIT_DISABLE=1
unset FLASHINFER_AOT_PATHS ASCEND_CUSTOM_OPP_PATH FLASHINFER_CATLASS_ROOT

python -B - <<'PY'
from pathlib import Path
import sys
import torch
import torch_npu
import flashinfer

location = Path(flashinfer.__file__).resolve()
print("python:", sys.executable)
print("flashinfer:", location)
print("versions:", torch.__version__, torch_npu.__version__)
print("NPU:", torch.npu.is_available(), torch.npu.device_count())
assert location.is_relative_to(Path(sys.prefix).resolve()), "加载到了环境外的源码"
assert torch.npu.is_available() and torch.npu.device_count() > 0
PY
```

`flashinfer` 路径必须位于新 conda 环境下。保留 `ASCEND_HOME_PATH` 等 CANN 运行时变量；`FLASHINFER_NPU_JIT_DISABLE=1` 让缺少 AOT 产物的情况直接报错。

上述三个示例使用 Ascend C/CANN，运行依赖中无需安装 Triton。若要测试 Triton 算子，另按 [A5 环境说明](docs/a5.md) 配置匹配的 Triton 环境。

## 4. 先运行包内 selftest

```bash
FI_SELFTEST_RC=0
python -B -m flashinfer selftest > "$FI_RUN_DIR/selftest.log" 2>&1 || FI_SELFTEST_RC=$?
printf '%s\n' "$FI_SELFTEST_RC" > "$FI_RUN_DIR/selftest.exitcode"
cat "$FI_RUN_DIR/selftest.log"
test "$FI_SELFTEST_RC" -eq 0
```

无选择器时 selftest 检查五项生产登记；A2 与 A5 实现须分别在匹配设备验收。单项验收可加
`--kernel-id ascendc/attention/sparse_flash_attention/native/Ascend910B`（A2），或选择 A5 的 FA/Matmul ID。
所选范围应为零失败、零跳过；`SKIP` 表示该项未完成，不得把单设备的部分结果写成全套通过。

## 5. 自己用 torch Tensor / torch.ops 调用并对照 CPU

在当前测试目录创建以下脚本。以下三个 case 可分别执行；应用可在同一进程调用两个生产算子。完整 AOT wheel 首次使用时一次性激活当前随包的生产 vendor 集合。加载器保持该集合及各 vendor/op type/API 的不可变注册，同名 generation 升级或追加新 vendor 须新进程。

```bash
cat > wheel_smoke.py <<'PY'
import argparse
from pathlib import Path
import sys

import torch
import torch_npu
import flashinfer as fi

package = Path(fi.__file__).resolve().parent
assert package.is_relative_to(Path(sys.prefix).resolve()), fi.__file__
assert torch.npu.is_available(), "NPU 不可用"
torch.npu.set_device(0)
torch.manual_seed(20261026)


def compare(output, reference, *, atol, rtol):
    torch.npu.synchronize()
    actual = output.cpu().double()
    torch.testing.assert_close(actual, reference.double(), atol=atol, rtol=rtol)
    print(f"max_abs={(actual - reference.double()).abs().max().item():.8g}")


def require_bundled_aot(kernel_id):
    from flashinfer.runtime import get_global_registry
    provenance = get_global_registry().get(kernel_id).runner.provenance
    assert provenance is not None and provenance.source == "aot"
    assert provenance.directory.resolve().parent == (package / "data/op_packages").resolve()
    print("source=aot", "spec=" + provenance.spec_fingerprint)


def matmul():
    from flashinfer.ascendc import MATMUL_BIAS_SILU_KERNEL_ID
    a = (torch.randn(16, 48) * 0.25).bfloat16()
    b = (torch.randn(48, 32) * 0.25).bfloat16()
    bias = (torch.randn(32) * 0.1).bfloat16()
    output = fi.matmul_bias_silu(a.npu(), b.npu(), bias.npu(), backend="ascendc")
    reference = torch.nn.functional.silu(a.double() @ b.double() + bias.double())
    compare(output, reference, atol=0.008, rtol=0.008)
    require_bundled_aot(MATMUL_BIAS_SILU_KERNEL_ID)


def attention():
    from flashinfer.ascendc import FLASH_ATTENTION_KERNEL_ID
    q = (torch.rand(128, 2, 128) * 2 - 1).half()
    k = (torch.rand(256, 2, 128) * 2 - 1).half()
    v = (torch.rand(256, 2, 128) * 2 - 1).half()
    output = fi.single_prefill_with_kv_cache_catlass(
        q.npu(), k.npu(), v.npu(), causal=False, backend="ascendc",
    )
    query, key, value = (x.double().transpose(0, 1) for x in (q, k, v))
    probability = torch.softmax((query @ key.transpose(1, 2)) / (128 ** 0.5), dim=-1)
    reference = (probability @ value).transpose(0, 1)
    compare(output, reference, atol=0.001, rtol=0.001)
    require_bundled_aot(FLASH_ATTENTION_KERNEL_ID)


def rmsnorm():
    from flashinfer.utils import enable_torch_library
    enable_torch_library()
    x = torch.randn(4, 128, dtype=torch.float32)
    w = torch.randn(128, dtype=torch.float32)
    device_x, device_w = x.npu(), w.npu()
    output = torch.empty_like(device_x)
    # schema 的第一个参数是可写 out；这个 torch op 返回 None。
    torch.ops.flashinfer.rmsnorm(output, device_x, device_w, 1e-6, "cann", None)
    reference = x.double() * torch.rsqrt(x.double().square().mean(-1, keepdim=True) + 1e-6)
    compare(output, reference * w.double(), atol=1e-4, rtol=1e-5)


parser = argparse.ArgumentParser()
parser.add_argument("case", choices=("matmul", "attention", "rmsnorm"))
case = parser.parse_args().case
{"matmul": matmul, "attention": attention, "rmsnorm": rmsnorm}[case]()
print("PASS", case)
PY

python -B wheel_smoke.py matmul
python -B wheel_smoke.py attention
python -B wheel_smoke.py rmsnorm
```

每条命令都应打印 `max_abs=...` 和对应的 `PASS`。前两条还必须打印 `source=aot`；数值不满足容差时脚本直接失败。这里测试的是 eager 正确性，没有性能测量。

扩展输入时保持当前生产 envelope：MatmulBiasSiLU 的 M/N/K 均为 16..256 的 16 倍数、连续 BF16；FlashAttention 为 batch=1、连续 FP16 NHD、D=128、相同 heads 1..8、S/T 为 128..512 的 128 倍数、非 causal 且无 mask。两条生产路径的图模式仍未验证。

## 常见问题

| 现象 | 检查 |
| --- | --- |
| `toolchain fingerprint does not match` | 确认构建解释器和 CANN、`cmake`、`c++`；按仓库规则确认后再重新冻结 spec |
| catlass 依赖缺失 | 检查第 1 步的 `FLASHINFER_CATLASS_ROOT` 和固定 Git revision |
| wheel 没有三个 AOT manifest | 检查是否遗留 `FLASHINFER_AOT_ALL=0` 或空值，以及安装的 wheel 是否来自本次输出目录 |
| `flashinfer.__file__` 指向源码树 | 在源码目录外执行，使用新 conda 环境并 `--force-reinstall` 安装本次 wheel |
| `GetWorkspaceSize returned 561002` | 检查 CANN 运行时变量及继承的 `ASCEND_CUSTOM_OPP_PATH`；日志中的具体原因需要进一步定位 |
| `process-scoped` / package identity 冲突 | 同 vendor 的新 generation 或重复 op/API 不可覆盖；升级该算子后启动新进程 |

构建日志保存在 `$FI_WHEEL_RUN`，selftest 日志和示例脚本保存在 `$FI_RUN_DIR`。最新 Torch/ACLNN、全量回归及 wheel 安装证据见 [接入验收记录](docs/reports/aclnn-integration-2026-10-28.md)。普通构建与 `dy_test` 的既有实测证据见 [默认 AOT wheel 验证报告](docs/reports/default-aot-wheel-2026-10-27.md)；迁移后的早期验证见 [目标机器回归报告](docs/reports/regression-target-2026-10-27.md)。
