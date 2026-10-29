# 上手：全量 AOT 构建、打 wheel、安装并自检

本文走一遍完整发布链路：
**构建全部可预编算子 → 打进同一个 wheel → 装到隔离环境 → selftest → 移走产物做反证**。

环境为 8× Ascend950PR / CANN 9.2.0-beta.2。
机制背景见 `framework_design.md` §3（自定义算子接入）、§5（AOT 与 wheel 发布）。

**“全部算子都编译打包”的口径**：43 个 op 中，40 个走 CANN aclnn（无需编译），
1 个 Triton 在运行期 JIT，真正需要 AOT 预编的是 2 个 Ascend C 生产算子。
因此 wheel 应携带**全部可预编算子的产物，以及其余构建所需源码**；
统一的 `csrc/**` 构建输入随包放在 `flashinfer/data/build/csrc/`。

> **先说清楚支持范围**：FA 当前是**受限 envelope** 的生产算子，不是通用 attention。
> 超出范围会在 dispatch 阶段被拒绝（这是设计如此，见 §7）。

## 1. 环境自检

```bash
mountpoint -q /mnt/fi || mount --bind /home/dy/projects /mnt/fi   # 重启后失效，需重做
source /usr/local/Ascend/cann-9.2.0-beta.2/set_env.sh
ls /dev/davinci* | head -3
echo $ASCEND_HOME_PATH                       # /usr/local/Ascend/cann-9.2.0-beta.2
ls $ASCEND_HOME_PATH/bin/bisheng             # 编译器
/root/miniforge3/envs/flashinfer-a5/bin/python -B -c \
  "import torch,torch_npu;print(torch.npu.is_available(), torch.npu.device_count())"
```

最后一条应输出 `True 8`。看不到设备时应明确报告，不能把 SKIP 当作 PASS。

**解释器必须用 `/root/miniforge3/envs/flashinfer-a5/bin/python`**——
默认 `python3` 与 `/usr/local/python3.12.14` 都没有 torch，用错会把依赖缺失误判成功能缺失。

## 2. 一条命令构建全部生产 AOT 产物

```bash
cd /mnt/fi/flashinfer
FLASHINFER_CATLASS_ROOT=/mnt/fi/catlass-ref \
/root/miniforge3/envs/flashinfer-a5/bin/python -B \
  -m flashinfer.aot --all --output-dir /tmp/t10-release
```

全量算子清单从生产登记表 `flashinfer/data/kernels/registry.json` 与
`flashinfer/data/kernels/ascendc_specs.json` 中的冻结 recipe 推导，
不在命令中逐个列名。按配置选择单个 recipe 时，扫描 `csrc/recipes/*.json`。
当前应同时产出：

- `gemm.matmul_bias_silu`：native Ascend C，BF16；
- `ascendc.single_prefill_catlass`：catlass FlashAttention，FP16/BF16，Q/K/V/out 同 dtype。

每个产物导出到 `/tmp/t10-release/<spec-sha>/`，输出中会列出各自 kernel、
spec 与 package 路径。新增生产 AOT 算子会自动纳入全量构建。

本环境一次冷构建的产物如下，两个算子的 `disk_cache_hit` 均为 `false`：

| recipe | spec-sha | bytes |
| --- | --- | --- |
| `matmul_bias_silu` | `4e13377d4b8047efbb9f0c98f8f633b842887dcbaf9741bb2b5f4c080e7334cd` | 897021 |
| `flash_attention` | `521ce132aad2acc1c48ab036fbf6b5b252e024bfd429edac2732bb4315c89b2f` | 941917 |

```text
{"summary": {"built": 2, "failed": 0, "requested": 2, "skipped": 0}}
```

spec-sha 随冻结声明与构建输入变化；后续版本应按输出与 manifest 核对，不要写死这些值。

`FLASHINFER_CATLASS_ROOT` 必须给：**wheel 不自带 catlass 头树**，
recipe 里固定了 catlass 的 Git revision 并校验头树摘要。缺失时全量构建会明确报错并以
非零退出；不能因为 native 算子能构建就报告“全量成功”。
catlass checkout 只在构建时需要；安装完整 wheel 后，selftest 无需该目录。

原有 `tools/build_flash_attention.py` 与 `tools/build_matmul_bias_silu.py`
仍可用于单个算子构建；发布 wheel 推荐使用全量入口。

## 3. 打进 wheel

```bash
cd /mnt/fi/flashinfer
FLASHINFER_CATLASS_ROOT=/mnt/fi/catlass-ref FLASHINFER_AOT_ALL=1 \
/root/miniforge3/envs/flashinfer-a5/bin/python -B setup.py -q bdist_wheel -d /tmp/t10-wheel
```

这一条命令完成“构建全部生产 AOT 算子 + 全部打进 wheel”，无需先执行 §2。
普通 `python3 setup.py bdist_wheel` 现在也默认执行全部生产 AOT 构建；当前 `dy_test` 环境的简化操作见 [wheel 编译与直接调用说明](../README-wheel-conda.md)。
已做过 §2 时，也可改用 `FLASHINFER_AOT_PREBUILT=/tmp/t10-release` 复用导出产物。
得到 `flashinfer-0.1.0.dev0-py3-none-linux_x86_64.whl`。
平台 tag 不是 `any`，因为包里带了设备产物。

下面三个开关**互斥，不能同时给**：

| 环境变量 | 行为 |
| --- | --- |
| `FLASHINFER_AOT_ALL=1` | 打包时构建全部生产 AOT 算子 |
| `FLASHINFER_AOT_PREBUILT` | 把已编好的 release 目录全部导出进包 |
| `FLASHINFER_AOT_CONFIG` | 按配置选择 recipe 构建；可配合 `FLASHINFER_AOT_SOCS` 指定多个目标 |
| 都不给 | 默认构建全部生产 AOT 算子并打入 wheel |
| `FLASHINFER_AOT_ALL=0` | 仅带源码与 Python 包，不带 AOT 产物 |

配置文件的 `recipe_id` 可为 `matmul_bias_silu`、`flash_attention` 或 `all`；
未填写时保留既有 `matmul_probe` 行为。CLI 可用 `--recipe` 覆盖配置中的选择。

确认产物确实进了包：

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python -B - <<'PY'
from pathlib import Path
import json
import zipfile

wheel = next(Path('/tmp/t10-wheel').glob('*.whl'))
with zipfile.ZipFile(wheel) as z:
    for name in sorted(z.namelist()):
        if name.startswith('flashinfer/data/op_packages/') and name.endswith('/manifest.json'):
            print(name)
            print(' ', json.loads(z.read(name))['spec']['kernel_id'])
PY
```

当前应看到两个不同 `<spec-sha>` 目录，对应 §2 的两个生产算子。
产物仍沿用 `flashinfer/data/op_packages/<spec-sha>/` 布局与导出校验。
本次 wheel 含 34 个 `data/op_packages` 文件条目，两个 manifest 的 spec-sha 与 §2 表中一致。

## 4. 装到隔离环境（**这步最容易做错**）

开发环境里 flashinfer 是 **PEP 660 editable 安装**，它通过 `.pth` 注册了一个
**自定义 meta path finder**。该 finder 优先级高于路径查找，
**`PYTHONPATH` 和 `sys.path` 顺序都压不过它**——结果是你以为在测 wheel，
实际 import 的仍是源码树，验证完全失效。

因此**不要用 `--system-site-packages`**。正确做法是建一个干净 venv，
装完 wheel 之后再用一条**纯路径** `.pth` 把 torch 等依赖引进来
（纯路径行不会触发 editable finder）：

```bash
rm -rf /tmp/t10-venv
/root/miniforge3/envs/flashinfer-a5/bin/python -B -m venv /tmp/t10-venv
/tmp/t10-venv/bin/pip install --no-deps --force-reinstall /tmp/t10-wheel/*.whl
echo "/root/miniforge3/envs/flashinfer-a5/lib/python3.11/site-packages" \
  > /tmp/t10-venv/lib/python3.11/site-packages/_deps.pth
```

两个细节：

- **顺序不能反**。先写 `.pth` 再装，pip 会看到父环境的 dist-info 认为"已安装"而跳过。
- **要加 `--force-reinstall`**。省略时 pip 可能静默跳过，`-q` 还会把提示藏掉。

**装完必须自检**，不自检等于没测：

```bash
cd /tmp      # 务必离开源码目录
/tmp/t10-venv/bin/python -B -c "
import flashinfer, pathlib, torch, torch_npu
p = pathlib.Path(flashinfer.__file__).resolve()
expected = pathlib.Path('/tmp/t10-venv/lib/python3.11/site-packages').resolve()
assert p.is_relative_to(expected), p
print('导入自:', p)
print('是 wheel 安装:', p.is_relative_to(expected))
print('带产物:', (p.parent/'data'/'op_packages').is_dir())
print('NPU 可用:', torch.npu.is_available(), torch.npu.device_count())"
```

确认导入路径、随包产物与设备都符合预期，才算隔离成功：

```text
导入自: /tmp/t10-venv/lib/python3.11/site-packages/flashinfer/__init__.py
是 wheel 安装: True
带产物: True
NPU 可用: True 8
```

## 5. 安装后直接 selftest

```bash
cd /tmp
/tmp/t10-venv/bin/python -B -m flashinfer selftest
```

入口只依赖安装包，**不需要源码 checkout 或 `tests/`**。它遍历随包生产登记表，
逐算子执行真机调用并与参考实现数值比对。对随包 AOT 算子，会在禁编译策略下验证命中。
每个算子输出 `PASS`、`FAIL` 或 `SKIP` 及原因，末尾给出汇总。
当前完整 wheel 应为两个生产算子均 PASS，退出码为 0。

本次隔离 wheel 安装后的实际输出如下，退出码为 0：

```text
PASS gemm.matmul_bias_silu: bundled AOT hit; compilation=disabled; CPU reference matched (max_abs=0.00358111, atol=0.008, rtol=0.008); spec=4e13377d4b8047efbb9f0c98f8f633b842887dcbaf9741bb2b5f4c080e7334cd
PASS ascendc.single_prefill_catlass: bundled AOT hit; compilation=disabled; CPU reference matched (max_abs=6.12283e-05, atol=0.001, rtol=0.001); spec=521ce132aad2acc1c48ab036fbf6b5b252e024bfd429edac2732bb4315c89b2f
Summary: 2 PASS, 0 FAIL, 0 SKIP (total=2)
```

没有设备、产物未随包、依赖缺失等都会给出明确原因。**SKIP 不计入成功**，
存在 FAIL 或 SKIP 时命令以非零退出，不能据此声称安装验证通过。

> **不要用 unset `ASCEND_HOME_PATH` 的方式"屏蔽编译器"**。
> CANN runtime 本身依赖这个变量，去掉会得到
> `aclnnFiFlashAttentionGetWorkspaceSize returned 561002` 这类运行时错误，
> 与是否命中 AOT 无关。selftest 会设置禁编译策略；手工调用时用
> `FLASHINFER_NPU_JIT_DISABLE=1`。

## 6. 反证：证明确实命中了产物

必须做反证：把包内产物临时移走，再运行同一个入口。
使用新进程，避免已经加载的动态库影响结果：

```bash
P=/tmp/t10-venv/lib/python3.11/site-packages/flashinfer/data/op_packages
mv "$P" "${P}.bak"
cd /tmp
result=0
/tmp/t10-venv/bin/python -B -m flashinfer selftest || result=$?
mv "${P}.bak" "$P"
echo "selftest exit: $result"
test "$result" -ne 0
```

应逐算子报告随包 AOT 产物缺失，汇总中的 PASS 为 0，退出码非零。
本次反证的实际输出如下；每行 `compilation=disabled` 后的详细候选诊断用 `...` 省略：

```text
SKIP gemm.matmul_bias_silu: AOT artifact not bundled in /tmp/t10-venv/lib/python3.11/site-packages/flashinfer/data/op_packages; compilation=disabled; ...
SKIP ascendc.single_prefill_catlass: AOT artifact not bundled in /tmp/t10-venv/lib/python3.11/site-packages/flashinfer/data/op_packages; compilation=disabled; ...
Summary: 0 PASS, 0 FAIL, 2 SKIP (total=2)
selftest exit: 1
```

如果仍然全绿，必须检查是否导入了源码树、外部产物路径或其他回退路径。
**反证符合预期，第 5 步才算数。**

## 7. envelope 与越界行为

FA 当前只承诺下表范围，超出的在 **dispatch 阶段**拒绝，不会算错：

| 维度 | 支持 |
| --- | --- |
| SoC | Ascend950 |
| dtype | FP16 或 BF16（Q/K/V/out 必须同 dtype） |
| layout | 连续 NHD |
| batch | 隐式 batch=1 |
| Q | `(S, H, 128)` |
| K/V | `(T, H, 128)`，K 与 V shape 相同 |
| S / T | 128~512，且为 **128 的整数倍** |
| H | 1~8 |
| head_dim | 固定 **128** |
| mask / causal | 无 mask、非 causal |
| scale | 固定 `1/sqrt(128)` |

越界示例（`S=300` 不是 128 的倍数）：

```bash
cd /tmp && FLASHINFER_NPU_JIT_DISABLE=1 /tmp/t10-venv/bin/python -B -c "
import torch, torch_npu, flashinfer as fi
q=torch.randn(300,4,128,dtype=torch.float16,device='npu')
k=torch.randn(256,4,128,dtype=torch.float16,device='npu'); v=k.clone()
try: fi.single_prefill_with_kv_cache_catlass(q,k,v)
except Exception as e: print(str(e)[:200])"
```

会得到两级选型的逐条淘汰原因：

```text
无可用 kernel：op=ascendc.single_prefill_catlass，backend policy=auto。逐条淘汰原因：
  第一级：选择实现栈
    [reject] ascendc：栈不可用：栈内全部 1 个实现被过滤；...
```

**这是预期行为**，不是故障。扩大 envelope 需要同步改四处：recipe 的
`shape_constraints`、descriptor 的 capability、host 侧实际支持、dispatch 门禁与拒绝测试。

## 8. 跑仓库自带验收

```bash
cd /mnt/fi/flashinfer
P=/root/miniforge3/envs/flashinfer-a5/bin/python
$P -B -m pytest -p no:cacheprovider -q                            # 至少 1975 passed，0 failed
FLASHINFER_CATLASS_ROOT=/mnt/fi/catlass-ref \
  $P -B -m pytest -p no:cacheprovider -q -m real_compile          # 期望 80 passed, 0 skipped
```

本次实际结果：全量 `1978 passed`，`real_compile` 为
`80 passed, 1898 deselected`；两者均为 0 failed、0 skipped，退出码为 0。
新增的 24 个发布检查由 3 个独立子进程套件执行，隔离 setup/provider 模拟对
原有真机测试进程状态的影响；原有测试及断言保持不变。

**`0 skipped` 比 `80 passed` 更重要**：真机用例在没有 NPU/CANN 的环境里会静默跳过，
届时 `passed` 仍是绿的，但什么都没验证。

`pytest-timeout` 未安装，传 `--timeout` 会报 unrecognized arguments。

## 9. 常见坑速查

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 测的其实是源码树 | 开发环境的 PEP 660 editable finder 优先级最高 | 按 §4 建干净 venv，并做四项自检 |
| `pip install` 静默无效 | pip 认为已安装 | 加 `--force-reinstall`；先装 wheel 再写 `_deps.pth` |
| `GetWorkspaceSize returned 561002` | unset 了 `ASCEND_HOME_PATH`，CANN runtime 失效 | 保留该变量，用 `FLASHINFER_NPU_JIT_DISABLE=1` 禁编译 |
| 构建报 catlass 头缺失 | 未设 `FLASHINFER_CATLASS_ROOT` | wheel 不自带 catlass，需自备 checkout |
| 真机用例大量 skipped | 解释器用错或无设备 | 用 `flashinfer-a5`，并确认 `True 8` |
| `/mnt/fi` 路径不存在 | 重启后 bind mount 失效 | 重新 `mount --bind /home/dy/projects /mnt/fi` |

## 10. 单算子调试与新增算子

发布默认使用 §2 的 `--all` 或 §3 的 `FLASHINFER_AOT_ALL=1`。
单算子调试仍可用 `tools/build_matmul_bias_silu.py`（native 档，无需 catlass）和
`tools/build_flash_attention.py`。Matmul 公共入口是
`flashinfer.matmul_bias_silu(a, b, bias)`，BF16，M/N/K 为 16~256 的 16 倍数。

接入一个全新算子需要哪些交付件，见 `adding-a-custom-operator.md` 与
`reports/custom-operator-onboarding.md`（33 项清单）。
