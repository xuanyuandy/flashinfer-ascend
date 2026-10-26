# 上手：编译 FlashAttention 算子、打 wheel、安装并验证

本文用 **catlass FlashAttention** 走一遍完整链路：
**编译产物 → 打进 wheel → 装到隔离环境 → 不依赖编译器调用 → 反证确实命中了产物**。

文中每条命令与输出都在 8× Ascend950PR / CANN 9.2.0-beta.2 上实跑验证过。
机制背景见 `framework_design.md` §3（自定义算子接入）、§5（AOT 与 wheel 发布）。

> **先说清楚支持范围**：FA 当前是**受限 envelope** 的生产算子，不是通用 attention。
> 超出范围会在 dispatch 阶段被拒绝（这是设计如此，见 §7）。

## 1. 环境自检

```bash
mountpoint -q /mnt/fi || mount --bind /home/dy/projects /mnt/fi   # 重启后失效，需重做
ls /dev/davinci* | head -3
echo $ASCEND_HOME_PATH                       # /usr/local/Ascend/cann-9.2.0-beta.2
ls $ASCEND_HOME_PATH/bin/bisheng             # 编译器
/root/miniforge3/envs/flashinfer-a5/bin/python -B -c \
  "import torch,torch_npu;print(torch.npu.is_available(), torch.npu.device_count())"
```

最后一条应输出 `True 8`。

**解释器必须用 `/root/miniforge3/envs/flashinfer-a5/bin/python`**——
默认 `python3` 与 `/usr/local/python3.12.14` 都没有 torch，用错会把依赖缺失误判成功能缺失。

## 2. 编译 FA 产物

```bash
cd /mnt/fi/flashinfer
FLASHINFER_CATLASS_ROOT=/mnt/fi/catlass-ref \
/root/miniforge3/envs/flashinfer-a5/bin/python -B \
  tools/build_flash_attention.py --output /mnt/fi/fa-release
```

实测 **8.4 秒**，输出 `{"release": "/mnt/fi/fa-release/release/<spec-sha>"}`，总计约 2.3M。
`release/` 是可分发产物，`build-cache/` 是构建中间物。

`FLASHINFER_CATLASS_ROOT` 必须给：**wheel 不自带 catlass 头树**，
recipe 里固定了 catlass 的 Git revision 并校验头树摘要。缺失时构建解析阶段会明确报错。

## 3. 打进 wheel

```bash
cd /mnt/fi/flashinfer
FLASHINFER_AOT_PREBUILT=/mnt/fi/fa-release/release \
/root/miniforge3/envs/flashinfer-a5/bin/python -B setup.py -q bdist_wheel -d /mnt/fi/fa-wheel
```

得到 `flashinfer-0.1.0.dev0-py3-none-linux_x86_64.whl`（约 1.0 MB）。
平台 tag 不是 `any`，因为包里带了设备产物。

两个互斥开关，**不能同时给**：

| 环境变量 | 行为 |
| --- | --- |
| `FLASHINFER_AOT_PREBUILT` | 把已编好的 release 目录导出进包（推荐，快） |
| `FLASHINFER_AOT_CONFIG` | 打包时现编；可配合 `FLASHINFER_AOT_SOCS` 指定多个目标 |
| 都不给 | 纯 wheel，不带任何产物 |

确认产物确实进了包：

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python -c "
import zipfile; z=zipfile.ZipFile('/mnt/fi/fa-wheel/flashinfer-0.1.0.dev0-py3-none-linux_x86_64.whl')
print(len([n for n in z.namelist() if 'data/op_packages' in n]), '个产物条目')"
```

## 4. 装到隔离环境（**这步最容易做错**）

开发环境里 flashinfer 是 **PEP 660 editable 安装**，它通过 `.pth` 注册了一个
**自定义 meta path finder**。该 finder 优先级高于路径查找，
**`PYTHONPATH` 和 `sys.path` 顺序都压不过它**——结果是你以为在测 wheel，
实际 import 的仍是源码树，验证完全失效。

因此**不要用 `--system-site-packages`**。正确做法是建一个干净 venv，
装完 wheel 之后再用一条**纯路径** `.pth` 把 torch 等依赖引进来
（纯路径行不会触发 editable finder）：

```bash
rm -rf /mnt/fi/fa-venv
/root/miniforge3/envs/flashinfer-a5/bin/python -m venv /mnt/fi/fa-venv
/mnt/fi/fa-venv/bin/pip install --no-deps --force-reinstall /mnt/fi/fa-wheel/*.whl
echo "/root/miniforge3/envs/flashinfer-a5/lib/python3.11/site-packages" \
  > /mnt/fi/fa-venv/lib/python3.11/site-packages/_deps.pth
```

两个细节：

- **顺序不能反**。先写 `.pth` 再装，pip 会看到父环境的 dist-info 认为"已安装"而跳过。
- **要加 `--force-reinstall`**。省略时 pip 可能静默跳过，`-q` 还会把提示藏掉。

**装完必须自检**，不自检等于没测：

```bash
cd /tmp      # 务必离开源码目录
/mnt/fi/fa-venv/bin/python -B -c "
import flashinfer, pathlib, torch, torch_npu
p = pathlib.Path(flashinfer.__file__).parent
print('导入自:', p)
print('是 wheel 安装:', 'fa-venv' in str(p))
print('带产物:', (p/'data'/'op_packages').is_dir())
print('NPU 可用:', torch.npu.is_available())"
```

四项都为真才算隔离成功：

```text
导入自: /mnt/fi/fa-venv/lib/python3.11/site-packages/flashinfer
是 wheel 安装: True
带产物: True
NPU 可用: True
```

## 5. 接口测试：不依赖编译器调用

```bash
cd /tmp
FLASHINFER_NPU_JIT_DISABLE=1 /mnt/fi/fa-venv/bin/python -B -c "
import torch, torch_npu, flashinfer as fi, math
S,T,H,D = 256,256,4,128
q=torch.randn(S,H,D,dtype=torch.float16,device='npu')
k=torch.randn(T,H,D,dtype=torch.float16,device='npu')
v=torch.randn(T,H,D,dtype=torch.float16,device='npu')
out = fi.single_prefill_with_kv_cache_catlass(q,k,v)
qf,kf,vf = q.float(),k.float(),v.float()
ref = torch.softmax(torch.einsum('qhd,khd->hqk',qf,kf)/math.sqrt(D), -1)
ref = torch.einsum('hqk,khd->qhd', ref, vf)
print('out', tuple(out.shape), out.dtype)
print('最大绝对误差 %.6f' % (out.float()-ref).abs().max().item())"
```

实测输出：

```text
out (256, 4, 128) torch.float16
最大绝对误差 0.000249
```

`FLASHINFER_NPU_JIT_DISABLE=1` 把编译策略置为禁止。能跑通，
说明**调用确实由随包产物服务，而不是现场编译**。

> **不要用 unset `ASCEND_HOME_PATH` 的方式"屏蔽编译器"**。
> CANN runtime 本身依赖这个变量，去掉会得到
> `aclnnFiFlashAttentionGetWorkspaceSize returned 561002` 这类运行时错误，
> 与是否命中 AOT 无关。禁编译用上面的策略开关。

## 6. 反证：证明确实命中了产物

第 5 步成功，也可能是走了别的路径。把产物移走再调一次：

```bash
P=/mnt/fi/fa-venv/lib/python3.11/site-packages/flashinfer/data/op_packages
mv $P ${P}.bak
cd /tmp && FLASHINFER_NPU_JIT_DISABLE=1 /mnt/fi/fa-venv/bin/python -B -c "
import torch, torch_npu, flashinfer as fi
q=torch.randn(256,4,128,dtype=torch.float16,device='npu'); k=q.clone(); v=q.clone()
try: fi.single_prefill_with_kv_cache_catlass(q,k,v); print('!! 不该成功')
except Exception as e: print('拒绝:', type(e).__name__)"
mv ${P}.bak $P
```

实测：

```text
拒绝: AotMissingError
no matching AOT op-package for kernel_id=ascendc/t4/flash_attention/catlass/op_package/Ascend950, target_soc=Ascend950, ...
```

**这一条通过，第 5 步才算数。**

## 7. envelope 与越界行为

FA 当前只承诺下表范围，超出的在 **dispatch 阶段**拒绝，不会算错：

| 维度 | 支持 |
| --- | --- |
| SoC | Ascend950 |
| dtype | FP16（Q/K/V/out） |
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
cd /tmp && FLASHINFER_NPU_JIT_DISABLE=1 /mnt/fi/fa-venv/bin/python -B -c "
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
$P -B -m pytest -p no:cacheprovider -q                            # 期望 1975 passed
FLASHINFER_CATLASS_ROOT=/mnt/fi/catlass-ref \
  $P -B -m pytest -p no:cacheprovider -q -m real_compile          # 期望 80 passed, 0 skipped
```

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

## 10. 换成另一个算子

把 §2 的脚本换成 `tools/build_matmul_bias_silu.py`（native 档，无需 catlass），
公共入口是 `flashinfer.matmul_bias_silu(a, b, bias)`，BF16，M/N/K 为 16~256 的 16 倍数。
其余步骤完全相同。

接入一个全新算子需要哪些交付件，见 `adding-a-custom-operator.md` 与
`reports/custom-operator-onboarding.md`（33 项清单）。
