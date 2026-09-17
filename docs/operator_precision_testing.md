# 算子精度测试指导

本文以 `norm.rmsnorm` / `norm.fused_add_rmsnorm` 为样板，说明 FlashInfer
算子精度测试的环境准备、运行方式、验收口径与新增算子的接入步骤。

## 1. 测试分层

[tests/test_norm.py](../tests/test_norm.py) 按"先链路、后数值"分四层，新增算子沿用同一结构：

| 层 | 类 | 依赖 | 说明 |
|---|---|---|---|
| 注册 | `TestNormRegistration` | 无 | kernel 已进全局 registry，capability 声明与实测支持集一致 |
| 调度 | `TestNormDispatch` | 无 | auto/backend/kernel_id 策略、reference opt-in、不支持输入显式报错、决策缓存 |
| 门面流 | `TestNormFacadeFlow` | 无 | 假张量（仅 `dtype`/`shape` 协议）+ custom probe kernel 走通 门面→签名→dispatch→runner |
| 数值 | `TestNormNumerics` | torch / torch+NPU | 见 §3 验收口径；host 无 torch 时自动 skip |

前三层是 host 单测（§14.1）：不 import torch，在开发机即可运行。
数值层通过 `pytest.importorskip("torch")` / `importorskip("torch_npu")` +
`torch.npu.is_available()` 按环境自动跳过。

## 2. 环境准备

### 2.1 host（开发机）

```bash
cd /home/jiangbo123/code/flashinfer
python3 -m pytest tests/test_norm.py -v
```

预期：链路层全过，数值层全部 `skipped`（无 torch）。

### 2.2 容器（NPU 数值验收）

容器按 `~/readme.md` 进入（`docker exec -it jiangbo123 bash`），非交互方式直接执行：

```bash
docker exec jiangbo123 bash -c "source /usr/local/Ascend/cann/set_env.sh; \
  cd /home/jiangbo123/code/flashinfer && \
  PYTHONPATH=/home/jiangbo123/code/flashinfer \
  /home/jiangbo123/miniconda3/envs/py310/bin/python3 -m pytest tests/test_norm.py -v"
```

要点：

- Python 环境：conda env `py310`（torch 2.7.1+cpu + torch_npu 2.7.1.post5）。
  缺 pytest 时先 `python3 -m pip install pytest`。
- **必须 source CANN `set_env.sh`**，否则 `import torch` 自动加载 torch_npu
  会因缺 driver 依赖报错。
- 仓库经挂载在容器内同路径可见；未安装包时用 `PYTHONPATH` 指向仓库根。
- 当前实测环境：CANN 9.0 × Ascend910B。

## 3. 数值验收口径

每个算子须通过三类数值断言（`TestNormNumerics`）：

1. **CPU reference 正确性**：`backend="reference"` 显式 opt-in，对照手算
   fp64 公式，验证参考实现本身语义正确。
2. **NPU vs fp64**：`auto`（NPU 上落到 CANN aclnn）结果对照 fp64 参考
   （输入拷贝到 CPU 后计算），三种 dtype 全测。
3. **CANN vs reference 一致性**：同输入分别跑 `backend="cann"`（NPU）与
   `backend="reference"`（CPU 副本），断言结果一致——M1 起跨 backend
   正确性的验收口径。

### 3.1 参考实现约定

- reference kernel 内部以 **float64 累计**，输出前一次性舍入回输入 dtype
  （见 `reference_rmsnorm` / `reference_fused_add_rmsnorm`）。
- fp64 参考公式在 **CPU 上**计算：`x.cpu()` 后再 `.double()`。
  **禁止在 NPU 张量上直接 `.double()`**——torch_npu 的 fp64 cast 会静默
  退化为 fp32，比较时报 `RuntimeError: Float did not match Double`。

### 3.2 容差标准

| dtype | atol（vs fp64 / 跨 backend） | 依据 |
|---|---|---|
| float32 | 1e-4 | 设备侧 fp32 累计，误差远在此界内 |
| float16 | 2e-2 | fp16 舍入量级（10 bit 尾数） |
| bfloat16 | 8e-2 | bf16 舍入量级（8 bit 尾数），最宽松 |

容差随算子数值特性调整，但必须在测试中显式写出，不允许无容差的
`allclose` 默认阈值掩盖 dtype 差异。

## 4. 新增算子的精度测试接入步骤

1. **确认上游语义**：函数名、参数名、默认值、计算定义与返回值结构，
   记录平台差异（如 `enable_pdl` 类 NVIDIA 专属参数）到 parity notes。
2. **写 reference 实现**（float64 累计）并注册 `reference/<op>_v1`
   descriptor——它是其余 backend 的对照基准，必须先可信。
3. **接入目标 backend**（CANN aclnn 等），capability 只声明**已实测**的
   dtype / SoC 集合。
4. **复制测试骨架**：按 §1 四层补齐；数值层三 dtype 参数化，容差按 §3.2。
5. **host 跑链路层** → **容器跑全量**：两层都绿才算完成。
6. **更新 `api_parity.json`**：`semantic_status` / `implementation_status`
   与 notes。

## 5. 常见问题

- **`No module named 'torch'`（容器）**：没 source `set_env.sh`，或用了系统
  python 而非 conda env `py310`。
- **`Float did not match Double`**：在 NPU 张量上调了 `.double()`；改为
  先 `.cpu()` 再 `.double()`。
- **auto 调度报 `无可用 kernel`**：host 开发机（unknown SoC）不在 CANN
  声明支持集内属预期；reference backend 默认不进 auto 候选，须显式
  `backend="reference"` 或 `kernel_id=` 锁定。
- **数值偶发超差**：先用固定种子/固定输入复现，再判断是实现错误还是
  容差过紧；不要在未定位原因前直接放宽容差。
