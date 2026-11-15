# GEMM、量化融合与 MoE

面向集成者和算子开发者：本页说明相近矩阵接口的区别、Matmul 三后端选型，以及 grouped/量化融合算子的实际约定。

## 普通与分段 GEMM

| 接口 | 当前实现 | 限制 |
| --- | --- | --- |
| `mm_bf16` / `bmm_bf16` | CANN | 名称保留上游，声明含 BF16/FP16/FP32；按真实输入与 SoC 校验 |
| `bmm_fp8` | CANN grouped matmul | Ascend950、E4M3/E5M2 输入与相应 scale/output 约定 |
| `gemm_fp8_nt_groupwise` | CANN | Ascend950、NT 与分组 scale 布局；不提供所有上游低精度格式 |
| `SegmentGEMMWrapper` | CANN | run-only：构造后 run，不增加上游没有的 plan 步骤 |
| `matmul` | CANN / Ascend C / Triton | 相同公共 BF16/ND 契约，当前三者均为 Ascend950 的限定尺寸域 |
| `matmul_bias_silu` | Ascend C | `silu(A @ B + bias)`，BF16、ND、M/N/K 为 16..256 的 16 倍数 |

## Matmul 的三种实现

```text
flashinfer.gemm.matmul / flashinfer.matmul
  → Torch functional 或 out
  → gemm._matmul_impl：元数据/输出约定/签名
  → dispatch：按输入、硬件、显式覆盖与能力选择
  → 公共层准备或复用 out
  → run_selected
      cann runner       → 已安装 torch.mm 的 NPU 路径
      ascendc runner    → 本仓已验证 op-package → ACLNN
      triton runner     → 本目录 Triton kernel / JIT / launch
```

三个实现共享 `op="gemm.matmul"`；不同 kernel_id 区分具体实现。当前共同输入是连续 BF16 ND，M/N/K 为 16..256 的 16 倍数。指定 backend 先筛选候选，指定 ID 锁定兼容实现；无调优 winner 时按现有档位选择，不按登记顺序，也不预设 Ascend C 或 Triton 更快。

Ascend C implementation 声明 NativeBinding/BuildConfig，设备源码在 `csrc/kernels/gemm/matmul/op_kernel/arch35/`；Triton implementation 引用本目录 `kernels.py`，runner 负责其编译身份和启动。二者都不是公共接口本体，CANN 则适配已有厂商执行路径。

未提供 out 时公共层在选型后分配，提供 out 时保留对象身份并检查 alias。Ascend C Matmul 是 AOT-only；缺包不会把同一次调用改成另一后端重试。

## GroupedMatmulSwigluQuant

`grouped_matmul_swiglu_quant` 使用 Ascend910B arch22 AOT：grouped integer GEMM → 反量化 → SwiGLU → 逐 token INT8 量化，返回 `[M,N/2]` INT8 输出与 `[M]` FP32 scale。

A8W8 与 A8W4 按完整组合声明，不是对任意存储 dtype 的笛卡尔积支持。A8W4 的 INT32 carrier 表示打包 INT4，某些 scale 由 INT64 carrier 保存 UINT64 位模式；这属于位解释，不是普通数值 cast。ND/NZ 排布、scale 的 per-channel/per-group 与 K/N 对齐由具体 case/binding 联合限制。

`group_list` 是非负、非递减并以 M 结束的专家前缀，可含空专家。A8W8 的 bias/offset 不作为有效计算语义，显式拒绝；A8W4 的 bias 参数承载辅助矩阵。权重打包必须与 native 的 INT4/NZ 字节顺序一致，不能以 Torch shape/dtype/format 元数据相同代替。

公共层保留量化/裁剪语义与必要值检查，后端 runner 适配真实属性与存储。该实现 AOT-only、NOT_CAPTURABLE；分组 GEMM 不等于提供专家并行或分布式路由。

## MoE

`fused_moe`、`fp8_per_tensor_scale_moe`、`fp8_block_scale_moe` 使用已有 CANN 原语组合与受限外部接入。它们不提供上游全部 backend、激活、量化方式和 moe_ep API。

一个 CANN MoE 声明使用 `Ascend910_93`，硬件探测归一化为 `Ascend910C`，精确能力匹配因此不能据该声明保证 A3 可路由。现有支持与名称差异按 [矩阵](../backends/overview.md) 判断；统一目标命名是 [规划](../roadmap.md) 中的扩展前置条件。

源码：[公共 GEMM](../../flashinfer/gemm.py)、[Matmul 公共定义](../../flashinfer/operators/gemm/matmul/)、[Ascend C](../../flashinfer/ascendc/gemm/)、[CANN](../../flashinfer/cann/gemm/)、[Triton Matmul](../../flashinfer/triton/gemm/matmul/)、[GMM 公共定义](../../flashinfer/operators/gemm/grouped_matmul_swiglu_quant/)、[MoE 适配](../../flashinfer/cann/fused_moe/)。
