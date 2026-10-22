# P0 / P1 执行交付记录

执行范围为用户授权的前两个阶段。依据 [`docs/TASKS.md`](../../../docs/TASKS.md)，
按 T1 → T3 → T2 → T4 → T5 推进。实现与自动验收已留痕，用户尚未逐项复核。

## 当前结果

| 验证 | 当前结果 | 证据 |
| --- | --- | --- |
| 包模块导入 | 119 个，0 失败 | [imports](steps/P1-final-imports.log) |
| 测试收集 | 1000 项，0 错误 | [collect](steps/P1-final-collect.log) |
| 全量测试 | 997 通过、3 失败、0 跳过 | [完整输出](steps/P1-full-after-workload.log)、[JUnit](P1-after-workload.xml)、[失败清单](P1-remaining-failures.json) |
| parity / enum / pdl / backend 门禁 | 131 通过 | [门禁输出](steps/T5-and-T4-gates.log) |
| 原有 A5 smoke | 23 项数值检查通过 | [smoke](steps/P1-final-A5-smoke.log) |
| 双账本完整性 | 退出码 0；覆盖状态仍 incomplete | [安装包验证](steps/T5-installed-final.log) |
| 主登记表 | 63 条，35 个 op；CANN 32、reference 30、Triton 1 | [登记表](steps/P1-registry-inventory.log) |

全量结果与最初 T1 的 734 通过 / 119 失败相比已显著收敛。
测试数量增加是新增功能验证的结果，不将新增用例与旧失败机械抵消。

## 已落地的任务

- **T1**：恢复实际使用的 runtime、artifact、POD、launch/binary/error ABI 和 JIT/CLI 导出，解除导入与收集阻塞；补齐量化分组校验及版本字段。历史冻结 ABI 身份尚未完全还原，任务保持未全量完成。
- **T3**：固定源码核实 59 条 NVIDIA 专属定义，逐条登记有意差异和原因，不再混入昇腾待实现缺口。
- **T2.1**：10 个 sampling 接口，签名按固定提交核验；边界同值 token、MinP、联合过滤、speculative 计数和本地随机复现均有对拍。跨 CUDA/NPU 的 Philox seed/offset 精确序列未实现，显式报错并登记差异。
- **T2.2**：单请求 prefill/decode、paged/ragged prefill、paged decode 走真实 A5 FIA；公开签名、GQA、NHD/HND、mask、out/LSE 和分页寻址有验证。固定上游 LSE 默认为 log2；A5 BNSD 的部分 LSE 异常通过使用已验证的 BSND 路径规避。混合 BatchAttention 仍只有 Host/provider 路径。
- **T2.5**：SiLU、GELU、tanh-GELU 三个融合激活入口；大宽度拆分及半精度/单精度有原生算子对拍。
- **T2.6 部分**：`merge_states` 支持原生 TensorList 合并及超过 15 段的合并树；`append_paged_kv_cache` 使用真实 `npu_scatter_pa_kv_cache(cache_mode="Norm")`，覆盖分离/packed、NHD/HND、尾页、无重复写入与原地缓存语义。
- **T4.1～T4.6**：路由枚举、单请求 decode 形状、错误 MoE 别名、decode plan/reset、backend 诊断、PDL 硬件降级全部完成定向与整体门禁。删除的三个前缀别名在账本指向保留的 NPU 原接口，未声称 drop-in。
- **T5 部分**：双账本固定到完整提交；库存逐条带来源、CANN 状态、本仓状态和下游使用字段；`parity-check`、package data 与 CI 配置可运行。原始覆盖表未提供，保持 bootstrap。

新增数值路径的验收基于真实 Ascend950PR A5，尚未宣称覆盖所有 SoC、shape 或性能优化。
批量 Attention 当前为 eager 的逐请求 KV 整理与 FIA 调用；native graph、最大 workspace、
torch.compile/meta 等仍归后续 T8～T11，入口对未支持功能明确报错。
旧 `merge_state` 的自然对数 LSE 与上游差异已登记；新 `merge_states` 遵守 log2，勿混用。

## 剩余项与所需资料

1. **T1**：3 个冻结测试失败，涉及 binary ABI v1/v2 和 metadata wire。POD 字节布局、pack/unpack、参数顺序/mutation、artifact 校验已通过；binary/error ABI 的历史序列化身份仍不匹配。需要原始 schema 实现或对应提交，未通过改断言、硬编码指纹或禁用测试掩盖。
2. **T2.3 / T2.4 / T2.6 余项**：任务正文未列出五个 RoPE、四个 GEMM、sparse/fp4 的确切接口。需要原始 147 条覆盖表确认，不能从上游更多候选里随意挑满数量。
3. **T5**：257 条是真实上游声明库存，不等于已导入原始 147 行三列覆盖矩阵。CANN 未核实映射与 vLLM/SGLang 使用状态保留 unverified；需要原表和 vllm-ascend 本地检出路径。
4. **后续阶段**：P2 及以后尚未启动，按已约定范围先交付 P0/P1 当前结果，再依据证据继续；未把未知项标成完成。

冻结 wire 的复查恢复了 `WorkloadSpec.schema_version=1`。
[诊断证据](T1-workload-version-recovery.json) 表明，补齐该字段后，诊断副本使用历史 binary header 即能复原原始 wire 哈希。
该替换只用于定位，生产序列化和原冻结断言保持真实计算结果；剩余失败仍完整保留。

## 复现

```bash
source /usr/local/Ascend/cann/set_env.sh
source /root/miniforge3/etc/profile.d/conda.sh
conda activate /root/miniforge3/envs/flashinfer-a5
python tools/check_imports.py
python -m pytest --collect-only -q
python -m pytest tests/ -q --tb=short
python -m flashinfer parity-check
python examples/a5_smoke.py
```

环境：Python 3.11、torch 2.10.0+cpu、torch_npu 2.10.0.post7、triton-ascend 3.6.0，CANN 9.2.0-beta.2。
上游均使用 `/home/dy/projects/flashinfer-ori` 内的固定 Git 对象
`2895577720741ddfedff118202c89584996b8875`，未修改其 checkout。

## 修改留痕与提交

[`events.jsonl`](events.jsonl) 记录操作时间、命令和退出码；`steps/*.patch` / `*.log` 保存每组源码差异和完整输出。
任务提交使用中文任务号标题；后续全量测试发现的同任务遗漏以补充提交记录，未改写既有提交历史。

前序 A5 修改保留在工作区，基线见 [`preexisting.patch`](preexisting.patch)。
本次任务提交没有把它们混入其他任务；当前数值证据基于包含这些 A5 改动的工作区，
不应被描述成只检出某一个新提交即可复现的干净树证据。

主要任务提交：

| 任务 | 提交 |
| --- | --- |
| T1 初次恢复 / 补充 | `dfdfa4a` / `8d7037a` / `0581b19` |
| T3 | `a39992d` |
| T2.1 | `86ad8d6` |
| T2.2 / 根导出补充 | `c39f565` / `6ccc5c1` |
| T2.5 | `3108f64` |
| T2.6 cascade/page | `a9760d9` |
| T4.1 / T4.2 / T4.3 | `af4f723` / `643a712` / `b5cd625` |
| T4.4 / 旧 smoke 更新 | `4f6b2f4` / `ff0c66d` |
| T4.5 / T4.6 | `7dd4b5e` / `23440fa` |

T5 提交可用 `git log --oneline --grep='^T5'` 查询；其安装包验证、任务进度和本报告一起提交。
