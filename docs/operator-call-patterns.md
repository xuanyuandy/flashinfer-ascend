# 算子调用方式与公共接口 UT

本轮对照 flashinfer-ori 提交 `62b5ae4e52e14d841e1b0455dc3cfa7053181741` 的源码及 `tests/`。
结论是：**并非所有算子都先创建 wrapper，再调用 plan 和 run。**
这里的 wrapper 特指需要用户实例化、持有状态的类；直接函数内部仍有参数检查、后端选择、模块缓存和 kernel 下发。
两者均可使用 AOT/JIT，是否公开 `plan()` 与编译方式没有一一对应关系。

## 按上游实际生命周期对标

| 调用方式 | 上游典型入口与测试证据 | Ascend 对标与 UT 要求 |
| --- | --- | --- |
| 直接函数 | `single_prefill_with_kv_cache`、`single_decode_with_kv_cache`；flashinfer-ori: `tests/attention/test_single_prefill.py`、`flashinfer/decode.py` | 用户直接传 Q/K/V。公共数值 UT 调用门面函数，不要求人为增加 wrapper/plan；仍检查支持域、输出和 LSE |
| 直接函数 | norm、RoPE、sampling、activation、普通 GEMM、部分 fused MoE；flashinfer-ori: `tests/norm/`、`tests/attention/test_rope.py`、`tests/gemm/`、`tests/moe/` | 直接通过对应公共函数验证；原地、`out`、量化变体分别保持各自契约，不因为共享底层实现就合并接口 |
| 直接 batch 函数 | `cudnn_batch_decode_with_kv_cache`、`trtllm_batch_context_with_kv_cache`；flashinfer-ori: `flashinfer/decode.py`、`flashinfer/prefill.py` | 函数消费传入的 batch/cache 元数据；是否 batch 并不能决定是否需要用户创建 wrapper |
| 创建 → plan → run | `BatchDecodeWithPagedKVCacheWrapper`、paged/ragged prefill wrapper、`BatchAttention`；flashinfer-ori: `tests/attention/test_batch_decode_kernels.py`、`test_batch_prefill_kernels.py`、`test_batch_attention.py` | 持有 workspace 和 batch/KV 元数据。UT 覆盖未 plan 拒绝、一次 plan 后多次 run、更新输入、重新 plan 及 out/LSE；不得只调底层 runner 代替公共路径 |
| 创建 → plan → run | block/variable-block sparse、MLA、cascade、POD；flashinfer-ori: `flashinfer/sparse.py`、`flashinfer/mla/_batch_mla/_wrapper.py`、`flashinfer/cascade.py`、`flashinfer/pod.py` | 按对应的索引、稀疏布局、压缩 KV 或共享前缀元数据设计，不能把普通 batch attention 的通过当作这些算子已支持 |
| 创建 → run | `SegmentGEMMWrapper`；flashinfer-ori: `flashinfer/gemm/gemm_base.py`、`tests/gemm/test_group_gemm.py` | 上游没有公开 plan。每次 run 提供 segment 信息；UT 验证重复使用 wrapper、不同输入/分段、权重索引和 out |
| 创建 → begin_forward → forward | shared-prefix paged decode/prefill wrapper；flashinfer-ori: `flashinfer/cascade.py` | 保留独立的旧式生命周期分类；不能按类名推断一定有 plan/run。本仓没有对应完整公共实现 |
| 准备函数 → 执行函数/对象 | `fmha_varlen_plan`/`fmha_varlen`、`prepare_sm110_gqa_decode`/`launch_sm110_gqa_decode_prepared` 等；flashinfer-ori: `flashinfer/prefill.py`、`flashinfer/decode.py` | 准备可以由函数或返回的对象承载；逐接口记录其实际契约，不统一包装成一个虚构的类 |

状态复用通常适合重复执行同一批结构：页表、indptr、稀疏索引、调度元数据和 workspace 可由 plan 准备，run 消费当前 Tensor。请求结构发生变化时应重新 plan；只更换同一支持域内的 Tensor 内容时可以复用 plan。
单请求、无需要公开保留的调度状态的运算通常直接调用。上游也有直接 batch 函数和 run-only wrapper，所以“single/batch”只能辅助理解，最终以接口源码为准。

API 的 plan、编译特化和 CANN host tiling 各有职责。`plan()` 准备的是调用方可复用的执行状态；JIT 特化决定二进制身份；ACLNN `GetWorkspaceSize` 执行 host tiling 并创建 executor。接口没有显式 plan 不表示没有后两者。

## 本仓公共 UT 的整改

本轮没有将底层契约测试改成公共接口测试：ABI、源码摘要、产物损坏拒绝、注册精确集合和缓存策略仍需要自己的单元测试。
**公共数值测试必须经过真实公共入口，生命周期按上游对应入口执行。**

| 本仓接口 | 本轮公共路径证据 | 验证边界 |
| --- | --- | --- |
| single prefill/decode | [公共 attention UT](../tests/test_attention_cann.py) 调用 `flashinfer.single_*` | 保留 FP16/BF16、GQA、layout、mask、输出及 LSE 断言 |
| paged/ragged prefill | 同一文件：创建 wrapper → 未 plan 拒绝 → plan → run → 使用新 Q 再 run → 重新 plan → run | 比较独立参考结果，保留原有容差和 out/LSE 对象身份检查；验证 plan 复用及 generation 更新 |
| paged decode | 同一文件：创建 wrapper → 未 plan 拒绝 → plan → workspace reset → run → 新 Q 再 run | 保留 q_len、layout、dtype、LSE 和错误 q_len 拒绝检查 |
| `SegmentGEMMWrapper` | [GEMM 公共 UT](../tests/test_gemm_segment.py)：创建 → run → 新输入写入 out | 没有添加上游不存在的 plan；保留不同分段、列主序和权重索引测试 |
| mixed `BatchAttention` | [mixed 公共 UT](../tests/test_batch_attention_frontend.py)：`flashinfer.BatchAttention(device="cpu")` → plan → run → 新 V 再 run | **仅 Host oracle/框架生命周期**。本仓虽导出该类，默认 NPU provider 尚未形成生产执行闭环，不能报告为 NPU 算子通过 |
| 其余直接函数 | 既有 norm/RoPE/sampling/activation/MoE/cascade/page 数值 UT | 继续通过公共函数执行；不因本轮生命周期整改新增不适用的 plan/run |

下面是调用顺序示意，Tensor 与元数据构造、完整断言见上述可运行 UT：

```python
# 直接接口
output = flashinfer.single_prefill_with_kv_cache(q, k, v)

# 有计划状态的批量接口
wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace)
wrapper.plan(qo_indptr, kv_indptr, kv_indices, last_page_len,
             num_qo_heads, num_kv_heads, head_dim, page_size)
output = wrapper.run(q, (k_cache, v_cache))
next_output = wrapper.run(next_q, (k_cache, v_cache))

# run-only wrapper
gemm = flashinfer.SegmentGEMMWrapper(workspace)
output = gemm.run(x, weights, batch_size, False, seg_indptr=seg_indptr)
```

## 避免把接口数当作 kernel 数

同一函数可能同时从顶层和子模块导出：例如 `flashinfer.rmsnorm` 与 `flashinfer.norm.rmsnorm` 指向同一声明。
`plan`、`run`、`forward`、workspace 查询是同一对象的不同阶段；O-only 与 O+LSE、原地与 out 形式也可能共享底层 kernel，但公开参数和输出契约仍需分别记录。
相反，一个公共 attention API 也可能因 backend、dtype、head dimension 或执行模式选择多份编译实现。

清单分别记录公开拼写、规范声明位置、生命周期、逻辑能力和 backend。按声明去重可避免重复计数，但不能仅因名字相似就宣称两个接口完全等价。
测试源码中出现一个调用，也不能替代该调用已在相应设备上通过的证据。

## 开发回归方式

```bash
# 日常：所有非 real_compile 用例，包含普通 NPU 数值 UT
python3 -B tools/run_regression.py --profile quick

# 只改特定接口时，可先定位其公共 UT
python3 -B tools/run_regression.py --profile quick tests/test_attention_cann.py tests/test_gemm_segment.py

# 编译相关专项；不要在同一次 full 后机械重复执行
python3 -B tools/run_regression.py --profile compile

# 发布或跨层改动：只执行一次完整矩阵，已包含 real_compile
python3 -B tools/run_regression.py --profile full
```

quick 的 `real_compile` 是 deselected，不能写成全量已通过，也不能推断冷 JIT/AOT 已验证。
脚本会记录实际命令、完整日志、JUnit、pytest 原始退出码及最终状态；skip 或没有执行任何用例均不视为成功。
上次完整矩阵 2003 项中，90 项 `real_compile` 消耗了约 900.8 秒 testcase 时间，其余 1913 项约 30.3 秒；这是该次日志的耗时分布，新的 quick 墙钟耗时以实际执行为准。
