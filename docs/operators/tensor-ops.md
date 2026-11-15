# 其他 Tensor 算子

面向终端用户：本页按公共语义归类已有 Norm、Activation、RoPE、Sampling、KV 和状态合并入口，并说明它们与完整上游族的边界。

| 类别 | 公共入口 | 当前后端与注意点 |
| --- | --- | --- |
| Norm | rmsnorm、fused_add_rmsnorm、gemma_rmsnorm、layernorm | CANN 与显式 reference，输入/权重/残差和 out 约定分别校验 |
| Gemma fused Add RMSNorm | gemma_fused_add_rmsnorm | Triton 与 reference；不是所有 Norm 共用的实现 |
| Activation | silu_and_mul、gelu_and_mul、gelu_tanh_and_mul | CANN 与 reference，最后一维按两个操作数分半；保留上游 enable_pdl 参数，NPU 关闭 PDL |
| RoPE | apply_rope、apply_rope_pos_ids、apply_rope_with_cos_sin_cache 及其 inplace 版本 | CANN 与 reference；位置、频率、cache、旋转格式和原地更新各有契约 |
| Sampling | probabilities/logits 采样、top-k/top-p/min-p、renorm、mask、chain_speculative_sampling | CANN/设备侧组合与 reference；不承诺跨平台随机位级一致 |
| KV cache | append_paged_kv_cache | CANN 与 reference；普通 paged KV 追加，不等于 MLA/FP4 全格式追加 |
| 状态合并 | merge_state、merge_states | CANN 与 reference；合并 output/LSE，不提供完整多级 cascade wrapper |

具体 op、存储 dtype、索引类型、SoC 与测试定义见 [支持矩阵](../backends/overview.md)。dtype 集合包含元数据和辅助 Tensor，不能据集合任意混配。

## 参数与路由

Norm 和部分 RoPE 接口新增 backend/kernel_id 关键词；Activation、Sampling、page、merge_states 保留上游签名，通过读取全局默认选择。不要给不接受这些参数的接口补 kwargs。

PDL 的关闭仅限硬件启动能力，不允许静默丢弃 mask、scale、位置或量化计算语义。upstream backend 值与本仓 backend 值不自动同义，参见 [兼容说明](../upstream-compatibility.md)。

Reference 需显式选择；它用于独立数值对照，不是目标后端运行失败时的隐式重算。部分门面的输出分配时机和前置校验与通用 schema 路径不同，参见 [路由边界](../concepts/backend-routing.md)。

## 数值与状态

inplace/out 接口按其声明保持或更新原对象；残差、LSE、采样控制参数和错误输入都需要专门测试，不能用一个纯 functional 样例替代。随机算子应核对分布、局部可复现性和明确的 seed/generator/offset 行为，不以单次输出相等推导全平台等价。

源码：[norm](../../flashinfer/norm.py)、[activation](../../flashinfer/activation.py)、[rope](../../flashinfer/rope.py)、[sampling](../../flashinfer/sampling.py)、[page](../../flashinfer/page.py)、[cascade](../../flashinfer/cascade.py)。新增覆盖规则见 [测试规范](../development/testing.md)。
