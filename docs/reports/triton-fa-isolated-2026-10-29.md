# Triton FA 隔离接入与验证

日期：2026-10-29。基线：`infer`，`d0d3f16a3c8b1eaef7be88be0bfab0f18772e6da`。

按用户要求，不修改生产精确集合断言、不扩展默认生产登记。默认仍为 81 条 descriptor，隔离 registry 为 82 条。代码尚未提交。

已实现同一 FA 接口接入 CATLASS AOT 与 Triton JIT，并验证两个真实 runner 的按 shape 路由。**真实测量尚未形成显著冠军，未达到“凭实测记录使两个 shape 自动选择不同实现”的验收条件。** 路由 UT 使用明确标注的合成计时，只验证机制，不代表最优实现或性能结论。

## 环境与设备

- 自检实际输出：`True 8`。
- 使用 `dy_test` 环境，Python 3.12、Torch `2.10.0+cpu`、TorchNPU `2.10.0.post7`、Triton `3.6.0`。
- 加载迁移机器提供的 CANN `set_env.sh`，实际 CANN 为 9.2.0；不使用 AGENTS.md 中原服务器的解释器路径。
- 硬件签名：`Ascend950`、56 AI Core、UB 221184 bytes。功能测试使用默认设备；额外调优检查使用设备 3。
- CATLASS AOT 与 C++ ACLNN 桥复用已有匹配产物，经过现有验证加载器实际运行。本次未重新编译 Ascend C 设备源码。

## 改动与调用路径

| 位置 | 改动 |
| --- | --- |
| `flashinfer/triton/_attention_kernels.py:8` | NHD dense FA，在线 softmax，FP16/BF16 输入和 FP32 累加 |
| `flashinfer/triton/attention.py:24` | 惰性 runner；dtype 对应完整 Triton spec，复用统一构建、验证、发布和加载 |
| `flashinfer/triton/attention.py:129` | `flash_attention_registry()` 复制默认绑定并追加实验 descriptor，不修改全局生产集合 |
| `flashinfer/runtime/registry.py:325` | `kernel_registry_scope()` 用 ContextVar 约束 registry 的执行作用域，异常退出和嵌套时恢复 |
| `flashinfer/attention/catlass_prefill.py:65` | 提取运行和调优共享的 `prefill_signature()`，保持原校验、分配顺序及公共 Torch 接口 |
| `flashinfer/jit/_triton_abi.py:14` | 新增显式混合二进制 ABI：必须同时校验确定的 AIC/AIV 两个入口；原单入口规则保留 |
| `flashinfer/tuning/benchmark.py`、`store.py`、`identity.py` | 精确 shape 记录与可选执行指纹校验；绑定 spec、软件版本及 runner/桥身份，拒绝过期记录 |
| `tools/tune_flash_attention.py:26` | 独立 registry 的正确性检查、实测采集、保存记录和公共接口自动选择验证 |

默认调用链不变。隔离作用域内的链路为：公共 Python 函数 → 同一个 Torch schema → 共享 signature → scoped registry / dispatch → CATLASS AOT runner 或 Triton JIT runner → 对应设备实现。

实验 Triton 与 CATLASS 共用现有受限契约：Q `[S,H,128]`，K/V `[T,H,128]`，S/T 为 128..512 的 128 倍数，H=1..8，统一 FP16 或 BF16，连续 NHD/ND，非 causal、无 mask。未扩张接口语义、冻结 ABI 或 Ascend C spec。Triton 经自身 launcher 使用当前 NPU stream，下发前后同步；不经 ACLNN，仍为同步 eager、`NOT_CAPTURABLE`。

## 命令、退出码与结果

完整输出位于本次证据目录 `triton-fa-20261029-icjq1wjc`。以下 `$E` 表示该目录；各回归目录内的 `command.json` 保存实际展开后的命令，`pytest.log`、`results.xml`、`pytest.exit`、`run.exit`、`summary.json` 保存原始结果。命令使用已激活的 `dy_test` 解释器。

```bash
python3 -B tools/run_regression.py --profile quick --log-dir "$E/quick"
python3 -B tools/run_regression.py --profile compile --log-dir "$E/compile-targeted" tests/test_triton_attention_real.py
python3 -B tools/run_regression.py --profile compile --log-dir "$E/compile-cold-fixed" tests/test_triton_attention_real.py::test_cold_jit_and_compiler_free_disk_hit
python3 -B tools/run_regression.py --profile compile --log-dir "$E/compile-routing" tests/test_triton_attention_real.py::test_public_api_uses_two_exact_shape_records_with_real_runners
```

| 执行 | 实际结果 | 原始退出码 |
| --- | --- | --- |
| 定向 host 首轮 | 270 passed、1 failed、21 deselected；新增 UT 误取 reference 候选，已修正 fixture | 1 |
| 定向 host 修正后 | 271 passed、0 failed、0 skipped、21 deselected | 0 |
| quick，全部非 real_compile | **1982 passed、0 failed、0 errors、0 skipped、98 deselected** | **0** |
| 初次 FA real_compile 专项 | 4 passed、1 failed、0 skipped；冷构建及缓存运行均成功，最后查找日志的 glob 路径错误 | 1 |
| 修正日志路径后，只重跑冷构建/缓存用例 | 1 passed、0 failed、0 skipped | 0 |
| 新增真实 runner 自动路由专项 | 1 passed、0 failed、0 skipped | 0 |

real_compile 共验证 **6 个不同用例**，按上述分次执行全部通过；不是一次全量编译矩阵的结果。覆盖两种 dtype、六组代表性 `(S,T,H)`：`(128,128,1)`、`(128,512,8)`、`(256,384,3)`、`(384,256,2)`、`(512,128,1)`、`(512,512,8)`，两条实现的 functional/out、CPU FP64 数值参考、非默认 stream、storage offset、别名/混合 dtype/causal 拒绝，以及 TorchNPU 队列 0/1/2 的生产者→FA→消费者顺序。

冷 JIT 为 FP16/BF16 分别生成两个不同 spec 的产物；新进程命中磁盘缓存。额外执行下列命令，缓存模式禁止 `subprocess.Popen` 和 Triton compiler backend，验证加载/执行仍可完成；同时运行原单入口 Gemma 示例，检查本次混合 ABI 扩展不破坏该路径。

```bash
python3 -B -m tests.triton_attention_acceptance cold cached
python3 -B -m examples.two_level_gemma --cache-dir "$E/single-abi"
```

实际结果：禁止子进程的缓存加载为 `PASS`、4 次数值检查、退出码 0；原 Gemma 示例为 2 条不同选择、2 个不同真实 ELF 产物及 overlap 数值检查通过，退出码 0。静态检查确认 206 个既有设备源码、声明及测试文件摘要不变，所有既有文档代码块完整保留，diff 空白检查通过。

自动路由 UT 在临时数据库内放入两条**合成计时记录**，随后不传 backend/kernel_id，通过公共 FA 接口实际运行：长度 128 选择 Triton，长度 512 选择 CATLASS，并比对真实 NPU 输出。该数据库属于测试 fixture，不得发布为调优数据。

补充强制执行验证：同一公共接口分别传入 `backend="triton"` 和精确 Triton `kernel_id`，用 runner 实际 launch 次数确认两者均进入 Triton，输出形状为 `(128, 1, 128)`，两次输出完全一致。设备自检为 `True 8`，执行日志 `force-triton.log` 显示两项 `PASS`，原始退出码文件 `force-triton.exit` 为 `0`。本次文档补充只做示例静态检查，不重复冷构建或全量回归。

## 真实调优结果

```bash
python3 -B tools/tune_flash_attention.py --output-dir "$E/tuning-first" --dtype float16 bfloat16 --require-both-backends
python3 -B tools/tune_flash_attention.py --output-dir "$E/tuning-second" --dtype float16 bfloat16 --iterations 500 --warmup 5 --require-both-backends
python3 -B tools/tune_flash_attention.py --output-dir "$E/tuning-isolated" --dtype float16 bfloat16 --iterations 500 --warmup 5 --device 3 --affinity 128,129,130,131 --timing-boundary runner --require-both-backends
```

三组均对 `(128,128,1)`、`(512,512,8)` 的 FP16/BF16 检查数值，通过统一边界测量并保存原始样本。第一组使用当时工具默认的 50 warmup / 20 iterations；后两组显式调整重复次数。前两组计时公共 out API，第三组计时已选定 runner，均包含同步 eager 下发；这些计时边界不同，不合并比较数值。

**共 12 条记录均为 `no_significant_difference`、`winner=null`，实际自动选择均为 CATLASS。** CLI 因 `--require-both-backends` 条件未满足而退出 **1**，不是数值验证失败。全部记录、比较门槛和数据库已保留；没有删掉离群值、降低噪声/显著性门槛或人工填写冠军。采集到较高 CPU 利用及测量波动，但不能据此唯一归因。

初次 Triton 编译曾在单入口校验处失败；诊断产物实际导出 `_dense_prefill_kernel_mix_aic` 与 `_dense_prefill_kernel_mix_aiv`。问题由显式声明并严格验证混合产物解决，未取消符号校验，也未修改原单入口准入条件。

## 未验证与边界

- 未执行 full 或全部 98 个 real_compile 用例；quick 不能称为全量回归。
- 未获得真实显著冠军，未完成基于实测的跨 shape 双后端自动选择验收；需要可比、稳定的测量条件及必要的 kernel 优化。
- 未修改默认生产登记及其精确集合断言，未发布隔离调优数据。
- 未重新打包或安装新 wheel；本次不作为装后新 Triton 路线验证。
- 未验证图 capture/replay、其他 SoC、软件版本组合、完整 shape 笛卡尔积、causal/mask/GQA/paged/ragged 扩展。
- 不给出性能优势结论。已有文档“尚无 Triton FA 实现”的表述已更新为“已有隔离实现，尚未进入默认生产登记”。
