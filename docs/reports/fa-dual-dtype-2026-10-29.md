# FA 双 dtype 与接入框架扩展验证

基线：`infer` 分支 `67488c2`。用户明确要求通过 FA 的 FP16/BF16 接入验证框架的 dtype 扩展能力，并授权将原 BF16 出界反例替换为 FP32 反例。按用户的速度要求，本轮只执行受影响测试和安装验收，没有运行全量回归或完整 real_compile 矩阵。

## 实际环境与结论

- 使用迁移机器的 `dy_test` 解释器及用户指定的 CANN `set_env.sh`；设备自检实际输出 **`True 8`**。
- FA 公共函数、kernel_id、ACLNN Tensor 参数顺序、tiling 数据布局保持一致；一个 op_package 真实编译并携带 FP16、BF16 两个不同 SHA-256 的设备二进制，由 CANN 根据实际 dtype 匹配。
- FA 继续限定连续 ND/NHD、batch=1、D=128、相同 heads 1..8、S/T 为 128..512 的 128 倍数、非 causal、无 mask。Q/K/V/out 必须全部 FP16 或全部 BF16；混用拒绝。
- 登记表仍为 81 descriptor / 45 op，生产 Ascend C 仍为两条。生产图能力仍为 `NOT_CAPTURABLE`。

## 框架改动与接入位置

| 文件位置 | 实际变化 |
| --- | --- |
| `flashinfer/jit/recipe.py:138`、`:255` | v2 增加可选 `dtype_signatures`，按 Tensor ABI 顺序列出完整合法组合；校验长度、类型、唯一性及基础组合，拒绝隐式笛卡尔积 |
| `flashinfer/jit/recipe.py:310` | 省略扩展字段时不改变既有 v1/v2 序列化身份；原 v1 能力拒绝及冻结测试保留 |
| `flashinfer/jit/op_package.py:150` | 验证每个设备 binary 的参数顺序和完整 dtype 组合，并要求包覆盖所有声明组合；缺少任一种 dtype 的包失败 |
| `tools/_production_release.py:10`、`tools/build_flash_attention.py:20` | descriptor 声明 dtype 集合与整组约束，重新冻结源码、recipe、descriptor、spec 身份 |
| `flashinfer/attention/catlass_prefill.py:39`、`flashinfer/aclnn/_ops.py:65` | 真实输出、signature 和 FakeTensor 的 dtype 均遵循 Q，复用通用契约校验 |
| `csrc/kernels/ascendc/flash_attention/op_host/flash_attention.cpp:83` | InferType 保留合法输入 dtype；OpDef 对齐 FP16/BF16 的 dtype/format 列表 |
| `csrc/kernels/ascendc/flash_attention/op_kernel/catlass.cpp:19` | 使用 CANN 的 `DTYPE_Q` 实例化 CATLASS 模板，编译真实的两种计算实现 |
| `tests/flash_attention.py:25`、`flashinfer/selftest.py:83` | 同一脚本和 FA selftest 项循环两种 dtype，使用精确量化输入的 CPU FP64 参考 |

这不是 FA 专用的 dtype 转换开关。通用契约 UT 还验证其他 Tensor 参数顺序下的完整混合组合表达，但那只是契约测试，不表示相应设备算子已经支持该组合。新增 dtype 仍须实现设备模板、同步 OpDef/recipe/登记并补数值证据，步骤见 [dtype 扩展指南](../adding-a-custom-operator.md#51-扩展-dtype以-fp16bf16-flashattention-为例)。

包校验器属于构建驱动身份的一部分，修改后工具链指纹变化；因此 MatmulBiasSiLU 也重新冻结 spec 并在 wheel 构建时重建。其源码、recipe 和 descriptor 未变，装后数值自检通过，没有通过绕过缓存身份校验复用旧产物。

## 实际命令与结果

完整日志、真实退出码、JUnit、命令 JSON、wheel 和隔离环境保存在交付的 `fa-dual-dtype-20261029-oi8tm1sr` 证据目录。以下 `$WORK` 指该目录，`python3` 指已经激活的 `dy_test`。

```bash
python3 -B tools/build_flash_attention.py --freeze --output "$WORK/fa-build"
python3 -B tools/build_matmul_bias_silu.py --freeze
python3 -B tools/run_regression.py --profile quick --log-dir "$WORK/affected-quick" \
  tests/test_ascendc_contract_v2.py tests/test_flash_attention_registration.py \
  tests/test_ascendc_recipe.py tests/test_ascendc_op_package.py \
  tests/test_ascendc_v2_public.py tests/test_aot_runtime.py \
  tests/test_aot_loader_paths.py tests/test_aot_multi_operator.py \
  tests/test_matmul_bias_silu_registration.py tests/test_aclnn_adapter.py
FLASHINFER_AOT_PATHS="$WORK/fa-build/release" python3 -B -m tests.flash_attention_acceptance numerical "$WORK/numerical/cold-store"
FLASHINFER_AOT_PATHS="$WORK/fa-build/release" python3 -B -m tests.flash_attention_acceptance out_of_range "$WORK/negative/cold-store"
FLASHINFER_AOT_CACHE_DIR="$WORK/fa-build/build-cache" python3 -B setup.py bdist_wheel --dist-dir "$WORK/wheel"
```

| 检查 | 实际结果 |
| --- | --- |
| 受影响 quick UT | **202 passed、2 deselected、0 failed/errors/skipped**；pytest 7.41 秒，入口总计 8.942 秒；pytest 与入口退出码均为 0 |
| FA 冷构建 | 1 个 op_package、两种 dtype 设备 binary；`disk_cache_hit=false`，退出码 0 |
| FA 数值 | 6 组 shape × 2 dtype，共 **12 组通过**；FP16 保持原 `atol=rtol=0.001`，最大绝对误差 `0.0001220703125`；BF16 的 `atol=0.004, rtol=0.008`，最大绝对误差 `0.0007331222295761108`；退出码 0 |
| 受限输入反例 | 原 13 项全部在 dispatch 拒绝；BF16 反例已按授权替换为 FP32，runner 未解析、编译调用为 0；混合 dtype 另由契约、dispatch 及 FakeTensor UT 拒绝 |
| 默认完整 wheel 构建 | 携带 FA 与 MatmulBiasSiLU 两个生产包及原生适配库；FA 命中上述已构建缓存，未重复冷编译；退出码 0 |
| 隔离安装的无参数脚本 | `venv/bin/python -B flash_attention_acceptance.py`，工作目录为 `tests/`；导入位置确认为 venv 的 site-packages；Torch functional 和 Python out 通过原生桥执行 FP16/BF16，最大误差分别为 `0.00006122825770479845`、`0.0004284663263522809`；退出码 0 |
| 隔离安装 selftest | `venv/bin/python -B -m flashinfer selftest`：**2 PASS、0 FAIL、0 SKIP**，FA 项包含两种 dtype；禁 JIT、catlass 路径不存在，退出码 0 |

隔离环境没有使用 `--system-site-packages`：先安装 wheel，再通过纯路径 `_deps.pth` 引入 Torch 等运行依赖，确认导入的是 wheel。无参数 FA 脚本显式启用原生桥；selftest 在独立进程验证默认 Python/ACLNN 路径。

FA spec 为 `45205b744cd268bb61967777a7c0dc9df4cfd83a49b135527e80c5d7b9e2cebe`，MatmulBiasSiLU spec 为 `886f2b346e18cddafc955f250333446db4a4cda6889301db5d8d1e3e42d9ad1e`。wheel 为 `flashinfer-0.1.0.dev0-py3-none-linux_x86_64.whl`，1,664,728 字节，SHA-256 为 `c5e1fa4428593882c36efcc5bc333341eaf5cbbe859a6af59825e35f92f5cd61`。

首次配置因 OpDef 的两项 dtype 对应了一项 format，opbuild 报 `The dtype size of input[0] ... is 0`；补齐一一对应的 format 列表后构建通过。首次失败及修正后的日志都保留，不将失败尝试计为成功。

## 本轮未验证及结论更新

- 按用户要求未执行全量或完整 real_compile 矩阵；上述 202 项是受影响 quick 范围，不是全量数字，2 项编译测试为 deselected。
- 未重复独立的冷 JIT、其他 SoC、所有 queue 模式、图捕获重放或完整模型测试；不能由 AOT 与同步 eager 的通过外推这些能力。
- 本轮推翻了旧结论“FA 的 BF16 必须拒绝”及“recipe 只能表达一组固定 dtype”。新增能力为显式完整组合；省略扩展的旧配方继续固定 dtype，FA 的混合 dtype、FP32 和原 shape/layout/mask 越界仍拒绝。
