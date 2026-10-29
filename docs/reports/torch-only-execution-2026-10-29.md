# Ascend C 生产算子统一 Torch 执行验证

源码基线为 `2a71708`，结果对应本轮工作树，包含此前尚未提交的通用编译框架修改。此前报告仍是各自时点的证据，本报告不替换其历史结果。

设备自检实际输出 `True 8`。使用 dy_test 的 Python 3.12、Torch 2.10.0+cpu、torch_npu 2.10.0.post7、CANN 9.2.0；环境通过用户指定的 `ascend-toolkit/set_env.sh` 设置，未沿用 AGENTS.md 的原服务器绝对路径。

## 实际改动与边界

| 位置 | 改动 |
| --- | --- |
| `flashinfer/attention/catlass_prefill.py:15`、`flashinfer/gemm.py:187` | FA 与 MatmulBiasSiLU 公共函数变为自动注册并转发的薄封装；保留 out 返回值及 backend/kernel_id 参数 |
| `flashinfer/aclnn/_ops.py:93` | 只注册四个生产 Torch schema，首次成功后复用；不再设置全局执行策略，不连带启用 norm 路由 |
| `flashinfer/aclnn/_ops.py` | PrivateUse1 callback 进入私有实现，继续构造 signature、dispatch、分配 out、解析 runner；不回调公共封装；四个 schema 增加兼容的可选选型关键字 |
| `flashinfer/jit/tensor_adapter.py:81` | 移除 Python ctypes 创建 Tensor/查询 workspace/launch 的执行分支，统一调用原生 Torch 算子；原契约门禁保留 |
| `csrc/aclnn_torch_adapter/adapter.cpp:228` | 增加只读诊断，读取真实原生 GetWorkspaceSize 返回值；旧 workspace 下界及随 shape 变化断言继续验证 |
| `flashinfer/aclnn/build.py:13`、`setup.py` | PREBUILT 仍不调用编译器，同时验证并复制预编译桥；缺桥、摘要/版本/源码身份不符时构建失败 |

唯一生产执行链为：Python 薄封装或直接 Torch 调用 → Torch PrivateUse1 → signature/dispatch → AOT/JIT → verified Tensor runner → C++ TorchNPU OpCommand → ACLNN → kernel。

`enable_aclnn()` 是幂等注册入口。Python 公共函数自动调用，直接 Torch 调用可在初始化阶段显式调用一次；它不再切换模式。首次真实 NPU 调用仍延迟加载桥和算子包。缺桥、损坏和版本错误明确失败。绑定仍复用，ACLNN host tiling/workspace 查询仍逐次执行。

本次用户明确要求去掉默认执行分支，因此旧的“默认不选原生桥”断言改为“始终要求原生桥”；新增 Python/Torch 同一路由与 CPU 拒绝断言。原 host 元数据测试转向注册回调使用的私有实现，原形状/dtype/输出别名断言未放宽，另用真机公共入口验证拒绝路径。生产 registry/spec、设备源码、recipe、冻结 ABI 常量均未修改。CANN、Triton 和 plan/run 不做本次链路迁移。

## 实际命令与结果

`WORK` 为仓库外本轮证据目录，名称为 `torch-only-20261029-h7p21fj0`。完整日志、JUnit、命令 JSON、原始退出码与汇总均保存其中。按用户要求只做受影响回归；AOT 复用已校验缓存，主机桥重新编译，两个冷 JIT 各使用独立空 store。补充脚本与装包验证设置 `OMP_NUM_THREADS=4`，只压缩参考计算的主机耗时，不作算子性能结论。

```bash
python3 -B -c 'import torch,torch_npu; print(torch.npu.is_available(), torch.npu.device_count())'

python3 -B tools/run_regression.py --profile quick \
  --log-dir "$WORK/targeted-quick-final" \
  tests/test_aclnn_adapter.py tests/test_flash_attention_registration.py \
  tests/test_matmul_bias_silu_registration.py tests/test_ascendc_v2_public.py \
  tests/test_ascendc_framework.py tests/test_release_checks.py \
  tests/test_torch_library.py tests/test_ascendc_torch_library.py \
  tests/test_aot_runtime.py tests/test_aot_multi_operator.py tests/test_aot_loader_paths.py

PYTEST_ADDOPTS="--basetemp=$WORK/compile-cases" \
python3 -B tools/run_regression.py --profile compile \
  --log-dir "$WORK/targeted-compile" tests/test_aclnn_adapter_real.py

python3 -B -m tests.matmul_bias_silu_acceptance missing_enabled "$WORK/matmul-cold-jit-store"
python3 -B -m tests.flash_attention_acceptance missing_enabled "$WORK/attention-cold-jit-store"

python3 -B -m tests.matmul_bias_silu_acceptance numerical "$WORK/matmul-aot-unused-jit-store"
python3 -B -m tests.flash_attention_acceptance numerical "$WORK/attention-aot-unused-jit-store"

python3 -B setup.py build --build-base "$WORK/wheel-build" bdist_wheel --dist-dir "$WORK/wheel"
FLASHINFER_AOT_PREBUILT="$WORK/compile-cases/native-aclnn0/release" \
python3 -B setup.py build --build-base "$WORK/prebuilt-build" bdist_wheel --dist-dir "$WORK/prebuilt-wheel"
```

冷 JIT 不设置额外 AOT roots；numerical 与 compile 用例使用本轮已验证 release。源码验收配置本轮桥目录 `FLASHINFER_ACLNN_ADAPTER_PATH`；PREBUILT 也从该目录复制桥。构建命令及环境差异保存在 `jit-validation.json`、`aot-numerical-validation.json` 和 `packaging-validation.json`。旧 store 不应复用于“冷”验证。

| 检查 | 实际结果 | 原始退出码 |
| --- | --- | --- |
| 最终定向 quick | 181 passed、6 deselected、0 failed/errors/skipped；pytest 17.34 秒 | 0 |
| 受影响 real_compile | 9 passed、0 failed/errors/skipped；pytest 92.34 秒 | 0 |
| 冷 JIT | Matmul 1 组、FA 两种 dtype 共 2 组；各 compile/toolchain_calls=1，重复调用复用绑定 | 均为 0 |
| AOT numerical | Matmul 8 组，workspace 随 shape 变化；FA 6 shape × 2 dtype 共 12 组；compile/toolchain_calls=0 | 均为 0 |
| 隔离 wheel selftest | 2 PASS、0 FAIL、0 SKIP；禁 JIT、CATLASS 配置目录不存在、PATH 无 bisheng | 0 |
| 隔离 wheel 无参数 FA 脚本 | FP16/BF16、Torch functional 与 Python out 通过 | 0 |
| 真机公共入口拒绝 | Python/Torch 两种写法共 16 项：shape、混合 dtype、输出重叠、causal、layout、mask、backend、错误 kernel；runner 未解析 | 0 |
| wheel 缺桥拒绝 | 公共 Python 入口明确报 adapter is not bundled；禁止 ctypes fallback 的探针未被触发 | 0 |

9 个编译标记用例包含三个数值接口组合，以及 queue 0/1/2 × 两种首调用顺序；覆盖非默认 stream、非零 storage offset、同进程双包和稳定调用不重新绑定。数值接口先调用 Python 封装自动注册，随后调用直接 Torch functional/out，并禁止 Python ACLNN launch，确认统一边界。

普通与 PREBUILT wheel 均含两个 op-package 和原生桥，ZIP 每个成员内容完全相同。普通 wheel SHA-256 为 `37256c9800a3303b728fb8942807c7f9096b2a7aed0ceac79f88efba91fabf15`。只对普通 wheel 建干净 venv 安装验证，未重复安装内容相同的 PREBUILT wheel；未使用 system-site-packages，只通过纯路径 `_deps.pth` 提供基础依赖，实际导入位置断言在新 venv 内。

文档增量更新 README、安装说明、Torch/ACLNN 指南、FA 调用链、编译选项与框架设计；代码块数量均保留，框架设计继续使用原 CRLF。

## 未执行与被替代的结论

- 本轮未执行全量与完整 real_compile 矩阵；181/9 是定向数字，不是全仓回归结果。
- 未验证其他 SoC、跨 Torch/CANN 版本、完整模型、多线程调用、capture/replay、异步返回和自动微分；生产自研算子继续为 NOT_CAPTURABLE。
- “默认 Python ctypes 执行、enable_aclnn 后切换原生桥”已被本轮统一 Torch 执行替代，历史报告不代表当前策略。
- 源码模式与 PREBUILT 仍不自行编译桥；源码运行须准备桥，PREBUILT 必须提供匹配的预编译桥，不能再依赖旧 Python 回退。
