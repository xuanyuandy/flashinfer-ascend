# Ascend C 工程结构与 TorchNPU 下发重构验收（2026-10-28）

基线为 `infer` 分支 `b742c5aa84916cc2a0be2528784cf1bbc93464b5`。本次按用户确认的职责划分实施工程结构、原生调用准备及 CANN vendor 激活重构；未新增生产算子或放宽支持域。全量复核为 2003 passed，随后串行的 real_compile 为 90 passed、1913 deselected；两轮均为 0 failed/errors/skipped、退出码 0。首轮失败与修正保留在本报告及完整日志中。

## 环境与证据

实际使用 `../../tools/conda-envs/dy_test/bin/python3`，加载用户指定的 `../../tools/miniconda3/Ascend/ascend-toolkit/set_env.sh`，CATLASS 为 `../catlass-ref`。Torch `2.10.0+cpu`、TorchNPU `2.10.0.post7`，TorchNPU 对应源码提交 `3363ea59d49f3e08cfd8d4a1f532b756ccda0682`。设备自检实际为 **True 8**；本轮此前元数据探测为 Ascend950、56 核、UB 221184 字节，旧报告的 UB=0 不代表当前探测结果。

证据目录从仓库根目录计为 `../../var/flashinfer/architecture-refactor-20261028-165913/`。每个运行标签均保留 `.command.json`、完整 `.log`、真实 `.exit` 和 `.result.json`；pytest 另存 JUnit。命令先重定向到文件，再记录子进程退出码，没有用管道末端状态替代。

证据压缩包为 `../../var/flashinfer/architecture-refactor-20261028-165913-evidence.tar.gz`，SHA-256 另存同名 `.sha256` 文件；包含最终结果汇总、工作树 diff/源码快照、完整测试和编译日志及校验清单，wheel 单独保留在上述目录内。

## 已确认的架构边界

`op_package` 是 CANN 可执行交付格式，由 ACLNN host API、推形/tiling 库、设备二进制和元数据组成。AOT/JIT 均产出该格式，构建时机不同；源码目录仍按算子组织。运行期只加载已验证 vendor 树，不执行 `.run` 安装器。

对照与本机安装版本一致的 [TorchNPU OpCommand 源码](https://github.com/Ascend/pytorch/blob/3363ea59d49f3e08cfd8d4a1f532b756ccda0682/torch_npu/csrc/framework/OpCommand.cpp#L183)，`RunOpApi` 在框架队列启用、stream 允许时提交 `EXECUTE_OPAPI`，否则按框架直接执行路径处理，`sync=true` 再完成流同步。本仓原生桥原本已接入此入口；此次整理绑定准备和资源生命周期，没有自建或替换 TorchNPU 队列。

wrapper 保留负载签名、能力校验、选择及资源管理；编译特化身份与运行 shape 分开。generator 的声明/生成与 Torch schema 注册分层。跨实现 tuning 在 backend 之外，正常 dispatch 只查有效记录；当前仍没有 Triton FA，CATLASS FA 与 CANN FIA 的逻辑入口不同，未宣称完成多后端 FA 实测选优。

## 实际改动

| 文件与位置 | 改动及边界 |
| --- | --- |
| `csrc/kernels/<op>/op_host`、`op_kernel`、`include` | 16 个文件按算子聚合；移动源码字节全部不变，包括 host/device 共享 tiling 结构 |
| `csrc/cmake/OpPackage.cmake:5`、`csrc/cmake/op_package/` | 通用 CANN 构建模板移入共享构建目录；公开 recipe 更新路径，内部旧闭包保留 |
| `flashinfer/jit/op_package.py:218` | 整组验证全部文件、建立 dot-free FD 别名、先配置全部 vendor 路径，再加载 host 库；失败恢复环境并关闭新 FD |
| `flashinfer/jit/runtime.py:134`、`flashinfer/ascendc.py:102` | 首次生产 AOT 解析只依据可信 descriptor/spec 匹配当前已随包同目标生产项，整组激活；缺失同组项不触发编译 |
| `flashinfer/aclnn/_native.py:65`、`flashinfer/jit/tensor_adapter.py:108` | 首次准备并持有已验证原生调用，稳定调用不重新注册绑定 |
| `csrc/aclnn_torch_adapter/adapter.cpp:45`、`:175` | 每个不可变 Binding 准备一次 libffi 签名；通过导出的 stream API 和 RunOpApi 管理下发，caller 保留资源至框架完成同步后释放 |
| `docs/framework_design.md:278` | 增量明确 op-package、签名/特化、注册、框架级 tuning 与 TorchNPU 队列边界；保留 CRLF 和全部 62 个代码围栏 |

生产 registry 仍为 81 descriptor、43 op：Ascend C 2、CANN 40、reference 38、Triton 1。精确生产集合、op 名、dtype/shape、workspace 声明、launch/binary ABI、图能力保持；descriptor 仅 `source.locator` 随目录更新。19 份内部旧探针文件字节不变。`identity-audit.json`、`moves.json` 与基线摘要记录了比对。

静态审计保留设计文档 CRLF 和全部 62 个代码围栏，修改文档的原有代码围栏数量均未减少；本轮相对链接、Git 空白检查及 Mermaid 图解析通过。

公共配方仍恰好 7 条、全部 schema=2。源码路径与共享 CMake 闭包改变，需要按新工具链身份重冻两条生产 spec，没有修改 ABI 常量或放宽加载校验：

| 对象 | 当前身份 |
| --- | --- |
| toolchain fingerprint | `a119d499eac60507535edead3a80be6520daa5a99d4976861dd980d741f10939` |
| MatmulBiasSiLU spec | `fdbfef717d0a6ba4baca0036adab0b79938f5cff74b3adeece5cd8a8a11f659d` |
| FlashAttention spec | `9778892d99b791b5ebfbe09a89b928e102d1051480ed83b44a40198ab3d9bddd` |

## 命令与验证

下列为实际执行的命令；解释器、CANN 初始化与 CATLASS 使用上文环境。构建/测试参数中的证据目录以 `$WORK` 表示上述本轮目录，完整展开命令见对应记录。

```bash
python3 -B tools/build_matmul_bias_silu.py --freeze
python3 -B tools/build_flash_attention.py --freeze
python3 -B -m pytest -p no:cacheprovider -q -m 'not real_compile' tests/test_aclnn_adapter.py tests/test_aot_multi_operator.py tests/test_aot_loader_paths.py tests/test_ascendc_v2_public.py tests/test_ascendc_contract_v2.py tests/test_aot_runtime.py tests/test_aot_discovery_index.py --junitxml="$WORK/host-activation-final.xml"
python3 -B -m pytest -p no:cacheprovider -q -s tests/test_aclnn_adapter_real.py --basetemp="$WORK/native-queue-final-tmp" --junitxml="$WORK/native-queue-final.xml"
python3 -B setup.py build --build-base "$WORK/package-build" bdist_wheel --dist-dir "$WORK/wheel"
python3 -B -m pytest -p no:cacheprovider -q --basetemp="$WORK/full-tmp" --junitxml="$WORK/full.xml"
python3 -B -m pytest -p no:cacheprovider -q --basetemp="$WORK/full-final-tmp" --junitxml="$WORK/full-final.xml"
python3 -B -m pytest -p no:cacheprovider -q -m real_compile --basetemp="$WORK/real-compile-tmp" --junitxml="$WORK/real-compile.xml"
```

| 验证项 | 实际结果与证据标签 |
| --- | --- |
| 主机契约、激活、拒绝与索引 | 100 passed、2 deselected，退出码 0；`host-activation-final` |
| 原生冷构建与队列矩阵 | 8 passed、0 failed/errors/skipped，退出码 0；`native-queue-final` |
| 普通 wheel 冷 AOT 构建 | 2 built、0 failed、0 skipped，两个 disk_cache_hit=false；`wheel-default` |
| 普通 wheel 隔离 selftest | 2 PASS、0 FAIL、0 SKIP；`installed-default` |
| 装后队列及直接脚本 | queue=2、FA 先执行、同进程 6 次混合调用通过；两个无参数 acceptance 脚本均 PASS，所有子命令退出码 0 |
| source-only / PREBUILT wheel | compiler_calls=0，7 条 V2 recipe；source-only 无二进制，PREBUILT 含 2 包且无原生桥；`wheel-source-only`、`wheel-prebuilt` |
| PREBUILT 隔离 selftest | 2 PASS、0 FAIL、0 SKIP；`installed-prebuilt` |
| 910_9391、310p1 探针交叉编译 | 两项均冷构建、退出码 0；`cross-910_9391`、`cross-310p1`；这些设备上的数值未验证 |
| 全量最终复核 | 2003 passed、0 failed/errors/skipped、4 warnings，946.22 秒，退出码 0；`full-final` |
| real_compile | 90 passed、1913 deselected、0 failed/errors/skipped，475.70 秒，退出码 0；`real-compile`，严格在全量结束后串行执行 |

原生矩阵覆盖队列 0/1/2 与两种首调用顺序、同进程两生产包、普通 TorchNPU 运算作为 producer/consumer、非默认 stream、functional/out、非零 storage offset。混合场景每项 6 次调用，warm_rebinds=0；Matmul 消费结果最大绝对误差 0，FA 为 0.000244140625。数值容差沿用既有边界，未放宽。

隔离安装均新建 venv，不使用 `--system-site-packages`；先用 pip `--python` 安装 wheel，再添加纯路径 `_deps.pth`。测试在源码目录外运行并断言 `flashinfer.__file__` 位于该 venv。正常 wheel 原生调用也使用安装包默认产物/桥路径，禁用 JIT；PREBUILT 沿用原 Python ACLNN 路径。

## 定位过的失败及修正

- 新增负例最初构造了不匹配的 op type/ACLNN entry，被既有 recipe 校验提前拒绝；修正测试数据为合法但注册冲突的产物，没有放宽校验。整组路径顺序断言随后明确钉死为声明顺序。
- 首版桥新增引用 `OptionsManager::GetTaskQueueEnable`，编译通过但 dlopen 报未导出符号，导致 8 项失败。删除该内部依赖，改用已导出的 `NPUStream::stream(true)`；原生冷验收随后 8 项通过，未修改 TorchNPU 二进制或绕过 manifest 校验。
- 仅按 vendor 放开顺序加载时，第二个 FA 的 GetWorkspaceSize 返回 161001。对照实验先准备两个路径后通过，确认须处理首次自定义执行的 vendor 路径快照。实现改为整组校验/激活，保留同名 generation、重复 op/API 及激活后追加 vendor 的明确拒绝。未调用 CANN 未公开的 reload/reset 接口。

- 首轮全量为 1995 passed、8 failed、0 errors/skipped，退出码 1：7 项 FA 验收尝试原服务器解释器而 PermissionError，另 1 项发布检查引用迁移前路径。验收默认解释器改为当前 `sys.executable`，保留显式覆盖及解释器一致性断言；发布检查更新为 `op_kernel` 路径，保留源码逐字节一致性断言，并新增旧目录污染清理检查。生产实现与数值容差不变。相关复核随后 10 passed、退出码 0（`regression-fixes`），再执行上表完整复核。

上述失败的完整日志与真实退出码均保留，不用最终通过日志覆盖。

## 仍未验证与限制

- 当前仍为同步 eager，未验证异步返回、生产算子 capture/replay、自动微分、跨 Torch/CANN 版本或完整模型接入；图能力不提升。
- 应在进程首次自定义执行前准备完整 vendor 集合。完整 AOT wheel 自动组装当前随包生产项；单算子冷 JIT 保持请求项构建。多个冷 JIT 包或第三方自定义包需要应用提前统一准备；激活后新增 vendor、同名升级仍要求新进程。
- 当前 CANN runtime 与构建版本的自动兼容比较仍未实现。其他 SoC 的数值、任意 scalar/attribute/optional Tensor/TensorList 通用 ABI、统一 FA 多实现调优均未验证。
- 未做同口径性能比较；减少准备工作和接入队列不构成性能优势证据。原服务器 bisheng SIGKILL 的发送者仍未定位，本轮没有据此认定源码、OOM 或守护进程问题。

普通 wheel 位于证据目录 `wheel/flashinfer-0.1.0.dev0-py3-none-linux_x86_64.whl`，大小 1633242 字节，SHA-256 为 `5e4026651353f5c0b90c18162a889c256fbc5dbc2e2f228ff3d9075805e60905`。本轮验收时修改尚未提交，历史报告保持其时点证据。
