# Ascend C 契约 v2 与首个生产算子闭环

日期：2026-10-25。任务：T2′。工作目录：`/mnt/fi/flashinfer`，分支：`infer`。本报告记录 `matmul_bias_silu` 的契约、实现、交付路径及真机验收。
P0-1 的 ABI 重冻常量、既有测试断言和 P0-2 discovery/fallback 机制不在本次重写范围。
`/mnt/fi/catlass-ref` 只作参考；生产 kernel 为本仓独立实现，没有复制动态 matmul 的多变体生成系统。

## 1. 环境与证据口径

统一解释器为 `/root/miniforge3/envs/flashinfer-a5/bin/python`；所有 Python 命令使用 `-B`。pytest 使用 `-p no:cacheprovider`，没有使用未安装的 pytest-timeout 插件或 `--timeout` 参数。
测试子进程使用标准库 `subprocess.run(timeout=240)` 约束运行时间，不依赖 pytest 插件。

设备自检命令：

```bash
cd /mnt/fi/flashinfer
ls /dev/davinci* | head
/root/miniforge3/envs/flashinfer-a5/bin/python -B -c "import torch,torch_npu;print(torch.npu.is_available(),torch.npu.device_count())"
```

设备自检实际输出：

```text
/dev/davinci0
/dev/davinci1
/dev/davinci2
/dev/davinci3
/dev/davinci4
/dev/davinci5
/dev/davinci6
/dev/davinci7
/dev/davinci_manager
True 8
```

另实际查询到 `device: Ascend950PR_9579`；`bisheng: /usr/local/Ascend/cann-9.2.0-beta.2/bin/bisheng`；runtime/version.info 为 `Version=9.2.0-beta.2`。
真机验收要求设备可见；设备不可见或测试 skip 不计为通过。
本任务目标环境为 8× Ascend950PR、CANN 9.2.0-beta.2；其他环境的支持不能从目标名称映射推断。

## 2. v1 → v2 契约差异

`BuildRecipe.schema_version` 是契约版本；`BuildRecipe.version` 是该 recipe 自身的修订版本。v2 继续使用 `output_adapter="ascendc.op_package.v1"`：后者表示包容器格式，并不表示 recipe 版本。
两层版本分别参与既有 spec/manifest 绑定，没有更改被冻结的 ABI 常量或 adapter hash。

| 维度 | v1 限制及原因 | v2 放开方式 | 新增校验与拒绝条件 |
|---|---|---|---|
| 任务类型 | 只允许 `aic_only`；原验证器只理解单 AIC 可执行文件 | 增加 `aic_aiv`，保留 AIC-only 分支 | 混合任务必须声明 `cross_core_flags`；不接受未知 task/sync 组合 |
| 同步策略 | `compiler_default`；探针没有跨核数据依赖 | 显式声明硬件 cross-core flag 同步 | 验证 MIX 元数据的 `intercoreSync=1` 和每个 kernel 的 `crossCoreSync=1` |
| Workspace | 必须为 0；原探针没有中间结果的动态生命周期 | 非负固定整数或字符串 `"dynamic"` | 拒绝负数、布尔值和未知字符串；动态值必须给出合法 `workspace_host_entry` |
| Tiling | host_shared 为 `none`、op_package 为 `fixed`，schema 固定 1 | v2 为 op_package 的 `host_dynamic`，tiling schema ≥2 | 必须有合法 `tiling_host_entry`；包 metadata 必须声明 dynamic op mode |
| 调用参数 | 固定三个 `aclTensor*`，符合原探针 a/b/c | 有序 `tensor_parameters` 描述任意正数量 tensor | 名称唯一，类型受支持，角色为 input/output，输入在输出之前，保留名称不可占用 |
| Shape | 原探针固定形状，扩展不能从其代码自动推断 | 每个参数声明完整 shape，维度可为正整数或符号 | 每个符号恰有一个有限、非空、正整数范围；dispatch 检查范围、倍数、rank 和跨张量一致性 |
| Device ELF | 单 AIC 元数据名必须对应 ELF symbol | MIX 的每个逻辑 kernel 对应 AIC/AIV 两个实际入口 | 要求 `_mix_aic` 与 `_mix_aiv` 同时存在，校验 MIX magic、NPU ET_EXEC 和文件 digest |
| 身份 | v1 recipe 的 canonical JSON 已形成现有产物身份 | v2 schema 和全部新增字段进入 recipe fingerprint | v1 序列化排除新增默认字段；v2 不可能复用 v1 recipe fingerprint |

主要入口：`flashinfer/jit/recipe.py:58` 的 `TensorParameter`、`:83` 的 `ShapeConstraint`、`:197` 的独立 v2 校验；`:241` 检查调用元数据，`:183` 保留 v1 护栏，`:279` 保留 v1 序列化，`:310` 计算包含契约版本的 fingerprint。
v1 独立保留原有 tiling、workspace、task、sync 护栏，不因新增 v2 放宽；同时拒绝携带 v2 参数、shape 或 host callback 声明。
`examples/ascendc_native.json` 与 `examples/ascendc_catlass.json` 的配置和相关既有测试未改动。
新增测试把每条原始 matmul_probe JSON 与 `to_dict()`、canonical fingerprint 逐项比较，确认 host_shared 和 op_package 两类 v1 产物的既有身份均保持不变。

Host callback 字段记录实现入口声明并参与身份，不是 Python 另行调用的函数名。
实际执行由 CANN 的 `AICore().SetTiling(...)` 注册完成；本算子在同一 callback 中计算 tiling 与 workspace。
构建准备验证源码闭包，运行期验证其冻结身份与包元数据，但不能仅凭 metadata 证明任意 C++ callback 函数体的数学正确性；动态计算由 host 检查、代码审查和真实运行共同验证。

## 3. 通用 ABI 的两层参数约定

ACLNN host ABI 保持两段式；workspace 和 tiling 不能当作额外的用户 tensor 混入第一段。

```text
GetWorkspaceSize(
    aclTensor* argument_0, ..., aclTensor* argument_n,
    uint64_t* workspace_bytes, aclOpExecutor** executor)
    -> CANN status
launch(void* workspace, uint64_t workspace_bytes,
       aclOpExecutor* executor, aclrtStream stream)
    -> CANN status
```

第一段 tensor 数量与顺序来自 recipe，名称来自 `tensor_parameter_names`；`OpPackageModule.get_workspace_size(*tensors)` 接受位置参数，也支持按 recipe 名称传关键字参数。
参数个数不符、名称缺失或位置/关键字混用在调用 CANN 前拒绝。
`load_op_package()` 根据声明数量设置 `ctypes.argtypes`，没有 `matmul_bias_silu` 名称特判。
第一段返回的 executor 绑定本次 host tiling；第二段必须使用其返回的 workspace 大小。
两段均检查非零 CANN 状态；第一段另检查 executor 非空。

本算子的实际 device ABI 为：

```text
fi_matmul_bias_silu(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR out,
                   GM_ADDR workspace, GM_ADDR tiling)
```

`tiling` 是 CANN 将 host tiling 数据下发后提供的 device GM buffer；应用不用创建第五个 aclTensor，也不用绕过 executor 手工上传 tiling。
`device_parameter_order` 表达这层顺序：声明的 tensor 参数，随后为 workspace、tiling。
v1 `matmul_probe` 默认 `tensor_parameter_names=(a,b,c)`，因此原三参数调用和 stage-two launch 保持有效。
没有增加第二个 package loader；AOT/JIT 都复用 `load_op_package()` 和原 store 验证流程。

`flashinfer/jit/tensor_adapter.py` 负责通用 torch tensor → aclTensor 转换。
它校验 recipe 的 dtype/rank/shape、同一 NPU、连续 ND，以及输出与其他参数没有存储重叠。
参数数据、维度数组、aclTensor 和 workspace 保活到当前 stream 同步完成，再释放描述符。
这一版是同步 eager adapter；非 tensor scalar/attribute ABI、异步生命周期和图捕获没有扩展。

## 4. 生产算子的实现与范围

公开语义：`out = silu(A @ B + bias)`；输入 A/B/bias 和输出均为 BF16、连续 ND，输出保持 BF16。
A 为 `[M,K]`，B 为 `[K,N]`，bias 为 `[N]`，out 为 `[M,N]`。
M/N/K 各为 16 的倍数且位于闭区间 `[16,256]`；仅支持 bias[N] 沿 M 广播，不支持其他 bias 形状、转置 B 或其他 rank。
这是一条正确性优先的实现，不声称性能优于 CANN、CATLASS 或其他后端。

| 交付文件 | 职责 |
|---|---|
| `csrc/kernels/ascendc/matmul_bias_silu/native.cpp` | 原生 BF16 MMAD cube、FP32 scratch、vector bias-add/SiLU、BF16 输出 |
| `csrc/op_package/matmul_bias_silu.cpp` | OpDef、shape/type inference、host tiling、动态 workspace 大小 |
| `csrc/include/flashinfer_ascend/matmul_bias_silu_tiling.h` | host/device 共用 POD tiling 数据 |
| `csrc/recipes/matmul_bias_silu.json` | v2 完整源闭包、参数、shape 范围、同步和入口声明 |
| `flashinfer/data/kernels/registry.json` | 第一条生产 Ascend C descriptor，`graph_capability=not_capturable` |
| `flashinfer/data/kernels/ascendc_specs.json` | 发布期冻结 recipe 与 JitSpec，运行期无需读源码或发现编译器 |
| `flashinfer/ascendc.py` | 注册真实 lazy runner、公开契约校验、受控 JIT backend factory |
| `flashinfer/gemm.py` | 公共 `matmul_bias_silu(a,b,bias,out=None,...)` 入口 |
| `tools/build_matmul_bias_silu.py` | 显式 freeze、真实构建、复用 store/export 导出 release |

Cube 每个 tile 写出后以 PIPE_ALL 保证写完成再复用 L0C；用 16×16×16 分块的 MMAD 将结果写入 FP32 scratch；vector subcore 消费完成后的结果，跨核顺序由 FIX pipe 的硬件 cross-core flag 保证，flag 本身不存放在 scratch 中。
Vector 执行 bias add、exp/div 实现 SiLU，再舍入为 BF16。
Host 根据 M/N/K 写入 tiling，而不是为每个 shape 生成新代码或选用 14 个模板变体。
用户 scratch 为 `4*M*N` 字节；ACLNN 总 workspace 还包含 CANN 系统保留区。
本机 CANN 系统保留区为 16 MiB；实际总 workspace 为 `16777216 + 4*M*N`。16×16 为 16778240 字节，16×32 为 16779264 字节，256×256 为 17039360 字节。descriptor 给出最后这个保守规划上界，实际每次仍按 host 结果动态分配。
真实分配大小始终以 GetWorkspaceSize 返回值为准，不能只分配用户 scratch 的字节数。

## 5. 登记与公共 API 的执行链

新增 kernel ID：`ascendc/t2/matmul_bias_silu/native/op_package/Ascend950`。
语义 op：`gemm.matmul_bias_silu`；backend：`ascendc`；compile_mode：`aot`；tiling schema：2。
recipe fingerprint：`98fae660c68ab744db8952b823d8348dff244fea3c8318dab3d4418aa8b25079`。
冻结 spec：`4e13377d4b8047efbb9f0c98f8f633b842887dcbaf9741bb2b5f4c080e7334cd`。
verified release 保存在 `/mnt/fi/t2-delivery/4e13377d4b8047efbb9f0c98f8f633b842887dcbaf9741bb2b5f4c080e7334cd`；通过现有 `export_op_package()` 导出，未运行 installer。
实际 `/root/miniforge3/envs/flashinfer-a5/bin/python -B -m flashinfer list-kernels` 输出：

```text
ascendc/t2/matmul_bias_silu/native/op_package/Ascend950	gemm.matmul_bias_silu	ascendc	aot (runtime discovery)
```

Capability 声明 BF16、Ascend950、ND、每个维度的范围，以及 `tensor_contract=supported` 条件；完整 recipe 进一步校验倍数、共享 K/N/M 符号、参数 rank/type 和布局。

调用顺序为公共 API → 构造签名 → 普通全局 registry dispatch → 已绑定的 `AotKernelRunner`。缺省 out 先用预期输出元数据参与 dispatch，选择成功才分配输出；超大 shape 不会先触发输出 OOM。
`flashinfer/runtime/registry.py:301` 将随包生产 manifest 合入全局 registry 后冻结；原 Python 静态 catalog 的 79 条保持不变，全局总数为 80。
Runner 使用既有 discovery index 查找并验证产物，再由既有 loader 加载并建立通用 tensor adapter。
只在 `AotMissingError` 且策略 ENABLED 时，调用 backend factory 进行 toolchain/source resolution 与 JIT。
corrupt 不触发 JIT；load error 也不触发 JIT；不兼容目标提供明确拒绝原因。
`compile_mode=aot` 是登记属性，实际使用来源由 `runner.provenance.source` 给出；验收分别检查 dispatch 的 kernel ID、`source=aot/jit` 和 compiler_calls，不能只看 descriptor 标签。
每个验收场景使用新进程，符合 CANN 当前单 package identity 的进程生命周期限制。
运行数据核对结果：registry.json 生产条目为 1，全局 runtime entries 为 80，source_digests=all_match，workspace_upper_bound=17039360。

## 6. 新增自定义算子的端到端交付清单

本表逐项对应 [ascend-parity-analysis.md §7.4](ascend-parity-analysis.md#74-外部用户接入一个-catlass-算子的端到端清单) 的十步。旧报告反映其审计时状态；本任务不修改该历史文档，在这里记录哪些阻塞已解除。

| 步骤 | 现在仍需交付什么 | 本次带来的简化或边界 |
|---|---|---|
| 1. 数学、tensor、ABI 与 reference | 精确公式、dtype/layout/shape、输出 mutation、容差与参考实现 | v2 可直接声明混合核、动态 tiling/workspace 和变参 tensor；无需先拆除 v1 护栏 |
| 2. Device translation unit | 自研 Ascend C 或依赖明确 revision 的 CATLASS kernel，以及实际架构支持 | 不自动生成数学实现；本次示例证明单变体 cube/vector 可以共用一个 v2 包 |
| 3. Host 桥 | OpDef、shape/type inference、SetTiling callback、workspace 计算和 POD 数据 | 延续标准 CANN 两段 ACLNN；tiling 下发由 executor 完成，无需再写 Python 上传协议 |
| 4. Recipe 与配置 | 全部 source/header SHA、目标、entry、flags、tensor_parameters、shape_constraints、同步策略 | 新字段统一校验并进入 fingerprint；不必为第四、第五个 tensor 修改 loader |
| 5. 工具链与依赖 resolve | 发布/编译时显式 discover 与 resolve；可选 CATLASS checkout 和固定 revision | 保留完整闭包校验；AOT 消费只读随包冻结 JSON，不读取构建源码或发现编译器 |
| 6. Spec 与构建 | descriptor、环境身份、specialization、declare_kernel、真实编译及 verified record | 复用 AscendCBackend/DiskJitBuildStore；本算子脚本示范 freeze/build/export，未新增第二套 cache |
| 7. Verified module → callable | 选择通用 tensor adapter，确保参数语义和 mutation 与 recipe 一致 | 固定三个 aclTensor 的限制已解除；任意 tensor 数量走同一 OpPackageModule 和 TorchOpPackageRunner |
| 8. Public dispatch/meta | public facade、真实 runner binding、capability、登记项及冻结 spec | 普通 API 可自动消费登记的 AOT；需要 torch.library/meta 的算子仍须自行交付，本次没有自动生成它们 |
| 9. Eager/compile/capture 验证 | 真机数值、边界、别名、错误路径和 AOT/JIT 七场景；其他执行模式分别测试 | 新增复用验收模式；eager 成功不升级 graph_capability，compile/capture 未因此获得证明 |
| 10. 预编译与发布 | release、manifest、完整 source closure、安装目录及目标环境验收 | 复用 P0-2 discovery/fallback 后公共 API 不需手工 load；通用 AOT CLI 的全 family 枚举和安装 wheel 验收仍未完成 |

对自定义 namespace 的注册与 opt-in 规则沿用现有 registry，不因这条内置生产记录改变。
发布作者需要在源码或 descriptor 改变后重新 freeze；运行期不会重写冻结身份。构建环境指纹和源码闭包不匹配时，受控 JIT 明确拒绝，不能拿旧 spec 为新源码编译。

## 7. 七个真机场景：实际命令与输出

场景使用 `tests/test_matmul_bias_silu_real.py`，其子进程入口为 `tests.matmul_bias_silu_acceptance`；所有计算调用 `flashinfer.gemm.matmul_bias_silu`，没有替换 registry、runner、loader 或数值实现。
仅对 compile/toolchain discovery 增加计数，并观测原 GetWorkspaceSize 返回值；使用空 JIT store，避免 cache hit 伪装成真实编译。
实际总执行命令如下；它覆盖原 36 项 real_compile 与本次 7 项，以下逐项列出真正执行的子命令和 JSON 原始输出。

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python -B -m pytest -p no:cacheprovider -q -s -m real_compile --basetemp=/mnt/fi/t2-real-final
```

fixture 使用 build_release() 真实构建并 export；AOT hit 临时安装到默认目录，其他非 missing 场景通过 FLASHINFER_AOT_PATHS 指向测试副本或 release。子命令需配合这些 fixture 环境执行，逐项重现可直接运行对应 pytest 节点。

### 7.1 命中 AOT，禁 JIT 且不可见编译器

构建 release 后临时放入 `flashinfer/data/op_packages/`，结束后由 fixture 清理自己创建的副本。
子进程设置 JIT DISABLED、PATH=/usr/bin:/bin；配置一个仅链接 lib64/share/conf/opp/runtime 的 CANN 运行时目录，不含 bin/bisheng。torch_npu 导入要求存在的 compiler 目录为空；工具链 discover 入口也被阻断。
保留平台配置是 host tiling 的运行时要求；没有卸载共享机器的 SDK。输出同时断言 compiler_path_visible=false、compiler_home_visible=false。
检查 `source=aot`、compiler_calls=0、toolchain_calls=0、输出数值和重复调用一致性。
实际命令：`/root/miniforge3/envs/flashinfer-a5/bin/python -B -m tests.matmul_bias_silu_acceptance hit_disabled /mnt/fi/t2-real-final/test_production_aot_without_co0/cold-store`。

```text
{"compiler_calls": 0, "compiler_home_visible": false, "compiler_path_visible": false, "directory": "/mnt/fi/flashinfer/flashinfer/data/op_packages/4e13377d4b8047efbb9f0c98f8f633b842887dcbaf9741bb2b5f4c080e7334cd", "errors": [{"aclnn_workspace_bytes": 16779264, "atol": 0.008, "max_abs_fp32": 0.0019237995147705078, "max_abs_fp64": 0.0019238364591760426, "rtol": 0.008, "shape": [16, 32, 48]}], "graph_capability": "not_capturable", "kernel_id": "ascendc/t2/matmul_bias_silu/native/op_package/Ascend950", "policy": "disabled", "repeated_public_call": true, "result": "PASS", "scenario": "hit_disabled", "source": "aot", "toolchain_calls": 0}
```

### 7.2 Missing，允许真实 JIT

discovery 中没有对应 release，JIT ENABLED，冷 store；必须调用真实 bisheng 构建，检查 `source=jit`、compiler_calls=1、toolchain_calls=1 与数值正确。
实际命令：`/root/miniforge3/envs/flashinfer-a5/bin/python -B -m tests.matmul_bias_silu_acceptance missing_enabled /mnt/fi/t2-real-final/test_production_missing_real_j0/cold-store`。

```text
{"compiler_calls": 1, "compiler_home_visible": true, "compiler_path_visible": true, "directory": "/mnt/fi/t2-real-final/test_production_missing_real_j0/cold-store/jit/.objects/4e13377d4b8047efbb9f0c98f8f633b842887dcbaf9741bb2b5f4c080e7334cd-e51f56e695d34000a6bb41b098eb7603", "errors": [{"aclnn_workspace_bytes": 16779264, "atol": 0.008, "max_abs_fp32": 0.0019237995147705078, "max_abs_fp64": 0.0019238364591760426, "rtol": 0.008, "shape": [16, 32, 48]}], "graph_capability": "not_capturable", "kernel_id": "ascendc/t2/matmul_bias_silu/native/op_package/Ascend950", "policy": "enabled", "repeated_public_call": true, "result": "PASS", "scenario": "missing_enabled", "source": "jit", "toolchain_calls": 1}
```

### 7.3 Missing，禁止 JIT

检查 `no matching AOT op-package`、disabled 策略、compiler_calls=0 和未产生构建记录。
实际命令：`/root/miniforge3/envs/flashinfer-a5/bin/python -B -m tests.matmul_bias_silu_acceptance missing_disabled /mnt/fi/t2-real-final/test_production_missing_disabl0/cold-store`。

```text
{"category": "missing", "compiler_calls": 0, "diagnostic": "no matching AOT op-package for kernel_id=ascendc/t2/matmul_bias_silu/native/op_package/Ascend950, target_soc=Ascend950, spec=4e13377d4b8047efbb9f0c98f8f633b842887dcbaf9741bb2b5f4c080e7334cd; searched paths=/mnt/fi/flashinfer/flashinfer/data/op_packages; found 0 candidates; compilation policy=disabled; JIT fallback is not permitted", "fallback": false, "kernel_id": "ascendc/t2/matmul_bias_silu/native/op_package/Ascend950", "policy": "disabled", "result": "PASS", "scenario": "missing_disabled"}
```

### 7.4 Corrupt，JIT 保持 ENABLED

只损坏 release 副本中的已登记文件字节，使 inventory digest 不匹配；必须得到 corrupt 类别、fallback=false、compiler_calls=0，不能把损坏解释为 missing。
实际命令：`/root/miniforge3/envs/flashinfer-a5/bin/python -B -m tests.matmul_bias_silu_acceptance corrupt /mnt/fi/t2-real-final/test_production_corrupt_reject0/cold-store`。

```text
{"category": "corrupt", "compiler_calls": 0, "diagnostic": "corrupt AOT op-package /mnt/fi/t2-real-final/test_production_corrupt_reject0/damaged/4e13377d4b8047efbb9f0c98f8f633b842887dcbaf9741bb2b5f4c080e7334cd: SchemaError: JIT cache file inventory mismatch: packages/vendors/flashinfer_t2_matmul_bias_silu/op_api/lib/libcust_opapi.so: digest mismatch (expected b7d9e08f9597144b89b3633b9dd7250c489695bff5280644b0de90b393850f17, got 1b4824289dd4ad4a27502420d7ffd67caf6a1e78e0b3c7e8014a3feaf97c3c34)", "fallback": false, "kernel_id": "ascendc/t2/matmul_bias_silu/native/op_package/Ascend950", "policy": "enabled", "result": "PASS", "scenario": "corrupt"}
```

### 7.5 Incompatible target_soc

测试重新发布 inventory 合法但 target_soc 为 Ascend950DT 的副本，以隔离目标不匹配与文件损坏。
该错误目标的可执行文件不会执行；检查 `all incompatible` 和 `target_soc mismatch` 的拒绝原因。
实际命令：`/root/miniforge3/envs/flashinfer-a5/bin/python -B -m tests.matmul_bias_silu_acceptance incompatible /mnt/fi/t2-real-final/test_production_incompatible_s0/cold-store`。

```text
{"category": "missing", "compiler_calls": 0, "diagnostic": "no matching AOT op-package for kernel_id=ascendc/t2/matmul_bias_silu/native/op_package/Ascend950, target_soc=Ascend950, spec=4e13377d4b8047efbb9f0c98f8f633b842887dcbaf9741bb2b5f4c080e7334cd; searched paths=/mnt/fi/flashinfer/flashinfer/data/op_packages, /mnt/fi/t2-real-final/test_production_incompatible_s0/other-product/jit/.objects/6dc0328d6543863a7f8ca5af744443b6bbe3f454a6c295b3844247841f26390e-a8f8adeb0a2e499bb5ff5f2829aef920; found 1 candidate(s), all incompatible: /mnt/fi/t2-real-final/test_production_incompatible_s0/other-product/jit/.objects/6dc0328d6543863a7f8ca5af744443b6bbe3f454a6c295b3844247841f26390e-a8f8adeb0a2e499bb5ff5f2829aef920: target_soc mismatch: package=Ascend950DT, requested=Ascend950, spec fingerprint mismatch: package=6dc0328d6543863a7f8ca5af744443b6bbe3f454a6c295b3844247841f26390e, requested=4e13377d4b8047efbb9f0c98f8f633b842887dcbaf9741bb2b5f4c080e7334cd, descriptor/ABI incompatible: JIT compilation unit does not bind kernel source and target_soc; compilation policy=disabled; JIT fallback is not permitted", "fallback": false, "kernel_id": "ascendc/t2/matmul_bias_silu/native/op_package/Ascend950", "policy": "disabled", "result": "PASS", "scenario": "incompatible"}
```

### 7.6 FP32/FP64 数值对照与动态 workspace

CPU 先生成并量化到 BF16，再将同一组精确输入值分别用于 NPU 与 FP32/FP64 reference；reference 为 `silu(A.float/double() @ B.float/double() + bias.float/double())`。
每个元素须满足 `|actual-reference| <= 0.008 + 0.008*|reference|`，不是只比较总量或均值。
测试覆盖 8 组 M/N/K，包括 16 和 256 边界、矩形形状、多个 K tile 与显式 out。
逐形状最大绝对误差、workspace 总量，以及实际命令如下。实际命令：`/root/miniforge3/envs/flashinfer-a5/bin/python -B -m tests.matmul_bias_silu_acceptance numerical /mnt/fi/t2-real-final/test_production_numerical_fp320/cold-store`。

```text
{"compiler_calls": 0, "compiler_home_visible": true, "compiler_path_visible": true, "directory": "/mnt/fi/t2-real-final/matmul-bias-silu0/release/4e13377d4b8047efbb9f0c98f8f633b842887dcbaf9741bb2b5f4c080e7334cd", "errors": [{"aclnn_workspace_bytes": 16779264, "atol": 0.008, "max_abs_fp32": 0.0019237995147705078, "max_abs_fp64": 0.0019238364591760426, "rtol": 0.008, "shape": [16, 32, 48]}, {"aclnn_workspace_bytes": 16778240, "atol": 0.008, "max_abs_fp32": 0.0009612143039703369, "max_abs_fp64": 0.0009612185805425444, "rtol": 0.008, "shape": [16, 16, 16]}, {"aclnn_workspace_bytes": 16785408, "atol": 0.008, "max_abs_fp32": 0.00386810302734375, "max_abs_fp64": 0.0038680358663951697, "rtol": 0.008, "shape": [32, 64, 80]}, {"aclnn_workspace_bytes": 16792576, "atol": 0.008, "max_abs_fp32": 0.002743244171142578, "max_abs_fp64": 0.002743359468443618, "rtol": 0.008, "shape": [80, 48, 32]}, {"aclnn_workspace_bytes": 16842752, "atol": 0.008, "max_abs_fp32": 0.006898641586303711, "max_abs_fp64": 0.006898552408690417, "rtol": 0.008, "shape": [128, 128, 128]}, {"aclnn_workspace_bytes": 17039360, "atol": 0.008, "max_abs_fp32": 0.007798671722412109, "max_abs_fp64": 0.0077984285240124684, "rtol": 0.008, "shape": [256, 256, 256]}, {"aclnn_workspace_bytes": 16793600, "atol": 0.008, "max_abs_fp32": 0.0078029632568359375, "max_abs_fp64": 0.007803267188764451, "rtol": 0.008, "shape": [256, 16, 256]}, {"aclnn_workspace_bytes": 16793600, "atol": 0.008, "max_abs_fp32": 0.0018034577369689941, "max_abs_fp64": 0.0018034314046528888, "rtol": 0.008, "shape": [16, 256, 16]}], "graph_capability": "not_capturable", "kernel_id": "ascendc/t2/matmul_bias_silu/native/op_package/Ascend950", "policy": "disabled", "repeated_public_call": true, "result": "PASS", "scenario": "numerical", "source": "aot", "toolchain_calls": 0}
```

数值摘要（精确原始值见上方 JSON）：

| M,N,K | max_abs_fp32 | max_abs_fp64 | ACLNN workspace 字节 |
|---|---:|---:|---:|
| 16,32,48 | 0.001923799515 | 0.001923836459 | 16779264 |
| 16,16,16 | 0.000961214304 | 0.0009612185805 | 16778240 |
| 32,64,80 | 0.003868103027 | 0.003868035866 | 16785408 |
| 80,48,32 | 0.002743244171 | 0.002743359468 | 16792576 |
| 128,128,128 | 0.006898641586 | 0.006898552409 | 16842752 |
| 256,256,256 | 0.007798671722 | 0.007798428524 | 17039360 |
| 256,16,256 | 0.007802963257 | 0.007803267189 | 16793600 |
| 16,256,16 | 0.001803457737 | 0.001803431405 | 16793600 |

### 7.7 声明范围外的 shape

M=272 的调用必须在 dispatch 阶段失败，错误含实际值和声明上界 256。
检查 compiler_calls=0、runner_resolved=false；此外 host 测试覆盖非 16 倍数和跨参数维度矛盾。
实际命令：`/root/miniforge3/envs/flashinfer-a5/bin/python -B -m tests.matmul_bias_silu_acceptance out_of_range /mnt/fi/t2-real-final/test_production_shape_rejected0/cold-store`。

```text
{"category": "dispatch", "compiler_calls": 0, "diagnostic": "\u65e0\u53ef\u7528 kernel\uff1aop=gemm.matmul_bias_silu\uff0cbackend policy=auto\u3002\u9010\u6761\u6dd8\u6c70\u539f\u56e0\uff1a\n  \u7b2c\u4e00\u7ea7\uff1a\u9009\u62e9\u5b9e\u73b0\u6808\n    [reject] ascendc\uff1a\u6808\u4e0d\u53ef\u7528\uff1a\u6808\u5185\u5168\u90e8 1 \u4e2a\u5b9e\u73b0\u88ab\u8fc7\u6ee4\uff1bascendc/t2/matmul_bias_silu/native/op_package/Ascend950\uff1a\u7279\u5316\u53c2\u6570 tensor_contract=dispatch shape M=272 is outside declared range [16, 256], multiple_of=16 \u4e0d\u5728\u5141\u8bb8\u96c6 ['supported']\uff1bM 272 \u4e0d\u5728\u652f\u6301\u533a\u95f4 [16, 256]\uff1bM 272 \u4e0d\u5728\u652f\u6301\u533a\u95f4 [16, 256]\n  \u7b2c\u4e8c\u7ea7\uff1a\u5404\u6808\u5185\u7684\u5b9e\u73b0\uff08\u542b\u9010\u6761\u8fc7\u6ee4\u4e0e\u6392\u5e8f\u539f\u56e0\uff09\n    \u6808 ascendc\uff0c\u9996\u9009=\u65e0\n      [reject] ascendc/t2/matmul_bias_silu/native/op_package/Ascend950 (ascendc)\uff1a\u7279\u5316\u53c2\u6570 tensor_contract=dispatch shape M=272 is outside declared range [16, 256], multiple_of=16 \u4e0d\u5728\u5141\u8bb8\u96c6 ['supported']\uff1bM 272 \u4e0d\u5728\u652f\u6301\u533a\u95f4 [16, 256]\uff1bM 272 \u4e0d\u5728\u652f\u6301\u533a\u95f4 [16, 256]", "kernel_id": "ascendc/t2/matmul_bias_silu/native/op_package/Ascend950", "result": "PASS", "runner_resolved": false, "scenario": "out_of_range"}
```

原始日志：`/mnt/fi/t2-real-final.log`；包含上述真实命令与 JSON 输出。环境变量和副本生命周期由测试 fixture 保留明确设置，未替换生产登记项。

## 8. 回归结果

已执行的契约测试命令：`/root/miniforge3/envs/flashinfer-a5/bin/python -B -m pytest -p no:cacheprovider -q tests/test_ascendc_contract_v2.py tests/test_ascendc_recipe.py tests/test_ascendc_op_package.py`。
实际输出：`94 passed in 0.26s`；其中新增 50 项，原有 recipe/op-package 44 项未修改。
该批是 host 契约测试，不替代真机编译或数值证据。

```bash
cd /mnt/fi/flashinfer
/root/miniforge3/envs/flashinfer-a5/bin/python -B -m pytest -p no:cacheprovider -q --basetemp=/mnt/fi/t2-full-final
/root/miniforge3/envs/flashinfer-a5/bin/python -B -m pytest -p no:cacheprovider -q -s -m real_compile --basetemp=/mnt/fi/t2-real-final
```

全量实际输出：`10 failed, 1897 passed, 4 warnings in 141.02s (0:02:21)；全量零失败条件尚未满足`。
real_compile 实际输出：`43 passed, 1864 deselected in 135.08s (0:02:15)；0 failed、0 skipped`。
原 36 项 real_compile 的通过情况和本次新增项单列：`原 36 项 + 新增 7 项 = 43 项全部通过；原项未改断言。另 --collect-only -q -m real_compile --ignore=tests/test_matmul_bias_silu_real.py 实际输出 36/1900 tests collected (1864 deselected) in 2.56s`。
全量的 10 项失败全部来自历史登记状态断言，与本任务加入首条生产条目直接冲突；不是编译、数值或 v1 ABI 失败。当前仍严格保留这些旧断言，已请求仅对此类登记快照作最小迁移的例外；未收到授权前不修改。
候选迁移补丁位于 `/mnt/fi/t2-registry-assertion-update.patch`，仅涉及下表四个测试文件；原 79 条记录继续单独检查，同时增加第 80 条生产记录的明确断言。
`git apply --check` 通过，外部副本的四文件验证输出为 `125 passed in 0.31s`；这不是当前工作区全量通过的证据，补丁尚未应用到仓库。

| 既有测试位置 | 旧断言 | 本任务真实状态 | 失败数 |
|---|---|---|---:|
| `tests/test_jit_trigger_layer.py:274` 与无 optional dependency 子进程测试 | 全局恒为 79 条且均无 source | 全局 80 条，新增条目有显式 source | 2 |
| `tests/test_manifests.py:20` | 随包登记 manifest 解析结果为 `()` | 1 条生产 descriptor | 1 |
| `tests/test_runtime_artifact.py:318` | 所有全局条目 source/target/ABI 均为空 | 新条目包含真实执行契约 | 1 |
| `tests/test_two_level_dispatch.py:52`、`:314` | 栈只有 CANN/reference/Triton，且总数恒为 79 | 新增 ascendc 栈 1 条，全局总数 80 | 6 |

验收需同时满足总数不低于原 1828、0 failed，且原 36 个 real_compile 全部通过；skip 不作替代。未通过的全量门槛在此明示，不以 targeted test 或 real_compile 通过代替。

施工中失败与修复：初次设备执行因漏掉 CANN 系统 workspace 和 tile 写出同步失败，已修复并由最终数值场景覆盖。第一次 AOT 测试清空全部 SDK 环境导致 host tiling 返回 561002，最终改用无编译器但保留运行时平台数据的目录。施工中并行全量曾因冻结 JSON 尚未生成报 2 项失败；最终全量结果以本节最终运行记录为准。

## 9. 仍未做与未验证

- **FA 迁移未做**：v2 提供将来所需的契约维度，不能据此声称已有 FA kernel、性能或数值覆盖。
- **CANN 版本兼容比较未做**：构建环境参与身份不等于 runtime 与 build CANN 的兼容性判断。
- **多 package identity 未做**：继续沿用一个进程一个已激活 package identity，换包需新进程。
- **入图未做、未验证**：保持 NOT_CAPTURABLE；同步 eager runner 不构成 aclgraph 安全证明。
- **多卡/并发未验证**：设备可见性为 8 张；数值执行使用默认 device 0，未做 8 卡并行或多线程/多 stream 竞态压力验证。
- **性能未验证**：未给出吞吐、延迟、并行扩展、长期压力或相对 CANN/CATLASS 优势承诺。
- **非 A5 未验证**：本算子没有 A2/A3/310P 真机运行，也没有跨架构可用性声明。
- **安装后 wheel 路径未验证**：本轮验证源码树中的随包 discovery 目录；不等同于新建 wheel 安装验收。
- **torch.compile/torch.library 集成未做**：公共 eager API 已接入，未为此算子新增 fullgraph/meta/opcheck 证明。
- **其他输入族未验证**：dtype、layout、转置、广播和超出声明 shape 的调用均不在支持范围。
- **数值覆盖有限**：报告中的误差针对给定 seed 与列明形状，不宣称穷尽全部 BF16 数值或特殊值。

未执行 `git add`、commit 或 push；未修改只读参考仓。报告中的验收结论只由本轮实际输出支持，不把历史基线、合成 ELF 或 skip 写成设备执行通过。
