# P0-2 AOT 运行期验收（2026-10-25）

## 结果与环境

最终代码：全量 **1828 passed，0 failed，0 skipped**；`real_compile` **36 passed，0 skipped**。
相对用户基线新增 48 个测试，其中 8 个为 real_compile；原有 28 个 real_compile 全部通过。
分支 `infer`，命令工作目录均为 `/mnt/fi/flashinfer`，解释器固定为
`/root/miniforge3/envs/flashinfer-a5/bin/python`。

开工第一步实际执行：

```bash
ls /dev/davinci* 2>/dev/null | head
/root/miniforge3/envs/flashinfer-a5/bin/python -B -c \
  "import torch,torch_npu;print(torch.npu.is_available(), torch.npu.device_count())"
```

实际输出：

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

环境为用户提供的 Ascend950PR / CANN 9.2.0-beta.2。以上设备可见性和后续实际执行
证明本次没有把设备不可见导致的 skip 当作 pass。

## 实现位置与错误分流

| 部分 | 文件与函数 |
| --- | --- |
| discovery | `flashinfer/jit/discovery.py:118` `op_package_search_paths()`；`:161` `AotPackageIndex`；`:204` `_snapshot()`；`:279` `get_op_package_index()` |
| match | `flashinfer/jit/discovery.py:229` `AotPackageIndex.find()`：SoC、spec 和 descriptor/ABI 指纹匹配；复用 spec/recipe 验证 |
| load | `flashinfer/jit/discovery.py:99` `AotPackageMatch.load()` → 既有 `flashinfer/jit/op_package.py:182` `load_op_package()` |
| bind | `flashinfer/jit/runtime.py:152` `bind_aot_kernel()` 安装延迟 runner；`:86` `AotKernelRunner.resolve()` 绑定真实 ABI adapter |
| fallback | `flashinfer/jit/runtime.py:97` 只捕获 discovery 的 `AotMissingError`；`:105` policy 拒绝；`:118` 复用 `build_and_load()` |
| provenance | `flashinfer/jit/runtime.py:26` `RuntimeProvenance`，记录实际 AOT/JIT 来源、目录、spec 和回落原因 |

| 类别 | 分流位置与行为 |
| --- | --- |
| missing | `discovery.py:272` 抛 `AotMissingError`；`runtime.py:97` 仅此错误可回落。`DISABLED`/`CACHE_ONLY` 在 `runtime.py:105` 拒绝，保留候选及 policy 诊断。 |
| corrupt | `discovery.py:135` `_verify()` 复用共享校验器，`:147` 归为 `AotCorruptError`，不回落；`cache.py:232` 起报告 manifest 或具体文件 digest/size/缺失项。 |
| incompatible | `discovery.py:251` 起逐项记录 SoC/spec/ABI 原因，`:267` 保存拒绝记录并继续候选；全部失败由 `:272` 作为带数量/原因的 missing 处理。 |
| load error | 缺符号：`op_package.py:108` 抛 `OpPackageSymbolError`，`discovery.py:145` 映射 `AotLoadError`；dlopen/激活：`discovery.py:112`；adapter 绑定：`runtime.py:131`。均保留原始 `__cause__`，不回落。运行中的 CANN 错误由 `runtime.py:146` 原样传播。 |

索引默认读取安装布局 `flashinfer/data/op_packages/<spec-sha>`，可通过
`FLASHINFER_AOT_PATHS` 追加根目录。进程内缓存根目录清单和已验证 metadata；根目录
inode/link count/mtime/ctime/size 变化重新枚举，manifest stat 变化重新验证，环境路径
变化选择新索引。支持 `invalidate_op_package_index()` 显式失效；不写磁盘索引。
命中时复用完整校验，成功绑定后直接复用模块，不逐调用扫描目录。索引失效不卸载旧 runner。

登记 hook 不改变 descriptor 序列化指纹。公共 API 沿用已有 dispatch 和 `.runner` 调用，
无需改写 metadata 排序。算子作者仍须提供可信 spec 与 ABI adapter；生产登记不在此次任务内。
新增 discovery/runtime 中没有 `matmul_probe` 分支。详细接口说明见
[aot-runtime-discovery.md](aot-runtime-discovery.md)。

## 真机发现的既有加载路径问题

冷缓存 JIT 真实编译完成后，CANN 曾以错误 `561108` 拒绝执行。日志指出它从完整 binary
路径的第一个点号截断，实际读取 `/mnt/fi/p0-2-debug-jit/jit/.json`，而产物位于 `.objects`。
同一份字节导出到无点路径后可执行，排除了数值实现问题。

`op_package.py:212` 在既有加载器内为含点路径打开已校验 generation 的只读目录 FD，
向 CANN 提供 `/proc/<pid>/fd/<fd>/packages/vendors/<vendor>`。这仍是同一份产物，
没有复制文件、第二缓存或第二加载器；保留 vendor 名称，FD 随活动模块持有，失败时关闭
并恢复环境变量。最终测试已验证真实 JIT store `.objects` 路径的执行和 FD 保留。

## 实际执行命令与结果

以下命令实际执行于最终代码。`-s` 用于保留逐场景证据。

```bash
cd /mnt/fi/flashinfer
/root/miniforge3/envs/flashinfer-a5/bin/python -B -m pytest -p no:cacheprovider -q
/root/miniforge3/envs/flashinfer-a5/bin/python -B -m pytest -p no:cacheprovider -q -s -m real_compile
```

实际终行输出：

```text
1828 passed, 4 warnings in 97.18s (0:01:37)
36 passed, 1792 deselected in 92.32s (0:01:32)
```

stdout 分别通过 `tee` 保存在 `/mnt/fi/p0-2-full-suite.log` 和
`/mnt/fi/p0-2-real-compile-final.log`。两次均返回退出码 0，无 skipped。
全量的 4 条 warning 来自已有 torch_npu 格式提示及 Triton deprecated API。

## 五项必需场景

下面原始 JSON 均来自上面的**实际统一 real_compile 执行**；另列的 node-id 命令用于
独立复现，不冒充另一次逐条执行。每条测试自动准备并清理自己的构建/安装目录。
测试统一通过公开 `flashinfer.gemm.mm_bf16` → dispatch → 延迟 runner，且在调用前
已创建 NPU 张量并真实执行 `a+b`。FP32 16×16 descriptor/adapter 仅在隔离测试进程登记，
不是生产 GEMM 支持声明。构建复用 AscendCBackend、`_build_record` 和 `export_op_package`。

### 1. 命中，JIT DISABLED

单项复现命令：

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python -B -m pytest -p no:cacheprovider -q -s \
  "/mnt/fi/flashinfer/tests/test_aot_runtime_real.py::test_aot_hit_with_jit_disabled_default_packaged_path"
```

实际输出：

```json
{"compiler_calls": 0, "directory": "/mnt/fi/flashinfer/flashinfer/data/op_packages/9b2088b4fcbae9b5b9e5b8e53b2c50706acb9f2867edd3cd4c46755d3bdab5a4", "late_activation": true, "numerical": "float32 16x16 allclose", "policy": "disabled", "repeated_public_call": true, "result": "PASS", "scenario": "hit_disabled", "source": "aot"}
```

### 2. missing，允许 JIT 冷编译

单项复现命令：

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python -B -m pytest -p no:cacheprovider -q -s \
  "/mnt/fi/flashinfer/tests/test_aot_runtime_real.py::test_missing_allows_real_cold_jit_compile"
```

实际输出：

```json
{"compiler_calls": 1, "directory": "/tmp/pytest-of-root/pytest-171/test_missing_allows_real_cold_0/cold-runtime-store/jit/.objects/9b2088b4fcbae9b5b9e5b8e53b2c50706acb9f2867edd3cd4c46755d3bdab5a4-f24d8b113b2a4f05b5a756c26e15d5ee", "late_activation": true, "numerical": "float32 16x16 allclose", "policy": "enabled", "repeated_public_call": true, "result": "PASS", "scenario": "missing_enabled", "source": "jit"}
```

### 3. missing，禁止 JIT

单项复现命令：

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python -B -m pytest -p no:cacheprovider -q -s \
  "/mnt/fi/flashinfer/tests/test_aot_runtime_real.py::test_missing_forbidden_jit_reports_explicit_error[missing_disabled]"
```

实际输出：

```json
{"category": "missing", "compiler_calls": 0, "diagnostic": "no matching AOT op-package for kernel_id=ascendc/t13/matmul_probe/native/op_package/Ascend950, target_soc=Ascend950, spec=9b2088b4fcbae9b5b9e5b8e53b2c50706acb9f2867edd3cd4c46755d3bdab5a4; searched paths=/mnt/fi/flashinfer/flashinfer/data/op_packages; found 0 candidates; compilation policy=disabled; JIT fallback is not permitted", "fallback": false, "policy": "disabled", "result": "PASS", "scenario": "missing_disabled"}
```

### 4. corrupt，启用 JIT 仍必须拒绝

单项复现命令：

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python -B -m pytest -p no:cacheprovider -q -s \
  "/mnt/fi/flashinfer/tests/test_aot_runtime_real.py::test_corrupt_digest_is_fatal_even_with_jit_enabled"
```

实际输出：

```json
{"category": "corrupt", "compiler_calls": 0, "diagnostic": "corrupt AOT op-package /tmp/pytest-of-root/pytest-171/test_corrupt_digest_is_fatal_e0/damaged-releases/9b2088b4fcbae9b5b9e5b8e53b2c50706acb9f2867edd3cd4c46755d3bdab5a4: SchemaError: JIT cache file inventory mismatch: packages/vendors/flashinfer_t13_native/op_api/lib/libcust_opapi.so: digest mismatch (expected e8aad48194b0ebc5df8c061bdd386efc0c054204b381832a45ef0e35db0cd842, got 1564672e7cd9010bd97c3caf84cb55babca5ef7edd8d1f46eb8389b7793afd7c)", "fallback": false, "policy": "enabled", "result": "PASS", "scenario": "corrupt"}
```

### 5. incompatible，真实 ascend910b1 包被拒绝

单项复现命令：

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python -B -m pytest -p no:cacheprovider -q -s \
  "/mnt/fi/flashinfer/tests/test_aot_runtime_real.py::test_incompatible_soc_has_candidate_reason"
```

实际输出：

```json
{"category": "missing", "compiler_calls": 0, "diagnostic": "no matching AOT op-package for kernel_id=ascendc/t13/matmul_probe/native/op_package/Ascend950, target_soc=Ascend950, spec=9b2088b4fcbae9b5b9e5b8e53b2c50706acb9f2867edd3cd4c46755d3bdab5a4; searched paths=/mnt/fi/flashinfer/flashinfer/data/op_packages, /tmp/pytest-of-root/pytest-171/test_incompatible_soc_has_cand0/other-soc/release; found 1 candidate(s), all incompatible: /tmp/pytest-of-root/pytest-171/test_incompatible_soc_has_cand0/other-soc/release/d4ec4b8baf39e71672a42e7584d1625a3ec6557c205055f8203ee7eaff3f79af: target_soc mismatch: package=ascend910b1, requested=Ascend950, spec fingerprint mismatch: package=d4ec4b8baf39e71672a42e7584d1625a3ec6557c205055f8203ee7eaff3f79af, requested=9b2088b4fcbae9b5b9e5b8e53b2c50706acb9f2867edd3cd4c46755d3bdab5a4, descriptor/ABI fingerprint mismatch: package=ee9234c22db0d3b8ecc470e6b92aefff0a2de8a93c0ac94ffd704f1ea86dc5f2, requested=19000d66b956e6a0adad11e7700ff4d375388a3e4866890a874cb2dd0ccad942, descriptor/ABI incompatible: JIT compilation unit does not bind kernel source and target_soc; compilation policy=disabled; JIT fallback is not permitted", "fallback": false, "policy": "disabled", "result": "PASS", "scenario": "incompatible"}
```

## 补充真机场景

以下三项同样包含在最终 36 项 real_compile 中：CACHE_ONLY 禁止回落；真实编译的 ELF
移除 ACLNN 符号、重新生成有效 inventory 后按 load error 拒绝；第一个包数值执行通过后，
第二个有效但不同 record identity 的包被拒绝。三者均断言没有编译和回落。

```json
{"category": "missing", "compiler_calls": 0, "diagnostic": "no matching AOT op-package for kernel_id=ascendc/t13/matmul_probe/native/op_package/Ascend950, target_soc=Ascend950, spec=9b2088b4fcbae9b5b9e5b8e53b2c50706acb9f2867edd3cd4c46755d3bdab5a4; searched paths=/mnt/fi/flashinfer/flashinfer/data/op_packages; found 0 candidates; compilation policy=cache_only; JIT fallback is not permitted", "fallback": false, "policy": "cache_only", "result": "PASS", "scenario": "missing_cache_only"}
```

```json
{"category": "load error", "compiler_calls": 0, "diagnostic": "AOT op-package load error at /tmp/pytest-of-root/pytest-171/test_missing_aclnn_symbol_is_l0/symbol-fixture/jit/.objects/9b2088b4fcbae9b5b9e5b8e53b2c50706acb9f2867edd3cd4c46755d3bdab5a4-ae8526761ee340ff812b8c71cda6c1f6: OpPackageSymbolError: op-package host ELF has no ACLNN symbol: aclnnFiMatmulProbeNative", "fallback": false, "policy": "enabled", "result": "PASS", "scenario": "load_error"}
```

```json
{"category": "load error", "compiler_calls": 0, "diagnostic": "AOT op-package load error at /tmp/pytest-of-root/pytest-171/test_second_package_identity_r0/second-identity/jit/.objects/9b2088b4fcbae9b5b9e5b8e53b2c50706acb9f2867edd3cd4c46755d3bdab5a4-9274ea43c7654b929c377081c76b810f: SchemaError: CANN op-package registration is process-scoped; load another package in a fresh process", "fallback": false, "first_numerical": true, "first_source": "aot", "policy": "enabled", "result": "PASS", "scenario": "identity"}
```

## 未做与未验证

- **生产算子登记未做**：`flashinfer/data/kernels/registry.json` 的 `kernels` 仍为 `[]`。
  机制通过 probe 验证，不宣称生产算子已支持。
- **CANN 版本兼容校验未做**：加载器不重新比较当前 CANN 与构建时环境；代码注释已明确。
  跨 CANN 版本运行**未验证**。
- **多 package identity 支持未做**：沿用单进程唯一 identity 限制，第二 identity 真机拒绝
  已验证；支持多个 identity 并存**未实现**。
- 真正重新构建并 pip install wheel 后的运行**未验证**；本次验证的是相同默认安装布局，
  无搜索环境变量的 AOT 命中。其他物理 SoC 的执行、graph capture 内首次延迟绑定、
  已激活模块跨 fork 复用均**未验证**。
- 真实 CANN 错误 561108 的原样传播已在修复前观测，修复后冷 JIT 数值通过；dlopen
  失败的原始异常/环境恢复/FD 清理由单元测试覆盖，独立真实 dlopen 依赖缺失场景**未验证**。

## 工作区边界

未修改用户已有三份 ABI 重冻测试和 `docs/reports/` 下五份报告，未操作 `.claude/`，
未修改 `/mnt/fi/flashinfer-ori`，未执行 git add/commit/push。
已有三个测试仍显示用户原始 diff，不属于本次实现。
本次修改既有 `jit/cache.py`、`jit/op_package.py`（共享验证/加载修复）以及
`jit/torch_library.py` 的加载时机说明；新增 discovery/runtime、测试及这两份文档。
`git diff --check` 通过；测试临时随包目录在测试结束后移除。
