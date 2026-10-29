# AOT 运行期发现与 JIT 回落

算子作者用 `flashinfer.jit.runtime.bind_aot_kernel` 将冻结的 `KernelDescriptor`、
`JitSpec` 和 ABI adapter 组成延迟 runner，再经现有 `KernelRegistry` 登记。
绑定不读文件、不探测设备或编译器，也不改变 descriptor 的序列化指纹。
应用继续调用已有公共 API：dispatch 按原规则选型，选中后的 runner 自动完成
发现、匹配、加载和执行。没有改写 dispatch 的 metadata 排序含义；实际执行来源
由 `runner.provenance` 的 `source`、`directory`、`spec_fingerprint` 记录。

```python
from flashinfer.jit.runtime import bind_aot_kernel

registry.register(bind_aot_kernel(
    descriptor,
    spec=frozen_spec,
    runner_factory=adapt_module_to_public_arguments,  # (module, spec) -> callable
    jit_backend_factory=prepare_compiler_backend,    # () -> JitBackend，只在 missing 时调用
))
```

这是算子登记接口。descriptor、spec 和 adapter 必须由算子实现提供，不从搜索到的
任意产物反推可信契约。AOT adapter 收到现有 `load_op_package` 的
`OpPackageModule`；JIT adapter 收到现有 `build_and_load` 的 `JitLoadedModule`。
生产 Ascend C 登记表目前仍为空；此次只用测试登记的固定 16×16 FP32 probe 证明机制。

## 发现与缓存

- 默认目录：安装位置下的 `flashinfer/data/op_packages/<spec-sha>/`。
- `FLASHINFER_AOT_PATHS` 可追加以 `os.pathsep` 分隔的搜索根或单个 release 目录。
  默认随包目录始终参与搜索，不需要配置环境变量。
- `AotPackageIndex` 只枚举一层目录。进程缓存保存目录清单及已验证 manifest 元数据，
  不写磁盘索引。根目录 inode、link count、mtime/ctime/size 改变会重新枚举；manifest
  inode、mtime/ctime/size 改变会重新验证元数据；搜索路径配置变化会选择新索引。
- `invalidate_op_package_index()` / `index.invalidate()` 可显式刷新，包括原地修复、
  特殊文件系统或保留时间戳的部署。release 应作为不可变目录原子发布。
- 索引不替代完整校验。命中的候选在 `find` 和 `load` 时均经已有
  `verify_build_directory` / `verify_op_package` 校验。第一次绑定之后直接使用已经
  加载的模块与 runner，不在每次算子调用时扫描或 hash 文件。
- 索引失效不会卸载已经绑定的模块。替换磁盘文件不会改变旧 runner 的执行来源；
  切换同一 vendor 的 package generation 必须启动新进程；首次激活集合中的不同 op type/API 可共存；激活后追加 vendor 仍需新进程。

匹配要求 target SoC、完整 spec fingerprint 一致，且 spec 中的 `kernel` 指纹等于
可信 descriptor 指纹（包含 launch/binary ABI）。同时复用 spec 和 recipe 对 descriptor
的校验。没有另建 ABI 序列化协议，也没有按算子名称分支。

真机还发现 CANN 9.2 会把含 `.objects` / `.cache` 的完整 binary 路径在第一个
点号处截断，从而寻找错误的 `.json` 文件。既有 `load_op_package` 对这些路径打开
已校验 generation 的只读目录 FD，并向 CANN 提供
`/proc/<pid>/fd/<fd>/packages/vendors/<vendor>` 别名。FD 随活动模块保留；加载失败
则关闭 FD 并恢复环境变量。文件没有复制，产物验证仍使用原目录，vendor 名称保持一致。
这个 Linux `/proc` 依赖在当前 CANN/NPU 环境已执行验证。

## 错误与回落

| 分类 | 处理 |
| --- | --- |
| missing | `AotMissingError`。只有 `ENABLED` 且登记了 JIT backend factory 时调用现有 `build_and_load`。`DISABLED`、`CACHE_ONLY` 明确拒绝，并包含 policy 和搜索诊断。 |
| corrupt | `AotCorruptError`。校验错误带路径、digest/size/manifest 等失败项；绝不回落。 |
| incompatible | 每个候选保存 `AotIncompatibleCandidate` 原因，可继续尝试其他候选。全部不兼容时 `AotMissingError` 包含候选数量和所有原因，再按 missing policy 处理。 |
| load error | `AotLoadError`。缺符号、dlopen、adapter 绑定失败和不同 identity 的激活失败均保留原始异常为 `__cause__`，绝不回落。运行中 CANN 报错保留原始异常和 traceback，也不回落。 |

未显式传 policy 时使用已有 `FLASHINFER_NPU_JIT_DISABLE=1` 配置，否则允许 JIT。
`CACHE_ONLY` 在本入口不以本地 JIT cache 替代缺失的随包 AOT 产物，仍明确报 missing。
成功绑定的 runner 可复用；后续 policy 变化不会触发重新编译或卸载。

## 范围与已知缺口

- 最初发现机制的验收只使用 probe；目前生产登记有 MatmulBiasSiLU 与 CATLASS FlashAttention。两者仍限定各自的 dtype/shape/语义域，不由发现能力扩大支持范围。
- **当前 CANN 版本兼容校验未做**。记录中的完整 spec 一致并不能证明当前 CANN 与
  构建时环境兼容；既有加载器不重新比较当前版本。
- 生产 AOT 首次解析会按可信声明匹配当前随包的同目标生产项，整组校验并激活 vendor 路径；该组不同 op type/API 的产物允许同进程共存。未随包的同组项不编译。同 vendor 另一 generation 或跨 vendor 同名注册仍明确拒绝，保留 fresh-process 提示，不改为 JIT。激活后追加 vendor 也要求新进程。整组失败恢复环境并关闭新 FD；多个冷 JIT 包须先构建，再用 `load_op_packages` 一次性准备。
- 其他设备/CANN 版本、真实 wheel 安装环境、graph capture 下的延迟绑定，以及已激活
  模块跨 fork 的复用均为**未验证**。当前真机证据与可复现命令见
  [P0-2 验收报告](p0-2-aot-runtime-validation-2026-10-25.md)。
