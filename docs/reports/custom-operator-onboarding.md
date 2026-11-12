# 自定义算子接入实录：catlass FlashAttention（T4）

> 工作目录 `/mnt/fi/flashinfer`，分支 `infer`。
> 真机 Ascend950PR，CANN 9.2.0-beta.2，2026-10-25。
> 解释器 `/root/miniforge3/envs/flashinfer-a5/bin/python`。
> catlass 只读来源 `/mnt/fi/catlass-ref`。
> 固定 revision `41bf90da655bba3c66d0acd7e00abe33960ecfd6`。

## 1. 交付口径与风险裁决

本次迁移的是 catlass `examples/49_ascend950_flash_attention_infer` 的一个明确子集。
公共入口为 `flashinfer.single_prefill_with_kv_cache_catlass`。
它复用生产登记、v2 recipe、ACLNN tensor adapter、AOT 发现和受控 JIT 回落。
这是第二条生产 Ascend C 登记；并非对所有 FlashAttention 配置的覆盖声明。

选择保留 v2 并缩小范围，不新增数据依赖 tiling 契约。
上游 `fai.cpp:127` 已明确拒绝 `isVariedLen > 0`。
因此，定长限制来自本次迁移的 catlass 例子本身，不能归因于本仓 v2 不足。
本次另把 batch 收窄到 1，消除单个稠密 Tensor 中多个请求的分段歧义。
Q/KV 的有效序列长度就是各自真实 shape 的第零维；两者可以不同。
这不是将运行期长度数据伪装成 shape 符号，也没有把长度写死为一个数。
host 每次从输入 shape 得到实际 S/T，再产生 tiling。
公共 API 不接收 `actual_seq_lengths` 张量，不推测 padding 后的有效长度。
代价是不能表达 ragged、padding 内的有效长度、变长 batch 或 paged KV。
**本次没有证明 v2 能表达由输入张量内容驱动的 tiling；该能力仍未经验证。**

原例子含 AIC/AIV 协作、QK、online softmax、PV 和输出 rescale 模板。
迁移保留这些 catlass 模板，不以 torch/CANN attention 替换实际数学计算。
构建和执行分别验证；导入成功、recipe 校验成功都不等于设备数值通过。

## 2. Envelope 口径表

| 维度 | 支持范围 | 拒绝或不承诺 |
|---|---|---|
| SoC | Ascend950，本轮设备 Ascend950PR | 其他 SoC/架构。 |
| dtype | Q/K/V/out 全部 FP16 | BF16、FP32、混合 dtype。 |
| batch | 隐式 batch=1 | rank4 batch Tensor、多个 packed 请求。 |
| layout | 连续 NHD，底层 ND | HND、非连续 stride、不同设备。 |
| Q | `(S,H,128)` | 其他 rank/head_dim。 |
| K/V | `(T,H,128)`，K/V shape 完全相同 | GQA/MQA、不同 K/V 维数。 |
| heads | H 为 1..8 整数，Q/K/V 相等 | 0、9 及以上、不同 Q/KV heads。 |
| Q 长度 | 128、256、384、512 | 非 128 对齐、空输入、超范围。 |
| KV 长度 | 128、256、384、512 | 非 128 对齐、空输入、超范围。 |
| 有效长度 | 全部稠密 Tensor 内容有效 | 长度张量、padding 有效长度、ragged。 |
| mask | 无 mask，`causal=False` | causal、custom mask、window。 |
| scale | 固定数学定义 `1/sqrt(128)` | 用户自定义 scale、soft cap。 |
| cache | 稠密 K/V | paged KV、block table。 |
| 输出 | 与 Q 同 shape/dtype，可传 `out` | LSE、量化输出、输出与输入重叠。 |
| graph | `NOT_CAPTURABLE` | aclgraph、隐式可捕获承诺。 |
| 生命周期 | eager 同步 adapter | 异步无同步性能保证、并发重入保证。 |

这里的“定长”指完整稠密序列，并非只支持一个编译期 shape。
128/256/384/512 由同一个包的 host 动态 tiling 处理。
独立 S/T 的交叉组合必须有数值证据，包括 S>T 的非 causal 情形。
范围校验在普通 dispatch 中完成，先于 lazy runner、编译和输出分配。
host OpDef/tiling 再核验 rank、dtype/format 契约和跨 Tensor shape 一致性。
设备端的防御检查不能代替 dispatch 错误；不支持输入不能静默返回空结果。

## 3. 从例子到生产路径的迁移实录

### 3.1 先读范本，保留已有基础设施

完整阅读 `csrc/recipes/matmul_bias_silu.json`、其 kernel、共享 tiling 和 host。
v2 的混合核任务类型直接使用 `aic_aiv`。
跨核同步策略直接声明 `cross_core_flags`。
host 动态 tiling 使用 `host_dynamic`、schema 2 和具名回调。
tensor ABI 使用有序 `q,k,v,out`，device 隐含追加 `workspace,tiling`。
构建输出使用 `op_package`，loader 继续使用 `load_op_package()`。
没有新写 package loader，没有修改 P0-1 ABI 冻结常量。
没有重做 T2′ 的 schema、通用 tensor adapter 或 T3 图资源接口。

### 3.2 核对上游与结构

读取固定 revision 的 `fai.cpp`、`fai_kernel.h`、`fai_kernel_utils.h`。
读取 `fai_tiling.h`、`tiling_data_def.h` 和 `gen_data.py`。
原 `fai.cpp` 同时承担参数解析、设备分配、文件 I/O、tiling 和 launch。
生产 host 只保留 OpDef、infer shape/type、tiling、workspace 查询的职责。
测试工具负责文件 I/O、golden 生成和误差统计。
原 kernel 的架构和 tile 类型必须与 Ascend950 匹配，不能只替换 SoC 字符串。
当前 QK/PV 模板采用 128 tile；本次只承诺 head_dim=128。
原 kernel 的跨核同步来自 catlass `arch/cross_core_sync.hpp` 等模板。
引用的 catlass 头作为外部依赖解析，并绑定 header tree 摘要。
本地移植文件保留上游版权许可说明及来源标记。

### 3.3 处理 actual sequence lengths

上游数据生成器写出 Q/KV 长度文件，原 host 把它们传给设备。
看到该参数不能直接推断“host tiling 必须读取设备 Tensor 内容”。
须同时检查 host 初始化、上游变长参数门禁和 kernel 消费方式。
本次采用完整稠密序列，host 用 S/T 填充上游 tiling 的序列字段。
设备使用同一 tiling 中的形状长度，不再接受独立长度 Tensor。
如此保留动态 shape，又避免遗漏 shape 与长度值不一致的语义。
未来引入数据依赖 tiling 时，需要另设版本、可读输入名单和独立校验。
还需定义数据所在设备、拷贝/同步、生命周期、错误与图模式行为。
这些工作没有因本次 FA 子集打通而自动完成。

### 3.4 设备 entry 与 CANN host

本地 translation unit 为统一的 `FI_DEVICE_ENTRY` 提供入口。
OpDef 使用 `FI_PACKAGE_OP_TYPE`，保持构建驱动的唯一命名机制。
host 配置输出类型、ND 格式、输入输出 shape 及 Ascend950 编译目标。
tiling 回调按 shape 计算分核信息，把 POD payload 交给 CANN。
blockDim 来自设备平台与合法工作划分，不能由 Python 硬编码猜测核数。
kernel 声明混合 AIC/AIV 任务，沿用模板内跨核握手。
该例子的 QK/PV 中间结果走内部存储路径，不需要新增 GM 用户 scratch。
recipe 的动态 workspace 回调查询 CANN 系统 workspace。
本机系统需求为 16 MiB；不能把它描述成随 S/T 增长的 FA scratch。
这也不构成本次对“数据相关 workspace 大小”能力的新证明。

### 3.5 发布身份和运行期绑定

`tools/build_flash_attention.py` 构建 descriptor 并冻结 recipe/spec。
冻结是发布期动作，会读取源码、依赖和工具链；导入时不会做这些工作。
生成的 registry/spec JSON 精确绑定 recipe、源摘要、目标和工具链身份。
新的生产 kernel_id 为 `ascendc/t4/flash_attention/catlass/op_package/Ascend950`。
原 `matmul_bias_silu` 记录保留，79 个 builtin descriptor 不变。
生产集合精确为两条，总 descriptor 数为 81。
只有用户明确授权的四处既有计数/集合断言随之更新。
没有把精确集合断言放宽成包含关系或数量下界。

### 3.6 公共调用链

`single_prefill_with_kv_cache_catlass` 先构造 metadata 和 specialization。
合法输入通过内部 `ascendc.single_prefill_catlass` 的普通 registry dispatch。
集成时发现 `attention.*` 域要求已有 Attention POD ABI 及 capability binding。
ACLNN 包使用另一套 ABI，因而采用独立内部域，公共函数仍采用 prefill 命名。
没有放宽原 attention 校验器，也没有伪造该域所需的 ABI/provenance。
非法 shape、dtype、layout、mask 在 dispatch 阶段得到清晰错误。
成功选中后才分配输出，并调用绑定的 `AotKernelRunner`。
AOT 命中经过既有 discovery、摘要验证、`load_op_package()`。
仅 missing 且 JIT ENABLED 时调用真实 AscendCBackend 编译。
损坏、加载失败及 package 身份冲突均不当作 missing 处理。
已验证 module 交给 `package_tensor_runner()` 进行 ACL Tensor 转换。
adapter 查询 workspace、创建 executor、获取当前 stream、launch 并同步。
输入、输出、workspace 和 ACL 描述符的存活范围由现有 adapter 管理。

### 3.7 依赖位置与 wheel 边界

`FLASHINFER_CATLASS_ROOT=/mnt/fi/catlass-ref` 指向本次只读 checkout。
构建与允许 JIT 的 miss 才解析 catlass，AOT 命中不需要 checkout。
缺失路径必须报含 `catlass dependency` 的错误，不能误报为数值失败。
recipe 固定 revision，并对解析时的实际头文件树计算摘要。
`setup.py:build_inputs()` 收集本仓 csrc/kernel build 源。
它不把 `3rdparty/catlass/include` 自动塞进 wheel。
移植的局部头文件不等于完整 catlass 头树。
用户构建/JIT 时须另供匹配 revision；预编译运行只需已验证产物与运行库。

## 4. 新增一个自定义算子的可执行交付清单

下表左列是 native v2 的完整必需项，右列是 catlass v2 对同一项的额外工作。
“复用”表示要正确接入并验证，不表示可以遗漏该文件或声明。

| 交付件 / 校验入口 | native v2：以 matmul_bias_silu 为例 | catlass v2：FA 额外需要 |
|---|---|---|
| 数学和 envelope | 定义 BF16 GEMM+bias+SiLU、M/N/K 和误差 | 选定例子、确认定长条件，列出 mask/cache/head 限制。 |
| kernel TU | `csrc/kernels/matmul_bias_silu/native.cpp` | `csrc/kernels/flash_attention/catlass.cpp` 实例化真实模板。 |
| device entry | `FI_DEVICE_ENTRY` 与 recipe `device_entry` 一致 | 验证 Arch/Tile/Policy 和 950 的混合核编译模式。 |
| 共享 tiling | `csrc/include/flashinfer_ascend/matmul_bias_silu_tiling.h` | 带入原 FA tiling POD，明确字段来源及本地适配。 |
| host TU | `csrc/op_package/matmul_bias_silu.cpp` | `csrc/op_package/flash_attention.cpp` 消除 standalone main/I/O。 |
| OpDef | Input/Output dtype、ND、InferShape、InferType | 保持 q/k/v/out 顺序，对 batch/head/layout 再核验。 |
| tiling 回调 | `FiMatmulBiasSiluTiling` | `FiFlashAttentionTiling` 连接上游分核算法。 |
| workspace 回调 | 声明动态大小，并计算系统前缀和用户 scratch | 分清 FA 用户 scratch 与 CANN 系统 workspace；勿虚报动态性。 |
| recipe | `csrc/recipes/matmul_bias_silu.json` | `csrc/recipes/flash_attention.json`，implementation=catlass。 |
| 源闭包 | sources/headers/host_sources 均填 SHA256 | 所有移植头也列入 headers，不能只摘要 TU。 |
| 外部依赖 | `dependencies=[]`，不探测 catlass | name/revision/include_dir 固定，校验 Git revision 和头树摘要。 |
| 编译配置 | target_soc、npu_arch、flags | `CATLASS_ARCH=3510` 与模板实际架构一致。 |
| v2 混合核 | `task_type=aic_aiv`、`cross_core_flags` | 审查模板中的跨核 flags 和生命周期。 |
| v2 ABI | tensor_parameters 输入在输出前 | 同样复用四 Tensor ACLNN ABI，无长度 Tensor。 |
| shape 约束 | shape_constraints 绑定 M/N/K | 绑定 S/T/H/D；不把 actual_seq_lengths 值当符号。 |
| package 身份 | vendor/op_type/host_entry/launch_abi | 独立 FA vendor/op_type，避免与已有包身份混淆。 |
| schema 校验 | `BuildRecipe.from_dict()`、`_validate_v2()` | 继续受同样校验，依赖条件另由既有 dependency 校验覆盖。 |
| resolve | `resolve_build_recipe()` | 提供 catlass root，检查缺失、revision 和内容变化。 |
| toolchain | `AscendCToolchain.discover()` | 无新的编译器发现器，继续同一工具链指纹。 |
| descriptor | CapabilitySpec、WorkspaceFormula、GraphCapability | 内部 op 为 ascendc.single_prefill_catlass，明确形状与 NOT_CAPTURABLE。 |
| 冻结 spec | `declare_kernel()` 和 `tools/build_matmul_bias_silu.py` | `tools/build_flash_attention.py` 仅增补 FA 记录。 |
| 生产登记 | `flashinfer/data/kernels/registry.json` | 第二条精确 kernel_id；builtin 计数不变。 |
| 静态声明 | `flashinfer/data/kernels/ascendc_specs.json` | recipe/spec 与新增 descriptor 同步，保留旧记录。 |
| lazy 绑定 | `flashinfer/ascendc.py:ascendc_descriptors()` | 复用同一循环，JIT 时解析 catlass override。 |
| tensor runner | `package_tensor_runner()` | 无需新写 ctypes ACL tensor ABI 或 loader。 |
| 公共 wrapper | `flashinfer.gemm.matmul_bias_silu()` | `flashinfer.attention.catlass_prefill` 和顶层导出。 |
| dispatch 门禁 | `tensor_contract_specialization()` | 加 causal/layout/mask 选项诊断，先拒绝后分配。 |
| host 测试 | recipe/metadata/dispatch/惰性行为 | catlass 来源、精确登记、非法配置和依赖缺失。 |
| real_compile | 真编译、真 launch、误差、AOT/JIT | 用上游 gen_data.py golden，覆盖独立 S/T 组合。 |
| 发布导出 | `_build_record()`、`export_op_package()` | 同样复用，另提供外部依赖的构建说明。 |
| 无编译器运行 | runtime-only CANN + JIT DISABLED | 同时使 catlass checkout 不可用，证明命中不解析源。 |
| 完整性反例 | 损坏摘要时拒绝且不回落 | 检查 compiler_calls=0，不能只断言异常字符串。 |
| 文档 | 支持范围、实际命令、风险、未验证项 | 来源差异、golden 口径、版本依赖、未证明的数据依赖 tiling。 |

实际接入顺序建议是数学/envelope→源码/host→recipe/resolve→冻结/构建。
随后接公共 dispatch→host 反例→设备数值→七个交付场景→全量回归。
先把不支持的配置拒绝清楚，再扩大范围；扩大时必须同步 host、recipe、dispatch 和测试。
修改源文件后重新计算 recipe 摘要，并重新冻结 spec；旧包不能冒用新身份。
修改 descriptor 后也要冻结 spec，因为 kernel fingerprint 包含能力与 ABI 元数据。
公共接口名字不能作为 graph safety 或跨 SoC 支持的证据。

## 5. 对标 ascend-parity-analysis §7.4 十步

历史报告保留，以下是增量更正，不重写历史测试结果。

| 原步骤 | T2′/P0-2/T4 后变化 | 仍需作者完成 |
|---|---|---|
| 1 数学/Tensor/ABI/reference | v2 已解除 AIC-only、零 workspace、固定三 Tensor 的前提 | envelope、数学语义、数据依赖边界和 golden。 |
| 2 device TU | T4 增加 Ascend950 FA 迁移实例 | Arch/Tile/Policy 选择、同步正确性、性能。 |
| 3 host 桥 | 通用 op_package 构建和 ACLNN ABI 可复用 | 每算子的 OpDef、infer、tiling、workspace。 |
| 4 recipe/配置 | v2 明确混合核、动态 tiling/workspace 和有序 Tensor | 每文件摘要、合法 shape、依赖与 entry。 |
| 5 工具链/依赖 | 固定 revision、头树摘要、显式外部 root | 安装匹配 CANN 和 catlass；wheel 不带完整头树。 |
| 6 spec/编译/加载 | 冻结 spec + lazy AOT runner 支持公共路径 | 发布身份冻结与实际构建，非登记即成功。 |
| 7 module→callable | T2′ 通用 tensor runner，T4 无需另写 FA ctypes runner | 特殊 attrs、非 Tensor ABI、数据驱动 tiling 不能外推。 |
| 8 public dispatch/meta | 两条生产登记和 FA eager wrapper | 新算子 wrapper、options 门禁、独立 torch.library/meta 集成。 |
| 9 eager/compile/capture | T4 有 FA eager 数值及错误路径 | torch.compile/FakeTensor/fullgraph/aclgraph 各自另验。 |
| 10 AOT 交付 | P0-2 提供发现/验证/回落；T4 可导出 FA release | 通用 family CLI 枚举、平台 wheel 发布与兼容策略。 |

第 1、4、7 步显著简化：不用为这类纯 Tensor、shape 驱动的算子再造基础契约。
第 6、10 步因 P0-2 已有运行期编排而简化，不能把这部分功劳记作 T4 新加载器。
第 2、3、5、9 步的正确性负担没有消失。
第 8 步只完成本次公共 eager 路线，没有自动得到 torch.compile 或图模式。
旧“CATLASS 闭环只有固定 GEMM 探针”结论被第二条生产 FA 子集取代。
旧“任意输入数必须逐算子另写 adapter”已被 T2′ 的 tensor_parameters 路线部分取代。
旧“生产默认登记为零”已过时；当前为 native 和 catlass 各一条。
相对 T3，默认 descriptor 总数由 80 变为 81，builtin 仍为 79。
原唯一具有图能力声明的 single-decode 条目不变；新 FA 不增加可入图条目。
不能把上述更正写成“v2 已支持任意 FA 类算子”。

## 6. 仍然很重的步骤与减负建议

第一，手动维护源码摘要、descriptor 和冻结 spec 容易漏同步。
建议提供 `new-op` 脚手架，生成 TU、host、tiling header、recipe 和测试目录。
脚手架必须默认 NOT_CAPTURABLE，并把支持范围留作必填字段。
生成器只生成可审查代码，不能默认为用户宣告数值正确。

第二，host shape 检查与 Python shape_constraints 仍有重复。
可从受限 schema 生成 C++ metadata 校验片段，保留算法专属 tiling。
跨 Tensor 相等关系、上界和 multiple_of 应有同一输入来源。
数据依赖 tiling 不应混入该 shape-only 生成器。

第三，移植大模板时 CANN 日志长，首次构建失败定位费时。
建议统一保存构建命令、host/kernel 日志和 recipe/spec 摘要到 release evidence。
保留错误原文、编译退出码和目标架构，避免只给“build failed”。

第四，AOT/缺失/损坏/JIT 场景目前需要隔离进程和目录。
建议抽取测试 harness，参数化 operator、golden 生成器和非法输入工厂。
隔离进程仍有价值，因为 CANN package 身份和加载环境是进程级状态。
抽取时不能用 mock 编译代替真实 missing-enabled 编译证据。

第五，外部头文件依赖的安装体验还不够集中。
建议构建命令统一接收 dependency root map，并打印名称/revision/摘要。
缺依赖时报名称与可操作路径配置，不要要求用户从 bisheng 错误猜 include 来源。

第六，扩大 envelope 依然需要设备数值和边界测试。
D64、BF16、GQA、mask、paged KV 都不能靠多加 JSON 选项获得支持。
性能、并发、多 stream、graph 资源生命周期应各自设验收任务。

## 7. 实际验收证据

所有 pytest 命令使用 `-B -p no:cacheprovider`，未传 `--timeout`。
设备检查输出为 `/dev/davinci0` 至 `/dev/davinci7`，解释器输出 `True 8`。
日志和构建产物保存于 `/mnt/fi/t4-validation`。

### 7.1 环境和首次阻塞

实际自检命令：
```bash
ls /dev/davinci* | head
/root/miniforge3/envs/flashinfer-a5/bin/python -B -c "import torch,torch_npu;print(torch.npu.is_available(),torch.npu.device_count())"
git -C /mnt/fi/catlass-ref rev-parse HEAD
```
输出分别为设备 0..7 与 manager、`True 8`、上述固定 revision。
初次 AOT 编译成功，但 golden 生成器导入失败：`No module named 'ml_dtypes'`。
那轮为 `6 failed, 1 passed in 23.33s`，没有计作数值通过。
原生成器虽然本轮只用 FP16，仍会导入 BF16 类型，所以必须安装其真实依赖。
没有修改原生成器、mock dtype 或替换参考实现。
实际修复命令：
```bash
/root/miniforge3/envs/flashinfer-a5/bin/python -B -m pip install --no-deps ml_dtypes==0.5.3
```
输出：`Successfully installed ml_dtypes-0.5.3`；未升级既有 numpy/torch。
这属于 golden 测试依赖，不是公共 FA 运行期依赖。

### 7.2 Golden 的精确口径

直接导入固定 revision 的 `gen_data.py`，只把 `WORKSPACE` 指向测试临时目录。
调用原 `TestFlashAttentionInfer.calc_data()`，读取其 q/k/v 和 `golden.bin`。
生成器 SHA256 为 `c988b56dc2113990f2d0978185db3eda2cd18db05018b2f042ae3c2f4a6f2fb9`。
每组设置原随机种子 1，不改原参考算法。
上游输出为参考结果先舍入到 FP16，再以 FP32 格式写 golden.bin。
不能把这个文件误称为未舍入的 FP64 或 FP32 精确 attention reference。
误差判定逐元素使用 `abs(actual-golden) <= 0.001 + 0.001*abs(golden)`。

| S | T | H | D | max_abs_golden | 最大容差比 |
|---:|---:|---:|---:|---:|---:|
| 128 | 128 | 1 | 128 | 0.0001220703125 | 0.108495161 |
| 128 | 256 | 2 | 128 | 0.0001220703125 | 0.107330680 |
| 256 | 384 | 4 | 128 | 0.0001220703125 | 0.108108111 |
| 384 | 512 | 3 | 128 | 0.00006103515625 | 0.057444848 |
| 512 | 512 | 8 | 128 | 0.00006103515625 | 0.059014454 |
| 512 | 128 | 1 | 128 | 0.0001220703125 | 0.108495161 |

六组 workspace 实测都为 16777216 bytes；容差比全部小于 1。
重复公共 API 调用要求结果完全一致，并保留同一个已解析 runner provenance。

### 7.3 七个独立进程场景

测试工具使用真实 registry、公共 API、runner 和 loader，仅记录编译/发现/workspace 次数。
每个场景有独立 cold store；corrupt 在 JIT ENABLED 下验证不回落。
AOT 命中使用 runtime-only CANN 视图，保留 host tiling 所需平台数据。
其 PATH、ASCEND_HOME_PATH 下都无 bisheng，配置的 catlass build root 也不存在。
golden 的只读来源仍由测试工具读取；算子执行路径不解析构建源。
七个实际子进程命令和 JSON 原文保存在 `seven-scenarios-final.log`。
完整 pytest 驱动负责为它们准备 release、runtime-only 目录和损坏样本。
不能脱离这些前置步骤仅重放 corrupt 或 hit_disabled 的子进程命令。

实际执行的驱动命令（cwd `/mnt/fi/flashinfer`）：
```bash
FLASHINFER_CATLASS_ROOT=/mnt/fi/catlass-ref /root/miniforge3/envs/flashinfer-a5/bin/python -B -m pytest -p no:cacheprovider -q -s /mnt/fi/flashinfer/tests/test_flash_attention_real.py --basetemp=/mnt/fi/t4-validation/acceptance-final
```
实际输出：`7 passed in 51.46s`，0 failed、0 skipped。
该命令产生以下七个实际子进程调用：
```bash
/root/miniforge3/envs/flashinfer-a5/bin/python -B -m tests.flash_attention_acceptance numerical /mnt/fi/t4-validation/acceptance-final/test_flash_attention_golden_mu0/cold-store
/root/miniforge3/envs/flashinfer-a5/bin/python -B -m tests.flash_attention_acceptance hit_disabled /mnt/fi/t4-validation/acceptance-final/test_flash_attention_aot_witho0/cold-store
/root/miniforge3/envs/flashinfer-a5/bin/python -B -m tests.flash_attention_acceptance missing_enabled /mnt/fi/t4-validation/acceptance-final/test_flash_attention_missing_r0/cold-store
/root/miniforge3/envs/flashinfer-a5/bin/python -B -m tests.flash_attention_acceptance missing_disabled /mnt/fi/t4-validation/acceptance-final/test_flash_attention_missing_d0/cold-store
/root/miniforge3/envs/flashinfer-a5/bin/python -B -m tests.flash_attention_acceptance corrupt /mnt/fi/t4-validation/acceptance-final/test_flash_attention_corrupt_r0/cold-store
/root/miniforge3/envs/flashinfer-a5/bin/python -B -m tests.flash_attention_acceptance out_of_range /mnt/fi/t4-validation/acceptance-final/test_flash_attention_envelope_0/cold-store
/root/miniforge3/envs/flashinfer-a5/bin/python -B -m tests.flash_attention_acceptance missing_dependency /mnt/fi/t4-validation/acceptance-final/test_flash_attention_missing_c0/cold-store
```

以下摘录各场景实际 JSON 字段，不用预期结果替代输出：

| 场景 | 实际输出字段 | 判定 |
|---|---|---|
| numerical | result=PASS, source=aot, compiler_calls=0；六组误差见上表 | golden 通过。 |
| hit_disabled | source=aot, policy=disabled, compiler_calls=0, toolchain_calls=0 | 数值正确。 |
| hit_disabled 隔离条件 | compiler_path_visible=false, compiler_home_visible=false, catlass_root_available=false | 无编译器及构建依赖命中。 |
| missing_enabled | source=jit, policy=enabled, compiler_calls=1, toolchain_calls=1 | 真编译，max_abs=0.0001220703125。 |
| missing_disabled | category=missing, policy=disabled, fallback=false, compiler_calls=0 | 明确拒绝。 |
| corrupt | category=corrupt, policy=enabled, fallback=false, compiler_calls=0 | 摘要损坏无静默回落。 |
| out_of_range | category=dispatch, runner_resolved=false, compiler_calls=0 | 13 个反例全部拒绝。 |
| missing_dependency | category=dependency, stage=build_source_resolution, compiler_calls=0 | 构建阶段明确报 catlass 缺失。 |

missing-disabled 错误包含 `compilation policy=disabled; JIT fallback is not permitted`。
corrupt 错误包含 `libcust_opapi.so: digest mismatch`；故意破坏了该文件首字节。
超范围示例包含 `dispatch shape S=640 is outside declared range [128, 512], multiple_of=128`。
其他反例覆盖最小值、对齐、D、heads、GQA、dtype、batch rank、stride、causal、layout、mask、out alias。
依赖错误包含 `catlass dependency headers are missing` 和 `FLASHINFER_CATLASS_ROOT` 修复提示。
该错误在 bisheng 启动前发生；提前失败符合构建依赖解析的职责。

### 7.4 回归与产物

实际全量命令：
```bash
/root/miniforge3/envs/flashinfer-a5/bin/python -B -m pytest -p no:cacheprovider -q --basetemp=/mnt/fi/t4-validation/full-suite
```
实际结果：`7 failed, 1968 passed, 4 warnings in 194.51s (0:03:14)`，没有 skip。
新增 31 个 host 测试和 7 个 real_compile 场景均通过；尚未满足全量 0 failed。
七个失败来自以下三个未在 §10.1 中授权修改的既有测试区域：

| 测试位置 | 冲突 | 当前处理 |
|---|---|---|
| `tests/test_manifests.py:25–27` | 断言生产 manifest 长度为 1，并按单条解包 | 保留原断言，待授权精确两条集合并按 ID 取原条目。 |
| `tests/test_two_level_dispatch.py:56–69` | ascendc=1、总数80、生产差集只有T2 | 保留原断言，待授权2/81/精确T2+T4，builtin79不变。 |
| `tests/test_two_level_dispatch.py:329` | 总数80，产生5个参数化失败 | 保留原断言，待授权精确81，其余调度不变性检查不动。 |

依用户 §7/§10.1，发现新的既有断言冲突后已停止实现修改并请求裁决。
独立 `-m real_compile` 回归尚未执行；不能以本轮七场景结果冒充原73个标记集回归。
日志 `/mnt/fi/t4-validation/full-suite.log` 保留全部实际失败及 traceback。
本报告此时记录的是待裁决状态，不是已完成 T4 验收。

构建产物已验证导出到：
`/mnt/fi/t4-validation/release/521ce132aad2acc1c48ab036fbf6b5b252e024bfd429edac2732bb4315c89b2f`。
重新构建命令为：
```bash
/root/miniforge3/envs/flashinfer-a5/bin/python -B tools/build_flash_attention.py --catlass-root /mnt/fi/catlass-ref --output /mnt/fi/t4-validation/rebuild
```
上述为复现入口；本轮实际编译由测试 fixture 的 `build_release()` 执行。

pure wheel 另在隔离源副本构建，避免改动用户工作区 build 输出。
实际命令：`/root/miniforge3/envs/flashinfer-a5/bin/python -B /mnt/fi/t4-validation/wheel-check/verify_wheel.py`。
输出 `result=PASS, catlass_headers_bundled=false, precompiled_artifacts_bundled=false`。
11 个必要文件与源码逐字节一致，精确包含两条生产登记。
wheel SHA256：`af6b7be7493691ebb92cd74b012abb23e1d3dad8bafda0d152482bf5b849b845`。
完整命令及文件清单见 `/mnt/fi/t4-validation/wheel-check/evidence.json`。
这里只验证 wheel 内容；没有声称该 wheel 已安装验收或包含 AOT 二进制。

## 8. 未做与未验证项

- 输入张量内容驱动的 host tiling：未新增契约，未验证，也未证明 v2 能表达。
- 变长 batch、ragged、padding 有效长度、实际长度 Tensor：不属于本次范围。
- BF16、D64/其他 D、GQA/MQA、causal/custom mask、paged KV：未支持。
- 超出序列/heads envelope 的数值：只验证拒绝，不宣称可计算。
- torch.library/FakeTensor/opcheck/torch.compile/fullgraph：没有为新 FA 接入和验证。
- aclgraph/capture/replay：没有验证，descriptor 明确 NOT_CAPTURABLE。
- 多进程并发、多 stream 重入、多卡运行：设备可见不等于这些能力已测。
- 性能与 CANN/其他 FlashAttention 实现对比：未做 benchmark，不宣称加速。
- 其他 SoC、CANN 版本、非本机发布环境：未验证。
- CANN 运行库跨版本二进制兼容性：沿用既有 loader 边界，未补新保证。
- 长期运行、极端数值输入、所有合法 shape 的穷举：未验证。
- wheel 安装后端到端运行：本轮仅检查 pure wheel 内容；AOT 默认路径在源码环境验收。
- Attention POD 框架 provider/batch planner 接入：未做；公共入口走独立 ACLNN eager 路线。

保留已有 P0-1/P0-2/T2′/T3 文件和历史报告；本报告只记录本轮新增证据。
未执行 git add、commit、push，未修改只读 catlass checkout。
