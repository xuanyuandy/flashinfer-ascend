# 仓库工具

本目录提供源码审计、回归验证、调优采集和兼容性维护工具。文件按职责命名；在仓库根目录使用 `python -B -m tools.<模块名>` 执行。工具脚本不随 wheel 分发，运行时 API 和构建入口由 `flashinfer` 包提供。

## 验证与回归

| 工具 | 职责 | 环境要求 |
| --- | --- | --- |
| [run_regression.py](run_regression.py) | 按 quick/compile/full 和 SoC 执行测试，保存日志、JUnit 与退出码 | 取决于所选测试范围 |
| [check_imports.py](check_imports.py) | 扫描包模块并报告导入错误 | 安装被扫描模块的依赖 |
| [validate_manifests.py](validate_manifests.py) | 校验兼容性台账、kernel manifest 与 Attention 能力绑定 | 已生成的源码元数据或完整包资源 |
| [check_torch_compile.py](check_torch_compile.py) | 检查 torch.compile、输出复用和 mutation；包含 graph-break 正向对照 | Torch 编译后端；CPU 或接口支持的 NPU |
| [check_ascendc_torch_library.py](check_ascendc_torch_library.py) | 验证 Ascend C probe 的 PrivateUse1/Meta、FakeTensor、内存和编译接口 | 匹配的 NPU 与产物；源码模式另需工具链 |

```bash
python -B tools/run_regression.py --profile quick
python -B -m tools.validate_manifests
python -B -m tools.check_torch_compile --device cpu
```

## 路由与图契约审计

| 工具 | 职责 | 验证范围 |
| --- | --- | --- |
| [audit_dispatch_selection.py](audit_dispatch_selection.py) | 保存并比较固定元数据样例的路由结果、拒绝信息和实现身份 | 不分配 Tensor，不执行 kernel；样例不是完整支持矩阵 |
| [routing_capability_demo.py](routing_capability_demo.py) | 在私有 registry 中验证标准字段、自定义属性及关联约束 | 元数据场景；不修改生产登记或扩展支持范围 |
| [audit_graph_plan.py](audit_graph_plan.py) | 运行图计划与资源冻结合同测试及故障注入对照 | Host 合同，不执行设备 capture/replay |
| [audit_attention_workspace.py](audit_attention_workspace.py) | 检查原生 Attention 最大 workspace 与负向控制 | 需要与测试声明匹配的设备和运行栈 |
| [audit_graph_replay.py](audit_graph_replay.py) | 校验图能力声明与测试证据，按模式执行接口探测或真机对照 | `--declarations` 仅检查声明；真机模式限定于声明中的设备与软件环境 |

```bash
python -B -m tools.audit_dispatch_selection --write build/dispatch-before.json
python -B -m tools.audit_dispatch_selection --compare build/dispatch-before.json
python -B -m tools.audit_graph_plan
python -B -m tools.audit_graph_replay --declarations
```

`torch.compile`、图计划合同和 workspace 校验分别验证不同的执行条件。设备 capture/replay 的支持范围由具体 Descriptor 及其真机证据限定，见 [图模式说明](../docs/npu_graph_extensions.md)。

## 调优与记录导出

| 工具 | 职责 | 环境要求 |
| --- | --- | --- |
| [collect_gemma_tuning.py](collect_gemma_tuning.py) | 显式测量 Gemma normalization 的 BLOCK=128/256 配置，记录数值校验与原始测量 | 真实 NPU、Triton Ascend；可指定 CPU affinity 与干扰条件 |
| [tune_flash_attention.py](tune_flash_attention.py) | 在独立 registry 中校验并测量 CATLASS AOT 与 Triton FA 候选 | 匹配支持域的真实 NPU、AOT 产物和 Triton Ascend |
| [export_tuning_data.py](export_tuning_data.py) | 将审核过的原始记录导出为受大小限制的 SoC 分组数据 | 不进行测量或编译；输出目录须为新目录 |

```bash
python -B -m tools.collect_gemma_tuning --help
python -B -m tools.tune_flash_attention --help
python -B -m tools.export_tuning_data --output-dir build/tuning-release
```

采集、数据存储与发布契约见 [离线调优](../docs/tuning.md)。没有输入记录时，导出命令生成空 manifest；该结果不构成性能测量或支持证据。

## API 兼容性维护

| 工具 | 职责 |
| --- | --- |
| [audit_operator_surface.py](audit_operator_surface.py) | 静态提取上游公开接口与测试中的调用方式 |
| [build_operator_comparison.py](build_operator_comparison.py) | 关联上游接口、本地登记和审核过的候选映射 |
| [update_parity_inventory.py](update_parity_inventory.py) | 根据固定上游提交刷新兼容性台账注解 |
| [update_nvidia_inventory.py](update_nvidia_inventory.py) | 维护 CUDA/NVIDIA 专有接口的差异说明 |
| [merge_parity.py](merge_parity.py) | 将审核后的条目或补丁合并至指定台账 |

源码分析使用 `--upstream` 指定的 FlashInfer checkout。刷新工具的默认位置为本仓相邻的 `flashinfer-ori`；固定比较基线和兼容范围见 [上游兼容性](../docs/upstream-compatibility.md)。静态匹配与台账注解分别记录接口事实和审核结论，不代替数值或设备验证。

`metadata/` 保存兼容性台账、Attention 能力与操作目录、kernel manifest schema，属于打包和运行时资源校验的输入。生成报告、测量数据和编译缓存写入 `build/` 等输出目录。

## 算子构建

原生算子通过统一 CLI 生成、冻结与构建，按 `kernel_id` 选择实现：

```bash
python -B -m flashinfer.operators generate
python -B -m flashinfer.operators check
python -B -m flashinfer.operators build --kernel-id ascendc/attention/sparse_flash_attention/native/Ascend910B --output-dir build/sfa-aot
```

多目标 AOT、原生桥和 wheel 配置见 [编译与交付](../docs/ascendc_aot.md)，数值与安装验收见 [测试说明](../tests/README.md)。
