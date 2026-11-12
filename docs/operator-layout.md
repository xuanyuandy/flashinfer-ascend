# 算子定义、后端实现与源码组织

本仓是多后端算子适配框架。一个逻辑算子可以登记多个实现，公共接口、实现能力、编译方式和执行适配分别建模。Ascend C 原生源码保留在 `csrc`，Triton 的 Python 设备实现位于 Python 后端包内。

## 目录与唯一来源

```text
flashinfer/
├── gemm.py                                  # 公共接口门面
├── operators/
│   ├── definition.py                        # 公共定义协议
│   ├── implementation.py                    # 实现声明协议
│   ├── catalog.py                           # 已安装接口索引
│   ├── authoring.py                         # 检查及生成
│   ├── runtime.py                           # 公共执行编排
│   └── gemm/<op>/
│       ├── operator.py                      # 用户：公共语义及接口定义
│       └── cases.py                         # 用户：需要时提供数值样例
├── ascendc/
│   ├── __init__.py                          # 后端登记入口及兼容导出
│   ├── _catalog.py                          # 冻结声明读取及验证
│   ├── _runtime.py                          # 共有 AOT/JIT 绑定
│   └── gemm/<op>/
│       ├── implementation.py                # 用户：能力、构建引用、runner 绑定
│       └── runner.py                        # 用户：需要时编写原生调用适配
├── triton/gemm/<op>/
│   ├── implementation.py                    # 用户：能力及 JIT 配置
│   ├── runner.py                            # 用户：参数及启动配置
│   └── kernels.py                           # 用户：Python 设备实现
├── cann/gemm/<op>/
│   ├── implementation.py                    # 用户：能力、依赖及 runner 绑定
│   └── runner.py                            # 用户：调用已有原生实现
├── reference/<domain>/<op>/                 # 参考后端采用同一实现协议
├── runtime/
│   ├── registry.py                          # 所有后端共用的登记表
│   └── runner.py                            # run_selected 执行交接
└── data/                                    # 仅在构建输出/安装包：交付索引与冻结声明
csrc/
├── kernels/<domain>/<op>/
│   ├── CMakeLists.txt                       # 用户：可选编译配置
│   ├── op_host/                             # 用户：OpDef、推形和 tiling
│   ├── op_kernel/                           # 用户：原生设备源码及头文件
│   └── include/                             # 用户：需要时提供共享声明
├── cmake/                                   # 框架共有构建模板
└── aclnn_torch_adapter/                      # 框架共有执行桥
```

历史逻辑 op 和 kernel_id 保持冻结身份，`source_dir` 可以保留移入实现的原始算子目录名，映射由声明明确指定。

`<domain>` 对应接口域，例如 `gemm`、`norm`、`rope`、`attention`、`sampling`。每个实现声明只有一个权威来源；生成的 JSON、索引和兼容导出均不作为第二份人工登记表。同名算子目录代表同一逻辑 op 的不同实现，kernel_id 仍各自唯一。

共有数值原语、布局工具和编译工具可放在域内 `_shared.py` 等共有模块；算子专属执行逻辑归其实现目录。小型 runner 可以直接是函数，多个 kernel 可以复用一个 runner 类，不要求每个 kernel 都复制加载器。兼容模块只转发既有导入路径。

## 定义、登记和执行

公共 `operator.py` 定义逻辑 op、公共调用约定、计算语义、输出推导及语义校验。它不选择具体 kernel，不加载某个后端的冻结 recipe，也不要求其他实现遵守 Ascend C 的原生 Tensor 顺序。已有的状态化 plan/run 接口可复用其域内前端与状态协议，不强行转换成无状态函数。

现有富接口通过 `OperatorDefinition.public_entry` 引用域内前端，沿用原有签名及公共元数据检查；这类接口的语义代码仍可由前端及共有模块实现。新增无状态接口可以直接声明 `schema`、`infer`、`validate_values`，由框架生成公共 callable 及 Torch 接线。两者使用同一个实现目录与登记表。

后端 `implementation.py` 声明实现身份、Capability、编译方式以及 runner 或 runner 工厂。Capability 归具体实现：dtype、shape、物理布局、SoC、量化绑定、图能力可以不同，公共定义不复制某一实现的完整限制。未执行 capture/replay 验证的实现继续声明 `NOT_CAPTURABLE`。

初始化时，框架收集实现，将静态 Descriptor 与惰性 runner 绑定，再加入已有 `KernelRegistry` 并冻结。登记不触发设备初始化、编译或设备源码导入。静态清单不序列化 Python callable；安装后的框架负责恢复 runner 绑定。

原生 recipe 中的联立 shape、dtype 和属性约束通过该候选的 `admission` 绑定检查；检查只读取签名元数据，并在执行边界复核。Python callable 不进入静态 ABI 序列化，recipe 和适配源码仍由冻结身份及随包摘要绑定。

调用时，从公共门面进入公共语义检查、输出推导与输入签名构建。dispatcher 根据输入、硬件和各实现 Capability 选择 kernel；框架准备公共输出，再通过 `run_selected` 调用所选 `descriptor.runner`。后端特有的参数重排、原生 dtype 视图、workspace 和启动适配在已选实现内完成。

Ascend C runner 复用既有 AOT 验证加载、允许策略下的 JIT、Torch/ACLNN 桥。Triton runner 复用 JIT、缓存及编译产物验证。CANN 原生 runner 直接调用已有运行时接口，只有使用本仓自定义原生源码时才需要额外构建输入。运行错误保留原异常，不解释为产物缺失或偷偷换后端。

## 接入与交付

算子作者维护公共定义、实际实现、必要的编译配置及调用适配。实现声明对原生源码目录和自定义 CMake 的引用只写一次。框架收集完整构建输入、计算摘要、生成 recipe 和实现索引、绑定 Descriptor、导出接口，并提供通用验证入口。

源码目录、已安装源码资源目录、生成源码目录和编译产物目录分别管理。Ascend C 的 host、device、tiling 可以保留原实现内部结构；同一目录用于 AOT 和显式允许的 JIT。wheel 必须携带运行时声明与 Python 适配；需要源码交付时，由打包规则把原生源码作为资源收集。生成与编译不得向只读安装目录写入缓存。

迁移现有实现时保留公共签名、数学语义、kernel_id、能力声明、ABI 和图能力证据。文件路径或源码变化后重新计算摘要并冻结相关 spec；不能继续使用旧缓存身份。精确登记集合继续由独立测试维护，生成工具不修改其期望值。

验证包括所有现有实现的登记覆盖、重构前后能力元数据比较、接口与 runner 绑定、跨后端分派、数值行为及相关 AOT/JIT 用例。跨层重构执行一次 full；缺少硬件、依赖或产物导致的 skip 必须单独报告，不计为通过。

## 对照上游

flashinfer-ori: `include/flashinfer/` 保存可独立于框架的 CUDA 实现，`csrc/` 保存原生实现、启动包装和绑定，`flashinfer/triton/` 及域内 Python kernel 包保存 Python DSL 实现，`flashinfer/jit/` 组织编译与加载。flashinfer-ori: `build_backend.py` 将原生源码作为安装资源收集。这里沿用这种语言与交付分层，并用显式实现声明和统一 KernelRegistry 关联多个后端。


## 作者接口与资源位置的实施更新

原先的目录已按 [作者声明与资源交付](authoring-and-resources.md) 实施：全部既有后端声明采用类型化作者接口，
生成 JSON 只位于 `build/generated/` 与安装包；非派生契约源位于 `tools/metadata/`。
Attention 编译协议已迁入 `flashinfer/attention/jit/`，历史导入保留薄兼容入口。
上文关于“尚未迁移”的文字记录整改前状态；实际测试证据以本次整改报告为准。
