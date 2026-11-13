# 算子作者声明与资源交付

本仓是统一算子接口与多后端实现之间的 adapter。一个逻辑 `op` 可以关联多个
KernelDescriptor；每个 Descriptor 绑定自己的 capability 与 runner。公共接口先构造调用
语义，dispatcher 按输入、硬件、显式后端或 kernel ID 选择实现，再调用所选 runner。
实际执行错误原样传播，不能通过执行失败后换后端掩盖问题。

## 文件位置与作者职责

以下是一个算子的完整落点；同名目录表示同一公共语义，各后端计算实现彼此独立。

```text
flashinfer/
├── gemm.py                                 # 领域公共入口；通用新接口自动收集
├── operators/gemm/example/
│   ├── operator.py                         # 作者：唯一的公共 schema、语义及输出推导
│   └── cases.py                            # 作者：独立数值样例、golden 与容差
├── ascendc/gemm/example/
│   ├── implementation.py                   # 作者：AscendCImplementation、Support、ABI、构建配置
│   └── runner.py                           # 作者：逻辑参数到 native ABI 的适配，可复用标准工厂
├── cann/gemm/example/
│   ├── implementation.py                   # 作者：CANNImplementation、Support 与依赖约束
│   └── runner.py                           # 作者：调用已安装的 CANN 原生接口
├── triton/gemm/example/
│   ├── implementation.py                   # 作者：TritonImplementation、Support 与源码引用
│   ├── runner.py                           # 作者：特化参数、launch grid 与设备启动
│   └── kernels.py                          # 作者：实际 Triton kernel
└── attention/jit/                          # Attention 的计划与编译身份协议
csrc/kernels/gemm/example/
├── op_kernel/                              # 作者：Ascend C 设备计算
├── op_host/                                # 作者：OpDef、推形、tiling 与 workspace
├── include/                                # 作者：按需共享 host/device 布局
└── CMakeLists.txt                          # 可选：算子自己的编译/链接选项扩展
tools/metadata/                             # 维护者：不可派生的对标资料与外部 API 契约
build/generated/                            # 框架输出，忽略 Git，不是第二份声明源
├── csrc/recipes/*.json                      # 构建闭包、参数 ABI、依赖、目标与选项
└── flashinfer/data/
    ├── operators.json                      # 公共定义、导出、样例及源码摘要索引
    ├── implementations.json                # 后端实现身份与源码摘要索引
    └── kernels/
        ├── registry.json                   # 原生静态 Descriptor
        ├── ascendc_specs.json              # 显式冻结的 recipe/spec 与编译环境身份
        └── triton_registry.json            # 显式实验实现的元数据
```

目录发现、公共导出、Torch functional/out 与 FakeTensor 接线、候选过滤、产物发现、
完整性校验、构建锁、缓存及 wheel 组装由框架实现。新增常规算子不需要编辑这些共有入口。
数值 golden、真实原生 ABI、实际支持范围与 capture/replay 证据仍需作者负责。
独立发布清单 `tests/production_inventory.py` 由维护者审核，生成器不会改写测试预期。

算子目录不维护重复的 `LICENSE` 或 `provenance.json`。导入源码保留版权头，
上游仓库、版本与原路径在根目录 `THIRD_PARTY_NOTICES.md` 集中记录；许可原文放入
根目录 `LICENSES/` 并自动随 sdist/wheel 交付。本地文件摘要由框架的 recipe 生成，
不再复制成一份人工维护的来源 JSON；这与 runner 的运行产物 provenance 是不同信息。

## 声明字段

| 声明 | 需要作者表达的事实 | 框架负责派生的内容 |
| --- | --- | --- |
| `OperatorDefinition` | `name/domain/op`、公共 schema、输出名与 `infer`；必要时检查值 | 公共门面、Torch schema、输出分配和 out 校验 |
| `Support` | `soc/dtype/layout`、具名尺寸、特化限制；复杂组合用 `CapabilityCase` | 绑定逻辑 `op`，将维度映射到 Tensor 轴，生成完整 CapabilitySpec |
| `DimRange` | 元素数上下界与对齐倍数 | 每个相应 Tensor 轴的范围及对齐准入 |
| `CANNImplementation` | 公共定义、支持范围与 `runner` | backend、默认 ID、Descriptor 默认字段 |
| `TritonImplementation` | 公共定义、支持范围、`PythonKernel` 与 runner/工厂 | backend、JIT 标志、源码 SHA256 与 Descriptor |
| `AscendCImplementation` | 公共定义、支持范围、`NativeBinding`、`BuildConfig`；必要的 workspace 上界 | 静态 Descriptor、recipe、源码闭包、延迟 AOT runner |
| `NativeBinding` | Tensor/属性顺序、可选性、逻辑形状、合法 dtype 组合与显式存储转换 | 契约校验、标准 Tensor ABI 适配 |
| `BuildConfig` | 设备/host 源码、OpDef 名、tiling 回调与特殊构建规则 | 默认目录和入口、输入摘要；独立 CMake 配置接入共有构建流程 |

`runner=` 接受普通 callable；`runner_factory=` 接受适配工厂，二者必须且只能选择一个。
Python 工厂接收 Descriptor，Ascend C 工厂接收已验证模块、冻结 spec 和 recipe。
复杂 runner 可以保有进程内缓存和启动状态，仍不能修改已冻结的能力或构建身份。

公共 `tensor_shapes` 用 `M/N/K` 等名字关联维度，避免重复写每个输入、输出轴的
ShapeRange。底层 `CapabilityCase`、量化布局、融合顺序和图证据继续保持完整表达能力。
这里的自动化不推断硬件能力，不把已有接口或编译成功当作数值/图模式验证。
`PythonKernel` 只声明源码文件；真正的函数入口和启动参数属于 runner 或显式 launch ABI。

既有 ID 保持稳定；新 Python 实现缺省 ID 是 `<backend>/<domain>/<name>/<实现名>`，
Ascend C 的缺省 ID 还带唯一目标 SoC。源码变化由摘要和 spec 指纹区分，不自动更名。
作者不用填写 digest；`BuildConfig.dependencies` 中外部依赖的版本仍需明确钉住。

## 源码、构建与安装的资源边界

源码模式从 Python 声明收集公共定义和实现；注册只读取作者源码，不读取生成 JSON，
不探测 SDK 或设备，也不启动编译器。显式 `generate/check` 输出并校验构建资源；
显式 `build` 完成生成、冻结与 AOT 构建。所有资源位于忽略 Git 的 `build/generated/`。
选择一个 kernel 构建时仍冻结完整的原生元数据目录，编译和导出只执行所选 kernel；
冻结需要声明的外部依赖源码，未选 kernel 的二进制不会因此编译。

```bash
python -B -m flashinfer.operators generate
python -B -m flashinfer.operators check
python -B -m flashinfer.operators build --kernel-id <kernel-id> --output-dir <release-dir>
```

原生 runner 在首次需要构建契约时读取冻结 metadata，并比较当前声明与 recipe 身份；
缺失或过期时明确要求生成/冻结。注册实现不要求二进制存在；执行才解析匹配产物。
冻结 metadata 与可选二进制是不同交付件，缺包错误不能触发无授权的编译或改用其他实现。

wheel 构建重新生成索引，将资源与非派生契约复制到构建输出的 `flashinfer/data/`；
编译模式显式冻结当前工具链身份。源码发布包包含作者源码与 `tools/metadata/`，
生成目录不进入 Git。显式探针 `csrc/recipes/matmul_probe.json` 继续作为兼容测试输入。

安装模式只读取包内索引、recipe 和冻结 metadata，校验源码摘要及身份。
资源缺失或损坏会报错，不扫描相邻源码仓，也不向 site-packages 写入缓存。
通用 `jit/` 接收领域或后端生成的 spec，负责缓存、编译、产物验证和加载。
历史 `flashinfer.jit.attention` 仅保留一个导入兼容文件，转向领域内同一组类型；没有第二份实现。
