# 配置与命令参考

面向集成者和开发者：本页区分运行期选择、构建期交付和诊断命令，说明哪些配置真正进入执行路径。

## 运行期配置

| 名称 | 默认或作用 | 限制 |
| --- | --- | --- |
| `FLASHINFER_NPU_BACKEND` | `auto` | 只有读取默认策略的入口消费；具体 facade 预处理见路由说明 |
| `FLASHINFER_NPU_DISPATCH_ORDER` | `external,cann,aot,jit,reference` | 必须完整且不重复；不代表实测排名 |
| `FLASHINFER_NPU_ALLOW_CUSTOM_KERNELS` | `0` | 为 1 才让 custom 参与普通 auto；显式 ID 单独检查 |
| `FLASHINFER_NPU_JIT_DISABLE` | `0` | 为 1 时禁止 miss 编译并排除普通 JIT 候选；不卸载已有绑定 |
| `FLASHINFER_NPU_DISABLE_DISPATCH_CACHE` | `0` | 为 1 时禁用决策缓存 |
| `FLASHINFER_NPU_CACHE_DIR` | `~/.cache/flashinfer` | 编译产物缓存；不是 tuning 数据库 |
| `FLASHINFER_NPU_JIT_CACHE_SIZE` | `1024` | JIT loader 创建时的进程内 LRU 容量 |
| `FLASHINFER_NPU_TUNING_DB` | 根据 XDG_CACHE_HOME 定位 SQLite | 空值关闭本地与随包调优查询；读取不创建数据库 |
| `XDG_CACHE_HOME` | `~/.cache` | 未显式指定数据库时，使用其下 flashinfer/tuning.sqlite3 |
| `FLASHINFER_AOT_PATHS` | 额外 AOT 根 | 按路径分隔符分开，仍严格匹配 spec、SoC 与 ABI |
| `FLASHINFER_ACLNN_ADAPTER_PATH` | 包内原生桥目录 | 仍校验 manifest 和匹配的 Torch 身份 |
| `FLASHINFER_NPU_TUNING_DIR` | config 可读取/显示 | 当前数据库定位函数不使用该值 |
| `FLASHINFER_NPU_ONLINE_TUNING` | config 可读取/显示 | 当前没有由此触发的在线采样路径 |

配置对象与 loader/硬件/runner 有不同生命周期，不能承诺修改环境变量后所有已有对象即时变化。跨进程修改调优数据或替换软件后按实际身份要求 reload 或重启。

## 构建期配置

| 名称 | 作用 |
| --- | --- |
| `FLASHINFER_AOT_ALL` | 默认构建本地交付分组的完整生产集合，A2/A3 共包；0 为源码模式，1 显式启用 |
| `FLASHINFER_AOT_CONFIG` | 选择生产 recipe/implementation，收集交付分组内对应的全部已声明变体 |
| `FLASHINFER_AOT_PREBUILT` | 收集并验证已有发布产物和原生桥 |
| `FLASHINFER_AOT_SOCS` | 显式选择 Ascend910B/Ascend910C（共 910B 包）或 Ascend950；也接受 a2a3/ascend950；不探测设备、不扩大能力 |
| `FLASHINFER_AOT_CACHE_DIR` | AOT 构建缓存 |
| `FLASHINFER_CATLASS_ROOT` | 固定 revision 的外部 CATLASS checkout |
| `FLASHINFER_<NAME>_ROOT` | 通用依赖根；AOT CLI 也支持 --dependency-root |
| `ASCEND_HOME_PATH` | CANN 工具链与运行栈根目录 |

ALL、CONFIG、PREBUILT 互斥。原生构建与安装要求见 [编译交付](../ascendc_aot.md)。`ASCEND_RT_VISIBLE_DEVICES` 是外部运行时可见设备设置，不是本仓 HardwareSignature 的字段；不要将它误写为旧 build_cache_key 的组成部分。

## 加载与开发环境

| 名称 | 实际读取或管理方式 |
| --- | --- |
| `ASCEND_TOOLKIT_VERSION` | 旧 toolchain_fingerprint 的版本字符串，未设置为 none；不能代替当前 JitSpec 的完整构建身份 |
| `ASCEND_CUSTOM_OPP_PATH` | op-package 加载器在首次激活前加入已验证 vendor 路径，并保留原有路径；不代替 FLASHINFER_AOT_PATHS 的发现和验证 |
| `TRITON_CACHE_DIR` | 通用 Triton 产物加载器在导入 launcher 时临时切换到该产物的缓存，随后恢复原值 |
| `CPATH` / `CPLUS_INCLUDE_PATH` / `C_INCLUDE_PATH` / `CXXFLAGS` | host 标准头探测时移除，使用选定编译器的显式标准头根，不作为算子 CMake 配置入口 |
| `FLASHINFER_TEST_NPU_DEVICE` | 测试工具选择的 NPU 索引，默认 0；不是生产路由策略 |
| `FLASHINFER_OPERATOR_TEST_RELEASE` | 通用验收复用已有 AOT 发布目录；未指定时测试 fixture 构建 |
| `FLASHINFER_OPERATOR_TEST_ADAPTER` | 通用验收复用已有原生桥目录；未指定时测试 fixture 构建 |

具体实现见 [旧缓存辅助接口](../../flashinfer/jit/cache_key.py)、[原生产物加载](../../flashinfer/jit/op_package.py)、[Triton 加载](../../flashinfer/jit/loading.py)、[标准头探测](../../flashinfer/jit/ascendc.py) 与 [测试 fixture](../../flashinfer/operators/pytest_support.py)。

## CLI

| 命令入口 | 子命令或动作 | 功能 |
| --- | --- | --- |
| `python -m flashinfer` | `show-config` | 显示 config 暴露的配置子集 |
| 同上 | `list-kernels` | 列出 Ascend C 生产声明；完整目录用 registry |
| 同上 | `selftest --kernel-id ...` | 安装包 AOT 自检，ID 可重复 |
| 同上 | `parity-report --scope all`、`parity-check` | 读取/校验版本化 API 台账，不执行目标算子 |
| 同上 | `attention-capabilities` | 查看 AttentionPlan profile 目录，不等于全局 CANN 支持表 |
| 同上 | `attention-replay`、`attention-coverage`、`attention-corpus` | Host trace/corpus 工具 |
| 同上 | `attention-accuracy`、`attention-accuracy-corpus`、`attention-protocol-validate` | Host 数值/协议资料检查，不是 NPU 后端验收 |
| `python -m flashinfer.operators` | `generate`、`check`、`build` | 类型化声明生成、检查与 AOT 构建 |
| `python -m flashinfer.aot` | `--freeze`、`--all`、`--kernel-id`、`--config` | 冻结、选择和发布原生编译单元 |
| `python -m flashinfer.aot.wheels` | 重复 `--soc`、`--output-dir` | 按 910B（A2/A3）和 950（A5）分组出包，B/C 请求去重 |
| `python -m flashinfer.aclnn.build` | `--output-dir` | 构建统一原生桥 |
| `python -m flashinfer.operators.testing` | 算子名 + `matrix/one/missing/corrupt/installed` | 通用数值及交付验收 |

只调用 `--help` 不执行编译或设备 handler。具体参数以当前模块帮助为准；开发环境变量如 `FLASHINFER_TEST_NPU_DEVICE` 属于验收工具，不作为生产路由开关。

代码来源：[config](../../flashinfer/config.py)、[CLI](../../flashinfer/cli.py)、[构建入口](../../setup.py)、[调优存储](../../flashinfer/tuning/store.py)、[AOT 发现](../../flashinfer/jit/discovery.py)。
