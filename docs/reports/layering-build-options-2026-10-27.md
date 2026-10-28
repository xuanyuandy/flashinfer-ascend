# 分层、契约版本与编译配置核查（2026-10-27）

本文记录首次文档/编译选项核查的时点状态。后续按用户要求实现的 Torch C++/ACLNN 适配与新一轮回归另见 [接入验收记录](aclnn-integration-2026-10-28.md)；下文“本轮只有文档”等描述保留为该阶段事实。

## 基线、提交与范围

目录归一化已先提交并推送到 `origin/infer`：`117db20e74c4c15481c3490186e0a73590845545`。提交前已核对改动与验证快照一致，远端 ref 返回相同 SHA。

后续核查基于该提交。本轮变更只有文档，没有修改运行时代码、测试断言、recipe、ABI、冻结 spec、registry 或公共编译驱动。原有设计文档增量扩展并保留 CRLF；新增分层 Mermaid 图和独立编译选项指南，未重写历史报告。

引用本地源码基线：

- flashinfer-ori: `62b5ae4e52e14d841e1b0455dc3cfa7053181741`。
- vLLM-Ascend：`96a69d793457917d5d8fdb6b3b4e18f6f372297f`。

所引用的 11 个上游/竞品实现文件均核对为与其 HEAD 字节一致，摘要保存在 `reference-files.json`。

设备自检在用户 `dy_test` 中、加载 `../../tools/miniconda3/Ascend/ascend-toolkit/set_env.sh` 后执行，实际为 `True 8`、退出码 0。构建环境为 Python 3.12、CANN 9.2.0、Ascend950；CATLASS checkout revision 为 `41bf90da655bba3c66d0acd7e00abe33960ecfd6`。

## 分层与契约结论

1. wrapper 承担输入/能力校验、选型、Tensor/out 组织及适用接口的 plan 资源契约；不只是保存地址。generator 形成源码、宏、模板参数和编译身份，运行期 shape 可以由 host tiling 处理，无需每次重新实例化模板。
2. typed export 是编译时将具名、带类型签名的 host 函数暴露到加载模块的边界；运行时调用这个函数后才可能提交 device kernel，也可以只执行 plan。flashinfer-ori: `csrc/single_prefill_jit_binding.cu` 导出 `run`，`csrc/batch_attention_jit_binding.cu` 还导出 `plan`。本仓对应 CANN ACLNN 导出 API 与 Python Tensor 适配器，尚非同一套通用 export 生成器。
3. `BuildRecipe.schema_version=1/2` 是本仓的兼容序列化与校验规则，不是 Ascend C 本体要求的两个接入档位。所核对上游用 generator、模板与 FFI 组织编译单元；其个别 JIT 元数据也有版本号，不能与本仓 recipe profile 等同。新生产算子按完整 v2 接入，v1 保持旧探针/旧身份兼容，未删除或放宽校验。
4. FA 与 MatmulBiasSiLU 已有 shape 驱动的 `host_dynamic` callback。FA 按 S/T/heads 计算 tiling 和 block 数，并声明动态 workspace；其有限 dense envelope 不表示支持读取 length Tensor 内容的变长 tiling、任意 scalar/attribute ABI 或任意 batch FA。本轮没有扩大支持域。

关键修改位置：`docs/framework_design.md` §2.4、§3.5、§5.6、§5.7，`docs/adding-a-custom-operator.md` §1.1，以及新增 `docs/ascendc-compile-options.md`。编译选项指南是第三项问题的主要交付，包含配置入口、当前生产矩阵、驱动/SDK 边界、竞品接入映射及新算子工作流。

同时修正了 §5.6 及附录 A 的过时表述：普通 wheel 构建已默认编译生产 AOT，CONFIG 已能按 recipe 选择生产算子；隔离安装与禁编译自检也已有证据。原先“默认仅源码、CONFIG 固定探针、只测过 pure wheel”的文字不再代表当前实现，修订没有改变这些既有行为。

## 编译参数的实际记录

证据目录为 `../../var/flashinfer/layering-build-20261027-qR9qHUvk/`，日志名相对于该目录。没有使用已有缓存作为新参数生效证据；实际执行：

```bash
strace -f -qq -e trace=execve -s 65535 -o "$FI_REVIEW_DIR/execve.log" \
  python3 -B -m flashinfer.aot --all \
  --cache-dir "$FI_REVIEW_DIR/cold-cache" --output-dir "$FI_REVIEW_DIR/release" \
  > "$FI_REVIEW_DIR/cold-build.log" 2>&1
FI_RC=$?
printf '%s\n' "$FI_RC" > "$FI_REVIEW_DIR/cold-build.exitcode"
```

实际退出码 0，两条均 `disk_cache_hit=false`，汇总：

```text
{"summary": {"built": 2, "failed": 0, "requested": 2, "skipped": 0}}
```

原始 argv 为 `execve.log`，解析结果为 `compiler-argv.json`。每个生产算子记录 5 次 bisheng 调用：cube/vec 预检查各一次、cube/vec 主核编译各一次，以及一次 tiling-key 结构预处理。它们不是五次相同的 kernel 编译。

`build-options-audit.json` 自动核对了当前 JSON、冻结 spec 和主核 argv：

| 项 | 两条生产算子的实际值 |
| --- | --- |
| recipe 优化参数 | `-O2` |
| spec 额外编译选项 / recipe 链接选项 | 空 / 空 |
| 主核 argv 顺序 | SDK 的 `-O3` 在前、recipe 的 `-O2` 在后 |
| 实际架构 | `dav-c310-cube` 与 `dav-c310-vec` |
| 同步参数 | `--cce-auto-sync`；未见本仓显式配置 `--cce-auto-sync=off` |
| 主核编译次数 | 每个算子 2 次，分别为 cube / vec |
| CATLASS 宏 | 仅 FA 的 recipe 声明 `CATLASS_ARCH=3510` |

host 库使用独立 target，实际参数含 `-std=gnu++17 -O2 -fPIC -fvisibility=hidden` 及 SDK 的 ABI/安全选项；recipe 的设备参数不直接控制所有 host 编译。编译驱动还清除外部 CXXFLAGS 等隐式配置；修改选项必须进入 recipe/spec 身份并走审核、重新冻结流程。

竞品配置只做源码核对，没有在本轮编译 vLLM-Ascend。主要依据包括其 `csrc/build_aclnn.sh` 的按 SoC 选择列表、`csrc/cmake/func.cmake` 的 `add_ops_compile_options`、RmsNormCast/AttnResFwd 的算子 CMake，以及 Torch schema/PrivateUse1/Meta 的 binding。

尤其区分了 RmsNormCast 的 A2/A3 列表与 AttnResFwd 的 A5 分支：A5 特有 LLVM 参数没有复制为本仓公共默认值。竞品的 Meta 实现也没有被当成图 capture/replay 的通过证据。

## 文档校验与已有回归

新图使用 Mermaid 11.4.1 parser，结果为 `diagramType=flowchart-v2`、退出码 0，见 `mermaid-validation.json` 和 `.log`。首次 Node 检查器因未提供 DOM 的 `Element` 全局失败，修正检查器环境后通过；初始日志另存为 `mermaid-validation-initial.*`，没有改写或绕过图语法规则。

`documentation-review.json` 确认全部变更为 Markdown，生产配置与驱动和基线字节一致，本地文件链接有效。`framework_design.md` 的围栏数量由 60 增至 62，新增一张图；新增算子指南仍为 8。空白检查使用原 CRLF 格式对应的 `git -c core.whitespace=cr-at-eol diff --check`，通过。

本轮仅修改 Markdown，未重复执行全量和 real_compile。源码状态与已提交的 `117db20` 相同，上一轮实际回归如下；这些是该提交的已有证据，**不是本轮新跑的数字**：

| 验证 | passed | failed | errors | skipped | deselected | 退出码 |
| --- | --- | --- | --- | --- | --- | --- |
| 全量（目录归一化验证） | 1978 | 0 | 0 | 0 | 0 | 0 |
| real_compile（目录归一化验证） | 80 | 0 | 0 | 0 | 1898 | 0 |
| 本轮独立冷构建 | 2 built | 0 | — | 0 | — | 0 |
| 本轮新图解析 | 1 | 0 | — | 0 | — | 0 |

前两项完整证据见 [目录归一化报告](kernel-layout-2026-10-27.md)。本轮仅对新产物完成编译与包校验，没有将它们记成新的数值验收或安装后运行证据。

完整原始记录和本轮文档改动归档于 `../../var/flashinfer/layering-build-20261027-qR9qHUvk-evidence.tar.gz`，同名 `.sha256` 保存校验摘要；Mermaid 检查工具的 node_modules 不进入归档或 Git。

## 未验证事项

- 其他 CANN 版本/SoC 的具体 argv、修改优化级别或同步选项后的数值、性能与图模式。
- vLLM-Ascend、FlashInfer CUDA 参考路径在本机的编译和运行。
- 新导出的两份 trace 构建产物的独立数值执行；已有同源码运行证据见上一轮报告。
- 任意 scalar/attribute、optional Tensor/Tensor list 的通用接入、数据内容驱动 tiling、新增生产算子、生产 capture/replay；本轮未实现或提升对应声明。
- Mermaid 在各 Markdown 托管平台的显示效果；本轮验证了图语法，没有进行浏览器截图验收。
