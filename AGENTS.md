# AGENTS.md —— 本仓协作约定

面向在本仓执行任务的编码代理。**每次开工先读完本文。**
这些规则来自实际踩过的坑，不是风格偏好。

## 1. 环境（弄错会把"依赖缺失"误判成"功能缺失"）

| 项 | 值 |
|---|---|
| 解释器 | **`/root/miniforge3/envs/flashinfer-a5/bin/python`** |
| 仓库路径 | `/mnt/fi/flashinfer`（`/home/dy/projects` 的 bind mount） |
| CANN | `$ASCEND_HOME_PATH` = `/usr/local/Ascend/cann-9.2.0-beta.2` |
| catlass | `/mnt/fi/catlass-ref`（只读），需要时 `export FLASHINFER_CATLASS_ROOT=/mnt/fi/catlass-ref` |
| 设备 | 8× Ascend950PR |

- 默认 `python3` 与 `/usr/local/python3.12.14` **都没有 torch**。
- bind mount **重启后失效**：`mountpoint -q /mnt/fi || mount --bind /home/dy/projects /mnt/fi`。
- **开工第一步做设备自检**，并把结果写进报告：

  ```bash
  /root/miniforge3/envs/flashinfer-a5/bin/python -B -c \
    "import torch,torch_npu;print(torch.npu.is_available(), torch.npu.device_count())"
  ```

  期望 `True 8`。**看不到设备必须明说，不得把 skip 当 pass。**

## 2. 测试

```bash
P=/root/miniforge3/envs/flashinfer-a5/bin/python
export FLASHINFER_CATLASS_ROOT=/mnt/fi/catlass-ref
$P -B -m pytest -p no:cacheprovider -q                     # 基线 1978 passed / 0 failed
$P -B -m pytest -p no:cacheprovider -q -m real_compile     # 基线 80 passed / 0 skipped
```

- **`pytest-timeout` 未安装**，传 `--timeout` 会报 unrecognized arguments。
- 日常迭代默认执行 `python3 -B tools/run_regression.py --profile quick`：运行全部非 `real_compile` 用例，编译用例记为 deselected，不把它称为完整回归。此档仍可能使用 NPU，并非 CPU-only。
- 修改编译、缓存、ACLNN 桥或设备源码时，补充受影响的 `--profile compile` 用例；发布或跨层重构执行一次 `--profile full`。**full 已包含 real_compile，不要紧接着重复运行相同编译矩阵。** 只有新增改动、失败定位或专项验证需要才重复。
- 该入口保存完整日志、JUnit、pytest 原始退出码和最终状态；有 skip 或没有执行用例时明确失败。只改文档或清单生成工具时，执行相应静态检查及受影响 UT，无需重复冷构建和 wheel 安装。
- **`0 skipped` 比 `passed` 更重要**：真机用例在没有设备时会静默跳过，此时 `passed` 仍是绿的。
- 装后验证用 `python -m flashinfer selftest`（不依赖源码树）。

## 3. 诚实红线（违反即判任务失败）

- **不得声称性能优势。** 本仓从未做过同口径性能测量。
- **不得把"有接口"写成"已验证"**，不得把"能编译"写成"已支持"。
- **不得虚报 `graph_capability`。** 没做过 capture/replay 验证就写 `NOT_CAPTURABLE`。
  类名带 `CUDAGraph`、参数带 `use_cuda_graph` 都不构成证据。
- **SKIP 不计入成功**，且必须给出原因（无设备 / 产物未随包 / 依赖缺失）。
- **依赖缺失要显式失败**，不要静默跳过。
- 报告里**未验证的点要写"未验证"**，不要用通用先验补空。

## 4. 遇到这些情况必须停下来问，不要自行决定

- **既有测试断言与任务目标冲突**。例如新增生产算子会撞到"登记表恰好 N 条"的断言——
  那是**设计如此**（断言钉死精确集合，新增 kernel 必须被显式确认）。
  备好补丁、说明影响，**停下来等授权**，不要顺手改。
- 需要改写 ABI 冻结常量、放宽校验、或删除护栏。
- 需要 force push 或改写已推送的历史。

更新这类断言时**保持原有强度**：继续钉死精确集合，不得放宽成 `>= N` 或"包含"。

## 5. 改文档的规矩

- **改文档 = 增量扩展，不是重写。** 成功标准是 **diff 小**：
  `deletions` 应是几十行量级。发现自己在重写一整节，就是做错了。
- **代码块一个都不许少**（可新增）。交付前比对数量。
- 搬动章节时：抹掉 `§x.y` 引用后做非标题行排序比对，**差异应为 0**。
- `docs/framework_design.md` 面向外部专家：
  **不要内部任务号**（T9/T18B/P0-2 这类），**不要 `[V]/[P]/[G]` 状态标记**。
  状态信息融进行文或放附录。链接路径里的文件名可保留，但可见文字须是描述性的。
- 文档里写**仓库相对路径**，不要出现 `/mnt/fi/` 前缀。
- 引用上游写成 `flashinfer-ori: path`。

## 6. 已知坑

| 现象 | 原因 | 处理 |
|---|---|---|
| 以为在测 wheel，其实是源码树 | 开发环境是 PEP 660 editable 安装，`.pth` 注册的 meta path finder 优先级高于路径查找，`PYTHONPATH` 压不过 | **不要用 `--system-site-packages`**。建干净 venv，装完 wheel 再写纯路径 `_deps.pth`；装时加 `--force-reinstall`。装完断言 `flashinfer.__file__` 在该 venv 下 |
| `GetWorkspaceSize returned 561002` | unset 了 `ASCEND_HOME_PATH`，CANN runtime 失效 | 禁编译用 `FLASHINFER_NPU_JIT_DISABLE=1`，不要动环境根路径 |
| 退出码测出来总是 0 | `$?` 取的是管道末端命令的 | 先重定向到文件再取 `$?`，不要经过 `| tail` |
| 构建报 catlass 头缺失 | 未设 `FLASHINFER_CATLASS_ROOT` | wheel 不自带 catlass 头树，需自备 checkout |
| 改了源码但命中旧产物 | recipe 的 `sources`/`headers` 带 SHA256 | 改任何源文件后重算摘要并**重新冻结 spec**；改 descriptor 同样要重冻（fingerprint 含能力元数据） |
| 契约校验直接拒绝 | v1 受限档：定形 tiling / AIC-only / 零 workspace / 固定三 Tensor | 用完整档（`schema_version=2`），见 `docs/adding-a-custom-operator.md` |

## 7. 先读哪些文档

| 文档 | 内容 |
|---|---|
| `docs/framework_design.md` | 架构、交付链路、实现进展（附录 A）、对标路线图（附录 B） |
| `docs/adding-a-custom-operator.md` | 新增算子的决策树、按序工作流、踩坑清单 |
| `docs/hands-on-flash-attention.md` | 编译 → 打 wheel → 安装 → selftest 全流程 |
| `docs/reports/custom-operator-onboarding.md` | 33 项交付件对照表 |
| `docs/reports/api-surface-by-category.md` | 本仓真实接口面（43 op / 9 域） |
| `docs/reports/ascend-architecture-review.md` | 架构评审与风险排序 |

**注意**：`docs/reports/` 下的报告是**时点证据**，带各自基线 commit。
其中对 `framework_design.md` 的行号引用针对写作时点，不对应当前版本。

## 8. 交付报告要有什么

1. 设备自检结果；
2. 关键改动的 `文件:行号`；
3. **实际执行过的命令与实际输出**（不是"应该可以"）；
4. 全量与 `real_compile` 的实际数字；
5. **仍未做的**与**未验证的**，逐条写明；
6. 被你推翻的既有结论（如有）。
