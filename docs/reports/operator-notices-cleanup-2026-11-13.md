# 算子目录附属文件清理

基线：`5f71231`。移除算子目录中的许可副本与手写来源 JSON，保留集中分发信息。

## 改动

- 两个算子目录的 `LICENSE` 合并为根目录
  `LICENSES/CANN-Open-Software-License-2.0.txt:1`，内容与原文件逐字节相同。
  未使用且只有许可注释的 `CANN_LICENSE.h` 同步移除。
- 删除 GMM 的 `provenance.json`。它没有执行链路消费者；当前本地文件摘要已有
  `flashinfer/ascendc/authoring.py:prepare_native()` 自动生成，无需另写或另生成一份 JSON。
- 根目录 `THIRD_PARTY_NOTICES.md:1` 集中记录原仓库、原路径、固定提交和适用许可。
  原有源码版权头保持；FlashAttention 包装入口只更新了许可文件路径注释。
- `pyproject.toml:20` 的 `tool.setuptools.license-files` 与 `MANIFEST.in:2` 统一控制分发，
  sdist/wheel 自动携带集中许可和来源说明，算子作者不再复制这些文件。
- `tests/test_tuning_release.py:test_wheel_and_sdist_include_only_release_data_and_installed_lookup_works`
  同时检查实际 sdist/wheel 内的许可原文、来源说明和重复文件清理结果。

算子、capability、ABI 和登记数量不变；全局 Descriptor 仍为 85。
四份原生 recipe 已重新生成和冻结，未复用过期元数据；生成资源仍只位于 build 目录。

## 实际验证

约定解释器返回 `Permission denied`、退出码 126。实际设备自检：

```bash
source /home/dy/tools/miniconda3/Ascend/ascend-toolkit/set_env.sh
/home/dy/tools/conda-envs/megamoe-a2/bin/python -B -c \
  'import torch,torch_npu;print(torch.npu.is_available(),torch.npu.device_count());print(torch.npu.get_device_name(4))'
# True 8
# Ascend910B3
```

生成和冻结命令未编译设备代码：

```bash
export FLASHINFER_CATLASS_ROOT=/tmp/flashinfer-catlass-ref
/usr/bin/python3 -B -m flashinfer.operators generate
/tmp/fi-author-test-env/bin/python -B -m flashinfer.aot --freeze --all
```

静态检查实际确认四份原生 recipe 的所有输入都存在、SHA256 匹配，且没有残留的
`LICENSE`、`CANN_LICENSE.h` 或 `provenance.json` 引用。算子目录中已无这些文件。
两个修改文档的代码围栏数分别保持 28 和 4。

```bash
export FLASHINFER_TEST_NPU_DEVICE=4
/tmp/fi-author-test-env/bin/python -B tools/run_regression.py \
  --profile quick --log-dir /tmp/fi-operator-notices-20261113 \
  tests/test_operator_authoring.py tests/test_author_api.py tests/test_operator_layout.py \
  tests/test_manifests.py tests/test_ascendc_framework.py \
  tests/test_flash_attention_registration.py tests/test_tuning_release.py \
  tests/test_release_checks.py
# 155 passed, 4 deselected, 0 skipped; exit_code=0
```

本次为附属文件和分发清单清理，执行了实际 sdist/wheel 构建与解包后查表测试。
全量及 `real_compile` 本轮未执行，4 个编译用例为 deselected，不能记作通过。
未新增设备数值、性能或图模式结论；Ascend950 的验证仍受当前设备限制。
