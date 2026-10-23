# T20B · 发版实测数据分发、优先级与失效规则

日期：2026-10-23。修改保留在工作区，未 commit / push。
解释器均为 `/root/miniforge3/envs/flashinfer-a5/bin/python`。
原始日志与临时构造的数据库位于 `/tmp/t20b-evidence-4VlTnD/`，没有放进仓库。

## 实现与信任选择

- 发版数据路径为 `flashinfer/tuning/data/<soc_model>/measurements.sqlite3`，
  `manifest.json` 列出分片；只在查询时读取清单，只打开当前 SoC 分片，SQL 使用 `mode=ro`。
  SoC 目录只是分区，记录里的 HardwareSignature 仍须独立通过校验。
- 沿用 20A 的 `(op, workload bucket, HardwareSignature 摘要)` key 和完整原始证据。
  对标已读本地 `flashinfer-ori/flashinfer/tuning_configs/` 的 B200/GB200 文件及其
  `autotuner.get_config_path`：上游把 GPU 型号和 CUTLASS 版本放进 Python 文件名；
  本仓有意保留 capability key，既不照搬 SKU 文件名，也不执行配置数据中的代码。
- `pyproject.toml` 显式把清单与规定路径的 SQLite 分片纳入 wheel/sdist。
  `.gitignore` 忽略本地 SQLite 和 sidecar，仅对发版分片路径开窄例外；打包模式也不收本地库。
- 新增离线 `tools/t20_export_release.py`：只消费显式给出的、已审阅的 20A JSON；
  校验原始结论、完整硬件身份和当前 registry 的候选指纹；每个 key 保留最新完整证据，
  包括无显著差异的结论。相同输入可字节复现，测试还比较了反转输入顺序后的全部输出文件。
  所有分片和清单未压缩总计最多 **8 MiB**，超预算直接失败，不截断原始样本。
- **有效本地证据 > 有效发版证据 > 确定性启发式**。本地实测更接近用户机器的 launch、
  软件与争用条件；同一静态硬件签名不代表另一台机器的测量更适用，发版身份本身不增加代表性。
  发版日期再新也不能压过有效本地记录。本地最新有效的 `winner=null` 也阻止发版冠军覆盖；
  失效本地行会被跳过，继续查较旧有效本地记录，再查发版库。
- `query_tuning_record` 仍返回 `Optional[str]`，两个参数时校验 global registry；
  新增可选 `registry=` 用于私有候选。dispatch 传自己的 registry，查询和读取都返回当前有效证据。
  explain 增加来源与失效行的拒绝原因；20A 的噪声判据没有改。
- `FLASHINFER_NPU_TUNING_DB=''` 保留“关闭全部 tuning”的旧契约。
  其他进程写库或原地替换包后仍用 `reload_tuning_records()` / 重启刷新，缓存命中不 stat、不查 SQL。

目前 **0 条真实发版记录**，只有 40 字节空清单。只有一台 A5，20A 又未发现显著差异，
无法据此声明可推广的发版冠军。所有下方有冠军的记录都是显式标记的合成拒绝/打包测试 fixture。

## 失效判断

| 情况 | 实际判断 |
|---|---|
| 源码变更 | 对全部比较候选比对当前 `descriptor_fingerprint`，其中包括已声明的 `SourceRef.digest`。 |
| 配置或产物变更 | 沿用同一个 descriptor fingerprint，覆盖已声明的 ID、编译模式、约束、ABI、op/tiling schema 及 `ArtifactRef` digest/build/size/target 等字段。 |
| capability 变更 | 另外逐个比对 `capability_fingerprint`；不依赖部分历史 descriptor 是否序列化 capability。 |
| 败者变更或候选移除 | 整条比较失效，不能因为旧冠军没变就继续采用；无显著差异的比较同样检查。 |
| 硬件不符 | stored/current 的 SoC、AI Core 数、每核 UB 字节容量全部相等且全部已知；再核对 SQL 硬件摘要与 payload。0/unknown 不作通配。 |
| 不同 schema | record/manifest 必须是整数版本 1；SQLite user_version 1 有效，兼容旧 20A 的 0，但仍逐条要求 v1 payload；其他版本拒绝。 |
| 损坏或篡改结论 | 复用 20A 原始样本/样本数/时间戳/bucket/噪声分析校验，拒绝后继续解析。 |

候选的既有 capability/target/policy/JIT/custom/reference 过滤与 T19A 两级选择结构保持原样。
记录只能使通过过滤的冠军优先，fallback 顺序仍按 T7 原排序。

## 实际命令与测试数字

修改前、最后修改后各跑同一命令（stdout/stderr 分别保存为 `pytest-before.txt` 和
`pytest-after-final.txt`）：

```bash
env PYTHONPATH=. FLASHINFER_NPU_TUNING_DB= /root/miniforge3/envs/flashinfer-a5/bin/python -m pytest -q
```

修改前原样：

```text
3 failed, 1521 passed, 4 warnings in 136.39s (0:02:16)
```

修改后原样：

```text
3 failed, 1568 passed, 4 warnings in 13.33s
```

新增 **47** 条通过。三条失败名称及 actual/expected 指纹逐项比较相同：

```text
FAILED tests/test_attention_metadata_wire.py::AttentionMetadataWireTests::test_config_directory_layout_and_canonical_blob_are_frozen
FAILED tests/test_attention_quant_launch_abi.py::AttentionQuantizedKVLaunchABITests::test_v1_is_frozen_and_v2_has_an_independent_binary_identity
FAILED tests/test_runtime_abi.py::KernelBinaryABITests::test_attention_binary_abi_matches_logical_order_mutation_and_stream
E       AssertionError: '4fe29f10a3bc5582d41bbf066246302d79c31110e30bfb4df063068c44231c98' != 'f04f7b3f0f1a8ce0550e265c4d563a6da0a53260d1baab8eacbfb56af6068352'
E       AssertionError: 'b1120f7bfd2f436baa17d7365bac8e1e8287526f87d567056a5914e5f0b3229c' != '3087d1b9867361fbb9beac7a116ca60d15964be575943e0258481656b4f721ac'
E       AssertionError: '21d0e29e9c9aca68650c7cdc0a916da3540d11dffb55306f1e1318e3721b1b8e' != 'cb5285127b5926a81fcd28d2e27997ce5fc556d6c1a4a797f5d24a0060821b7c'
```

相关定向测试的最后命令：

```bash
env PYTHONPATH=. FLASHINFER_NPU_TUNING_DB= /root/miniforge3/envs/flashinfer-a5/bin/python -m pytest -q tests/test_tuning.py tests/test_tuning_release.py tests/test_dispatch_ordering.py tests/test_two_level_dispatch.py
```

```text
........................................................................ [ 47%]
........................................................................ [ 94%]
........                                                                 [100%]
152 passed in 1.16s
```

空发布数据的复现命令及实际输出（随后 `cmp` 与仓内 manifest，退出码 0）：

```bash
env PYTHONPATH=. FLASHINFER_NPU_TUNING_DB= /root/miniforge3/envs/flashinfer-a5/bin/python tools/t20_export_release.py --output-dir /tmp/t20b-evidence-4VlTnD/empty-release-data
cmp flashinfer/tuning/data/manifest.json /tmp/t20b-evidence-4VlTnD/empty-release-data/manifest.json
```

```text
{"bytes": 40, "records": 0, "socs": []}
```

## 可证伪验证 a：库内指纹不符的记录被拒

同时跑了构造硬件的发版拒绝测试及真实 wheel/sdist 构建、安装后查询：

```bash
env PYTHONPATH=. FLASHINFER_NPU_TUNING_DB= /root/miniforge3/envs/flashinfer-a5/bin/python -m pytest -q -s tests/test_tuning_release.py::test_injected_descriptor_fingerprint_mismatch_is_rejected 'tests/test_tuning_release.py::test_release_hardware_mismatch_is_rejected[all]' tests/test_tuning_release.py::test_wheel_and_sdist_include_only_release_data_and_installed_lookup_works
```

以下为该命令完整实际输出。合成记录保留显著冠军 block256，错误指纹写进 SQLite；
查询返回 None，dispatch 回到 block128。没有把冠军改成启发式赢家来制造“看似没采用”。

```text
SYNTHETIC rejection fixture; not performance evidence
stored winner=triton/t19a/gemma/block256; selected=triton/t19a/gemma/block128
dispatch norm.gemma_fused_add_rmsnorm → triton/t19a/gemma/block128（backend=triton）
  选型依据：确定性启发式 tier 3（自研 JIT / 未声明 AOT）；tuning 记录拒绝：source=local record=synthetic-stale-descriptor；实现指纹不匹配：triton/t19a/gemma/block256 descriptor_fingerprint saved=000000000000 current=0e6aa17e5eed
  第一级：选择实现栈
    [reject] reference：栈不可用：栈内全部 1 个实现被过滤；reference/norm/gemma_fused_add_rmsnorm_v1：reference 需显式 opt-in（backend='reference' 或 allow_reference=True）
    [select] triton：栈内 2/2 个实现通过过滤；比较边界：tier=3（自研 JIT / 未声明 AOT）, priority=0, kernel_id=triton/t19a/gemma/block128；栈间 T7 排序第一
  第二级：各栈内的实现（含逐条过滤与排序原因）
    栈 reference，首选=无
      [reject] reference/norm/gemma_fused_add_rmsnorm_v1 (reference)：reference 需显式 opt-in（backend='reference' 或 allow_reference=True）
    栈 triton，首选=triton/t19a/gemma/block128（最终选中栈）
      [accept] triton/t19a/gemma/block128 (triton)：通过全部候选过滤；tier=3（自研 JIT / 未声明 AOT）, priority=0, kernel_id=triton/t19a/gemma/block128；栈内 T7 排序第一
      [accept] triton/t19a/gemma/block256 (triton)：通过全部候选过滤；tier=3（自研 JIT / 未声明 AOT）, priority=0, kernel_id=triton/t19a/gemma/block256；栈内排序落后于 triton/t19a/gemma/block128：同 tier / priority，kernel_id 字典序 triton/t19a/gemma/block128 < triton/t19a/gemma/block256
  fallback：triton/t19a/gemma/block256
.SYNTHETIC release rejection fixture; not performance evidence
stored winner=triton/t19a/gemma/block256; selected=triton/t19a/gemma/block128
dispatch norm.gemma_fused_add_rmsnorm → triton/t19a/gemma/block128（backend=triton）
  选型依据：确定性启发式 tier 3（自研 JIT / 未声明 AOT）；tuning 记录拒绝：source=release record=synthetic-wrong-release-hardware；HardwareSignature 不匹配：soc_model saved=Ascend910B current=Ascend950, aicore_count saved=16 current=32, ub_size_bytes saved=131072 current=262144
  第一级：选择实现栈
    [reject] reference：栈不可用：栈内全部 1 个实现被过滤；reference/norm/gemma_fused_add_rmsnorm_v1：reference 需显式 opt-in（backend='reference' 或 allow_reference=True）
    [select] triton：栈内 2/2 个实现通过过滤；比较边界：tier=3（自研 JIT / 未声明 AOT）, priority=0, kernel_id=triton/t19a/gemma/block128；栈间 T7 排序第一
  第二级：各栈内的实现（含逐条过滤与排序原因）
    栈 reference，首选=无
      [reject] reference/norm/gemma_fused_add_rmsnorm_v1 (reference)：reference 需显式 opt-in（backend='reference' 或 allow_reference=True）
    栈 triton，首选=triton/t19a/gemma/block128（最终选中栈）
      [accept] triton/t19a/gemma/block128 (triton)：通过全部候选过滤；tier=3（自研 JIT / 未声明 AOT）, priority=0, kernel_id=triton/t19a/gemma/block128；栈内 T7 排序第一
      [accept] triton/t19a/gemma/block256 (triton)：通过全部候选过滤；tier=3（自研 JIT / 未声明 AOT）, priority=0, kernel_id=triton/t19a/gemma/block256；栈内排序落后于 triton/t19a/gemma/block128：同 tier / priority，kernel_id 字典序 triton/t19a/gemma/block128 < triton/t19a/gemma/block256
  fallback：triton/t19a/gemma/block256
.PASS wheel/sdist include SoC release DB and manifest; exclude local DB
PASS installed wheel: release hit; no provider import, collection, or compiler probe
.
3 passed in 0.58s
```

## 可证伪验证 b：对实际探测的当前机器拒绝异硬件发版数据

额外执行了明确请求的硬件探测，然后把 `Ascend910B / 16 / 131072` 的合成记录
故意放在当前 `Ascend950` 发版目录里，SQL 硬件摘要也与错误 payload 保持一致。
实际当前硬件签名为 `Ascend950 / 56 / UB=0`；并未沿用 host fixture 的 32 核签名。

```bash
env PYTHONPATH=. /root/miniforge3/envs/flashinfer-a5/bin/python /tmp/t20b-evidence-4VlTnD/verify_actual_hardware.py
```

```text
[W1023 04:20:38.231402314 Module.cpp:235] Warning: AclrtGetDeviceInfo failed to get total global memory, error code is 507899. The possible cause is that the driver version is too old or does not match CANN. Fall back to aclrtGetMemInfo. Please upgrade the driver to the matching version. (function operator())
explicit current_hardware(): {'soc_model': 'Ascend950', 'aicore_count': 56, 'ub_size_bytes': 0}
SYNTHETIC invalid release fixture; not performance evidence
stored winner=triton/t19a/gemma/block256; selected=triton/t19a/gemma/block128
dispatch norm.gemma_fused_add_rmsnorm → triton/t19a/gemma/block128（backend=triton）
  选型依据：确定性启发式 tier 3（自研 JIT / 未声明 AOT）；tuning 记录拒绝：source=release record=synthetic-actual-hardware-mismatch；HardwareSignature 不匹配：soc_model saved=Ascend910B current=Ascend950, aicore_count saved=16 current=56, ub_size_bytes saved=131072 current=0
  第一级：选择实现栈
    [reject] reference：栈不可用：栈内全部 1 个实现被过滤；reference/norm/gemma_fused_add_rmsnorm_v1：reference 需显式 opt-in（backend='reference' 或 allow_reference=True）
    [select] triton：栈内 2/2 个实现通过过滤；比较边界：tier=3（自研 JIT / 未声明 AOT）, priority=0, kernel_id=triton/t19a/gemma/block128；栈间 T7 排序第一
  第二级：各栈内的实现（含逐条过滤与排序原因）
    栈 reference，首选=无
      [reject] reference/norm/gemma_fused_add_rmsnorm_v1 (reference)：reference 需显式 opt-in（backend='reference' 或 allow_reference=True）
    栈 triton，首选=triton/t19a/gemma/block128（最终选中栈）
      [accept] triton/t19a/gemma/block128 (triton)：通过全部候选过滤；tier=3（自研 JIT / 未声明 AOT）, priority=0, kernel_id=triton/t19a/gemma/block128；栈内 T7 排序第一
      [accept] triton/t19a/gemma/block256 (triton)：通过全部候选过滤；tier=3（自研 JIT / 未声明 AOT）, priority=0, kernel_id=triton/t19a/gemma/block256；栈内排序落后于 triton/t19a/gemma/block128：同 tier / priority，kernel_id 字典序 triton/t19a/gemma/block128 < triton/t19a/gemma/block256
  fallback：triton/t19a/gemma/block256
PASS release fixture rejected on the actual probed hardware
```

上面独立验证脚本的完整内容（在已保存的测试构造器上生成一条合成记录，不计真实 NPU 性能）：

```python
"""Explicit hardware probe + synthetic invalid release injection; no performance claim."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile

import pytest

from flashinfer.runtime import HardwareSignature, current_hardware, dispatch, explain_dispatch
from flashinfer.tuning import query_tuning_record, save_tuning_record, reload_tuning_records
from tests.test_tuning import A, B, SIG, STORE, fixture_record

hardware = current_hardware()
print('explicit current_hardware():', asdict(hardware))
root = Path(tempfile.mkdtemp(prefix='actual-hardware-', dir='/tmp/t20b-evidence-4VlTnD')) / 'release'
with pytest.MonkeyPatch.context() as patched:
    patched.setenv('FLASHINFER_NPU_TUNING_DB', str(root.parent / 'no-actual-local.sqlite3'))
    patched.setattr(STORE, '_RELEASE_DATA_ROOT', root)
    registry, record = fixture_record(patched)
    record['record_id'] = 'synthetic-actual-hardware-mismatch'
    record['hardware'] = asdict(HardwareSignature('Ascend910B', 16, 131072))
    shard = root / hardware.soc_model / 'measurements.sqlite3'
    save_tuning_record(record, shard)
    (root / 'manifest.json').write_text(json.dumps(dict(schema_version=1, socs=[hardware.soc_model])))
    reload_tuning_records()
    decision = dispatch(SIG, hardware, registry=registry)
    assert query_tuning_record(SIG, hardware, registry=registry) is None
    assert record['result']['winner'] == B and decision.kernel_id == A
    explanation = explain_dispatch(SIG, hardware, registry=registry)
    assert 'source=release' in explanation and 'HardwareSignature 不匹配' in explanation
    print('SYNTHETIC invalid release fixture; not performance evidence')
    print('stored winner=%s; selected=%s' % (record['result']['winner'], decision.kernel_id))
    print(explanation)
    print('PASS release fixture rejected on the actual probed hardware')
```

## 去掉规则后的负对照

在独立 pytest 进程用临时 plugin 把 `_identity_reasons` 置空，并仅关闭读取时的 record
schema 检查；其余原始样本、bucket、SQL 摘要检查保留。工作区源文件未改。对应三条
测试确实全部失败，原因都是查询错误地返回 block256，而测试要求 None。
这是**故意禁用规则的负对照**，不是最终全量测试新增失败。

```bash
env PYTHONPATH=.:/tmp/t20b-evidence-4VlTnD FLASHINFER_NPU_TUNING_DB= /root/miniforge3/envs/flashinfer-a5/bin/python -m pytest -q -p t20b_disabled_guards tests/test_tuning_release.py::test_injected_descriptor_fingerprint_mismatch_is_rejected 'tests/test_tuning_release.py::test_release_hardware_mismatch_is_rejected[all]' 'tests/test_tuning_release.py::test_unsupported_record_schema_is_rejected[2-release]'
```

```text
        assert query_tuning_record(SIG, HW, registry=registry) == B
        record["schema_version"] = version
        rewrite_payload(path, record)
>       assert query_tuning_record(SIG, HW, registry=registry) is None
E       AssertionError: assert 'triton/t19a/gemma/block256' is None
E        +  where 'triton/t19a/gemma/block256' = query_tuning_record(InputSignature(op='norm.gemma_fused_add_rmsnorm', dtypes=('float32',), shapes=((4, 128),), layouts=(), specializations=(), physical_layouts=()), HardwareSignature(soc_model='Ascend950', aicore_count=32, ub_size_bytes=262144), registry=<flashinfer.runtime.registry.KernelRegistry object at 0x7de1b25af7d0>)

tests/test_tuning_release.py:208: AssertionError
=========================== short test summary info ============================
FAILED tests/test_tuning_release.py::test_injected_descriptor_fingerprint_mismatch_is_rejected
FAILED tests/test_tuning_release.py::test_release_hardware_mismatch_is_rejected[all]
FAILED tests/test_tuning_release.py::test_unsupported_record_schema_is_rejected[2-release]
3 failed in 0.06s
```

临时 plugin 完整内容：

```python
"""Negative control: disable only record identity and record schema read guards."""
import inspect
import pytest


@pytest.fixture(autouse=True)
def disabled_trust_guards(monkeypatch):
    import flashinfer.tuning.store as store
    original_validate = store._validate
    namespace = dict(vars(store))
    namespace['_identity_reasons'] = lambda *args: []
    namespace['_validate'] = lambda record: original_validate(dict(record, schema_version=1))
    source = inspect.getsource(store._read_database).replace(
        'if type(version) is not int or version != _SCHEMA_VERSION:', 'if False:')
    exec(compile(source, '<disabled tuning guards>', 'exec'), namespace)
    monkeypatch.setattr(store, '_read_database', namespace['_read_database'])
```

## 79 条无数据选型逐条对照

在任何改动之前写 snapshot，之后分别在“关闭全部 tuning”和“启用查询、本地文件不存在、
发版清单为空”两种条件下对照。第二种可证明实际查询路径的空库行为也不改变选择。

```bash
env PYTHONPATH=. FLASHINFER_NPU_TUNING_DB= /root/miniforge3/envs/flashinfer-a5/bin/python tools/t19_selection_audit.py --write /tmp/t20b-evidence-4VlTnD/t19-before.json
env PYTHONPATH=. FLASHINFER_NPU_TUNING_DB= /root/miniforge3/envs/flashinfer-a5/bin/python tools/t19_selection_audit.py --compare /tmp/t20b-evidence-4VlTnD/t19-before.json
env PYTHONPATH=. FLASHINFER_NPU_TUNING_DB=/tmp/t20b-evidence-4VlTnD/no-local.sqlite3 /root/miniforge3/envs/flashinfer-a5/bin/python tools/t19_selection_audit.py --compare /tmp/t20b-evidence-4VlTnD/t19-before.json
```

两次 compare 都通过所有字段逐条断言。下面完整粘贴启用查询且两级无数据时的实际输出：

```text
records: 79
stack: cann records: 40
stack: reference records: 38
stack: triton records: 1
multi_config_op_source_groups: 0 []
| record | before auto | after auto |
|---|---|---|
| cann/activation/gelu_and_mul_v1 | cann/activation/gelu_and_mul_v1 | cann/activation/gelu_and_mul_v1 |
| cann/activation/gelu_tanh_and_mul_v1 | cann/activation/gelu_tanh_and_mul_v1 | cann/activation/gelu_tanh_and_mul_v1 |
| cann/activation/silu_and_mul_v1 | cann/activation/silu_and_mul_v1 | cann/activation/silu_and_mul_v1 |
| cann/attention/batch_decode_paged_fia_v1 | cann/attention/batch_decode_paged_fia_v1 | cann/attention/batch_decode_paged_fia_v1 |
| cann/attention/batch_prefill_paged_fia_v1 | cann/attention/batch_prefill_paged_fia_v1 | cann/attention/batch_prefill_paged_fia_v1 |
| cann/attention/batch_prefill_ragged_fia_v1 | cann/attention/batch_prefill_ragged_fia_v1 | cann/attention/batch_prefill_ragged_fia_v1 |
| cann/attention/single_decode_fia_v1 | cann/attention/single_decode_fia_v1 | cann/attention/single_decode_fia_v1 |
| cann/attention/single_prefill_fia_v1 | cann/attention/single_prefill_fia_v1 | cann/attention/single_prefill_fia_v1 |
| cann/cascade/merge_state_attention_update_v1 | cann/cascade/merge_state_attention_update_v1 | cann/cascade/merge_state_attention_update_v1 |
| cann/cascade/merge_states_v1 | cann/cascade/merge_states_v1 | cann/cascade/merge_states_v1 |
| cann/decode/cudnn_batch_decode_fia_v1 | cann/decode/cudnn_batch_decode_fia_v1 | cann/decode/cudnn_batch_decode_fia_v1 |
| cann/fused_moe/cutlass_fused_moe_gmm_v1 | cann/fused_moe/cutlass_fused_moe_gmm_v1 | cann/fused_moe/cutlass_fused_moe_gmm_v1 |
| cann/fused_moe/fp8_block_scale_moe_cascade_v1 | cann/fused_moe/fp8_block_scale_moe_cascade_v1 | cann/fused_moe/fp8_block_scale_moe_cascade_v1 |
| cann/fused_moe/fp8_per_tensor_scale_moe_cascade_v1 | cann/fused_moe/fp8_per_tensor_scale_moe_cascade_v1 | cann/fused_moe/fp8_per_tensor_scale_moe_cascade_v1 |
| cann/gemm/bmm_bf16_v1 | cann/gemm/bmm_bf16_v1 | cann/gemm/bmm_bf16_v1 |
| cann/gemm/bmm_fp8_grouped_v1 | cann/gemm/bmm_fp8_grouped_v1 | cann/gemm/bmm_fp8_grouped_v1 |
| cann/gemm/gemm_fp8_nt_groupwise_block_v1 | cann/gemm/gemm_fp8_nt_groupwise_block_v1 | cann/gemm/gemm_fp8_nt_groupwise_block_v1 |
| cann/gemm/mm_bf16_v1 | cann/gemm/mm_bf16_v1 | cann/gemm/mm_bf16_v1 |
| cann/gemm/segment_gemm_v1 | cann/gemm/segment_gemm_v1 | cann/gemm/segment_gemm_v1 |
| cann/norm/fused_add_rmsnorm_aclnn_v1 | cann/norm/fused_add_rmsnorm_aclnn_v1 | cann/norm/fused_add_rmsnorm_aclnn_v1 |
| cann/norm/gemma_rmsnorm_aclnn_v1 | cann/norm/gemma_rmsnorm_aclnn_v1 | cann/norm/gemma_rmsnorm_aclnn_v1 |
| cann/norm/layernorm_aclnn_v1 | cann/norm/layernorm_aclnn_v1 | cann/norm/layernorm_aclnn_v1 |
| cann/norm/rmsnorm_aclnn_v1 | cann/norm/rmsnorm_aclnn_v1 | cann/norm/rmsnorm_aclnn_v1 |
| cann/page/append_paged_kv_cache_v1 | cann/page/append_paged_kv_cache_v1 | cann/page/append_paged_kv_cache_v1 |
| cann/rope/apply_rope_cos_sin_cache_aclnn_v1 | cann/rope/apply_rope_cos_sin_cache_aclnn_v1 | cann/rope/apply_rope_cos_sin_cache_aclnn_v1 |
| cann/rope/apply_rope_cos_sin_cache_out_v1 | cann/rope/apply_rope_cos_sin_cache_out_v1 | cann/rope/apply_rope_cos_sin_cache_out_v1 |
| cann/rope/apply_rope_inplace_v1 | cann/rope/apply_rope_inplace_v1 | cann/rope/apply_rope_inplace_v1 |
| cann/rope/apply_rope_pos_ids_inplace_v1 | cann/rope/apply_rope_pos_ids_inplace_v1 | cann/rope/apply_rope_pos_ids_inplace_v1 |
| cann/rope/apply_rope_pos_ids_v1 | cann/rope/apply_rope_pos_ids_v1 | cann/rope/apply_rope_pos_ids_v1 |
| cann/rope/apply_rope_v1 | cann/rope/apply_rope_v1 | cann/rope/apply_rope_v1 |
| cann/sampling/chain_speculative_sampling_v1 | cann/sampling/chain_speculative_sampling_v1 | cann/sampling/chain_speculative_sampling_v1 |
| cann/sampling/min_p_sampling_from_probs_v1 | cann/sampling/min_p_sampling_from_probs_v1 | cann/sampling/min_p_sampling_from_probs_v1 |
| cann/sampling/sampling_from_logits_v1 | cann/sampling/sampling_from_logits_v1 | cann/sampling/sampling_from_logits_v1 |
| cann/sampling/sampling_from_probs_v1 | cann/sampling/sampling_from_probs_v1 | cann/sampling/sampling_from_probs_v1 |
| cann/sampling/top_k_mask_logits_v1 | cann/sampling/top_k_mask_logits_v1 | cann/sampling/top_k_mask_logits_v1 |
| cann/sampling/top_k_renorm_probs_v1 | cann/sampling/top_k_renorm_probs_v1 | cann/sampling/top_k_renorm_probs_v1 |
| cann/sampling/top_k_sampling_from_probs_v1 | cann/sampling/top_k_sampling_from_probs_v1 | cann/sampling/top_k_sampling_from_probs_v1 |
| cann/sampling/top_k_top_p_sampling_from_probs_v1 | cann/sampling/top_k_top_p_sampling_from_probs_v1 | cann/sampling/top_k_top_p_sampling_from_probs_v1 |
| cann/sampling/top_p_renorm_probs_v1 | cann/sampling/top_p_renorm_probs_v1 | cann/sampling/top_p_renorm_probs_v1 |
| cann/sampling/top_p_sampling_from_probs_v1 | cann/sampling/top_p_sampling_from_probs_v1 | cann/sampling/top_p_sampling_from_probs_v1 |
| reference/activation/gelu_and_mul_v1 | reference/activation/gelu_and_mul_v1 | reference/activation/gelu_and_mul_v1 |
| reference/activation/gelu_tanh_and_mul_v1 | reference/activation/gelu_tanh_and_mul_v1 | reference/activation/gelu_tanh_and_mul_v1 |
| reference/activation/silu_and_mul_v1 | reference/activation/silu_and_mul_v1 | reference/activation/silu_and_mul_v1 |
| reference/attention/decode_v1 | reference/attention/decode_v1 | reference/attention/decode_v1 |
| reference/attention/prefill_ragged_v1 | reference/attention/prefill_ragged_v1 | reference/attention/prefill_ragged_v1 |
| reference/cascade/merge_state_v1 | cann/cascade/merge_state_attention_update_v1 | cann/cascade/merge_state_attention_update_v1 |
| reference/cascade/merge_states_v1 | reference/cascade/merge_states_v1 | reference/cascade/merge_states_v1 |
| reference/decode/cudnn_batch_decode_v1 | cann/decode/cudnn_batch_decode_fia_v1 | cann/decode/cudnn_batch_decode_fia_v1 |
| reference/fused_moe/cutlass_fused_moe_v1 | cann/fused_moe/cutlass_fused_moe_gmm_v1 | cann/fused_moe/cutlass_fused_moe_gmm_v1 |
| reference/fused_moe/trtllm_fp8_block_scale_moe_v1 | reference/fused_moe/trtllm_fp8_block_scale_moe_v1 | reference/fused_moe/trtllm_fp8_block_scale_moe_v1 |
| reference/fused_moe/trtllm_fp8_per_tensor_scale_moe_v1 | reference/fused_moe/trtllm_fp8_per_tensor_scale_moe_v1 | reference/fused_moe/trtllm_fp8_per_tensor_scale_moe_v1 |
| reference/gemm/bmm_bf16_v1 | cann/gemm/bmm_bf16_v1 | cann/gemm/bmm_bf16_v1 |
| reference/gemm/bmm_fp8_v1 | reference/gemm/bmm_fp8_v1 | reference/gemm/bmm_fp8_v1 |
| reference/gemm/gemm_fp8_nt_groupwise_v1 | reference/gemm/gemm_fp8_nt_groupwise_v1 | reference/gemm/gemm_fp8_nt_groupwise_v1 |
| reference/gemm/mm_bf16_v1 | cann/gemm/mm_bf16_v1 | cann/gemm/mm_bf16_v1 |
| reference/gemm/segment_gemm_v1 | cann/gemm/segment_gemm_v1 | cann/gemm/segment_gemm_v1 |
| reference/norm/fused_add_rmsnorm_v1 | cann/norm/fused_add_rmsnorm_aclnn_v1 | cann/norm/fused_add_rmsnorm_aclnn_v1 |
| reference/norm/gemma_fused_add_rmsnorm_v1 | triton/norm/gemma_fused_add_rmsnorm_v1 | triton/norm/gemma_fused_add_rmsnorm_v1 |
| reference/norm/gemma_rmsnorm_v1 | cann/norm/gemma_rmsnorm_aclnn_v1 | cann/norm/gemma_rmsnorm_aclnn_v1 |
| reference/norm/layernorm_v1 | cann/norm/layernorm_aclnn_v1 | cann/norm/layernorm_aclnn_v1 |
| reference/norm/rmsnorm_v1 | cann/norm/rmsnorm_aclnn_v1 | cann/norm/rmsnorm_aclnn_v1 |
| reference/page/append_paged_kv_cache_v1 | reference/page/append_paged_kv_cache_v1 | reference/page/append_paged_kv_cache_v1 |
| reference/rope/apply_rope_inplace_v1 | cann/rope/apply_rope_inplace_v1 | cann/rope/apply_rope_inplace_v1 |
| reference/rope/apply_rope_pos_ids_inplace_v1 | cann/rope/apply_rope_pos_ids_inplace_v1 | cann/rope/apply_rope_pos_ids_inplace_v1 |
| reference/rope/apply_rope_pos_ids_v1 | cann/rope/apply_rope_pos_ids_v1 | cann/rope/apply_rope_pos_ids_v1 |
| reference/rope/apply_rope_v1 | cann/rope/apply_rope_v1 | cann/rope/apply_rope_v1 |
| reference/rope/apply_rope_with_cos_sin_cache_out_v1 | cann/rope/apply_rope_cos_sin_cache_out_v1 | cann/rope/apply_rope_cos_sin_cache_out_v1 |
| reference/rope/apply_rope_with_cos_sin_cache_v1 | cann/rope/apply_rope_cos_sin_cache_aclnn_v1 | cann/rope/apply_rope_cos_sin_cache_aclnn_v1 |
| reference/sampling/chain_speculative_sampling_v1 | reference/sampling/chain_speculative_sampling_v1 | reference/sampling/chain_speculative_sampling_v1 |
| reference/sampling/min_p_sampling_from_probs_v1 | reference/sampling/min_p_sampling_from_probs_v1 | reference/sampling/min_p_sampling_from_probs_v1 |
| reference/sampling/sampling_from_logits_v1 | reference/sampling/sampling_from_logits_v1 | reference/sampling/sampling_from_logits_v1 |
| reference/sampling/sampling_from_probs_v1 | reference/sampling/sampling_from_probs_v1 | reference/sampling/sampling_from_probs_v1 |
| reference/sampling/top_k_mask_logits_v1 | reference/sampling/top_k_mask_logits_v1 | reference/sampling/top_k_mask_logits_v1 |
| reference/sampling/top_k_renorm_probs_v1 | reference/sampling/top_k_renorm_probs_v1 | reference/sampling/top_k_renorm_probs_v1 |
| reference/sampling/top_k_sampling_from_probs_v1 | reference/sampling/top_k_sampling_from_probs_v1 | reference/sampling/top_k_sampling_from_probs_v1 |
| reference/sampling/top_k_top_p_sampling_from_probs_v1 | reference/sampling/top_k_top_p_sampling_from_probs_v1 | reference/sampling/top_k_top_p_sampling_from_probs_v1 |
| reference/sampling/top_p_renorm_probs_v1 | reference/sampling/top_p_renorm_probs_v1 | reference/sampling/top_p_renorm_probs_v1 |
| reference/sampling/top_p_sampling_from_probs_v1 | reference/sampling/top_p_sampling_from_probs_v1 | reference/sampling/top_p_sampling_from_probs_v1 |
| triton/norm/gemma_fused_add_rmsnorm_v1 | triton/norm/gemma_fused_add_rmsnorm_v1 | triton/norm/gemma_fused_add_rmsnorm_v1 |
PASS 79/79 auto selections unchanged; 79/79 descriptor fingerprints unchanged
PASS hot-path selections, fallback order and per-candidate filters unchanged
```

## 相邻问题与边界（本次未修改）

1. 本机现有硬件探测返回 UB=0，并出现 `AclrtGetDeviceInfo` 507899 警告。
   没有改硬件探测器、驱动或 CANN；也不凭该 warning 断定具体环境故障。
   UB 未知的记录仍可保存供审计，但不用于选型或发版导出。采集工具兼容这一拒选结果，
   不再假设“写入成功就一定能用于当前 dispatch”。
2. 三条 ABI 冻结指纹失败保留原样，名称和 actual/expected 都未变化。
3. 指纹检查依赖声明准确：源码/配置/产物变化必须反映到已有 descriptor/ID/SourceRef/ArtifactRef。
   部分现有 provider 没有完整源码/产物声明；T20A 的 JitSpec、JIT 产物和软件版本只存于
   自由格式 metadata，dispatch registry 没有当前值可比。没有新造环境指纹，也没有在
   import/dispatch 中探测编译器。**不能声称能自动发现未进入 descriptor 的 JIT 选项、
   CANN/驱动或二进制更换**；这需要后续绑定现有运行身份。在此之前相关变更后需丢弃/重新采集记录。
4. 跨进程数据库刷新仍显式 reload/重启；没有添加文件 watcher 或 TTL。

`git diff --check` 通过；`benchmark.py`、runtime `schema.py`/`registry.py`、
`docs/tasks/` 与 `docs/TASKS.md` 的 `git diff --exit-code` 均为 0。
没有自动搜索、调参空间探索、commit 或 push。
