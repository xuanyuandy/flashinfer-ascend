"""Migration evidence and backend-independent registration/execution contracts."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from flashinfer.operators.authoring import discover, discover_implementations, generate
from flashinfer.operators.implementation import ImplementationDefinition, implementation_entries
from flashinfer.runtime import (
    Backend, CapabilityCase, CapabilitySpec, FeatureConstraint, HardwareSignature, InputSignature, KernelDescriptor,
    KernelRegistry, SchemaError, dispatch, get_global_registry,
)
from flashinfer.runtime.registry import kernel_registry_scope
from flashinfer.runtime.runner import run_selected

ROOT = Path(__file__).resolve().parents[1]


def test_existing_85_and_reviewed_arch22_capabilities_match_exactly():
    baseline = json.loads((ROOT / 'tests/fixtures/operator_layout/capabilities.json').read_text())
    actual = {d.kernel_id: d.capability.to_dict() for d in get_global_registry().descriptors()}
    # Keep the original 85-entry fixture frozen. Reviewed additions and SoC
    # changes are explicit below, never copied from implementation discovery.
    expected = dict(baseline)
    expected["ascendc/t4/flash_attention/catlass/op_package/Ascend910B"] = dict(
        baseline["ascendc/t4/flash_attention/catlass/op_package/Ascend950"],
        soc_models=["Ascend910B"],
    )
    expected["ascendc/attention/sparse_flash_attention/native/Ascend910B"] = CapabilitySpec(
        op="attention.sparse_flash_attention",
        soc_models=("Ascend910B",),
        dtypes=("float16", "bfloat16", "int32", "float32", "none"),
        cases=tuple(CapabilityCase(
            name="arch22_mla_lse" if lse else "arch22_mla_no_lse",
            compute_dtypes=("float16", "bfloat16"),
            accumulator_dtypes=("float32",), attention_variants=("mla",),
            kv_cache_types=("dense", "ragged", "paged"), causal_values=(False, True),
            sparse_formats=("block_sparse",),
            auxiliary_outputs=("softmax_max", "softmax_sum") if lse else (),
        ) for lse in (False, True)),
    ).to_dict()
    # Only these reviewed general-purpose backends extend their SoC lists.
    # The frozen fixture remains unchanged; every other capability field is exact.
    for kernel_id in (
        'cann/cascade/merge_state_attention_update_v1',
        'cann/decode/cudnn_batch_decode_fia_v1',
        'cann/fused_moe/cutlass_fused_moe_gmm_v1',
        'cann/gemm/bmm_bf16_v1',
        'cann/gemm/mm_bf16_v1',
        'cann/gemm/segment_gemm_v1',
        'cann/norm/fused_add_rmsnorm_aclnn_v1',
        'cann/norm/gemma_rmsnorm_aclnn_v1',
        'cann/norm/layernorm_aclnn_v1',
        'cann/norm/rmsnorm_aclnn_v1',
        'cann/rope/apply_rope_cos_sin_cache_aclnn_v1',
        'cann/rope/apply_rope_cos_sin_cache_out_v1',
        'cann/rope/apply_rope_inplace_v1',
        'cann/rope/apply_rope_pos_ids_inplace_v1',
        'cann/rope/apply_rope_pos_ids_v1',
        'cann/rope/apply_rope_v1',
        'triton/norm/gemma_fused_add_rmsnorm_v1',
    ):
        expected[kernel_id] = dict(baseline[kernel_id],
                                   soc_models=["Ascend910B", "Ascend910C", "Ascend950"])
    for kernel_id in (
        "ascendc/t4/flash_attention/catlass/op_package/Ascend910B",
        "ascendc/gemm/grouped_matmul_swiglu_quant/native/op_package/Ascend910B",
        "ascendc/attention/sparse_flash_attention/native/Ascend910B",
    ):
        expected[kernel_id.rsplit("/", 1)[0] + "/Ascend910C"] = dict(
            expected[kernel_id], soc_models=["Ascend910C"])
    assert actual == expected


def test_all_existing_implementations_have_one_authority_and_a_bound_runner():
    declarations = discover_implementations(ROOT)
    production = [d for _, _, d in declarations if d.scope == 'global']
    registry = get_global_registry()
    assert {d.kernel_id for d in production} == {d.kernel_id for d in registry.descriptors()}
    assert len(production) == 90
    assert len(discover(ROOT)) == 48
    assert len([d for _, _, d in declarations if d.scope == 'experimental']) == 1
    for directory, _, declaration in declarations:
        assert directory.parent.parent.name == declaration.backend.value
        assert callable(declaration.bind().runner)
    assert {r['op'] for r in implementation_entries()} == {o.op for _, o in discover(ROOT)}
    generate(check=True)


def test_admission_rejects_only_its_own_backend_before_ranking_and_execution():
    calls = []
    first = KernelDescriptor('cann/example/narrow', op='example.op', backend=Backend.CANN,
        capability=CapabilitySpec('example.op'), runner=lambda value: calls.append(('narrow', value)),
        admission=lambda signature: ('native shape bound exceeded',) if signature.shapes[0][0] > 8 else ())
    second = KernelDescriptor('triton/example/wide', op='example.op', backend=Backend.TRITON,
        capability=CapabilitySpec('example.op'), runner=lambda value: calls.append(('wide', value)))
    registry = KernelRegistry((first, second)); registry.freeze()
    signature = InputSignature('example.op', shapes=((16,),))
    hardware = HardwareSignature('Ascend910B', 20)
    with kernel_registry_scope(registry):
        decision = dispatch(signature, hardware)
        assert decision.kernel_id == second.kernel_id
        run_selected(decision.kernel_id, 'logical argument')
        with pytest.raises(RuntimeError, match='native shape bound exceeded'):
            dispatch(signature, hardware, kernel_id=first.kernel_id)
    assert calls == [('wide', 'logical argument')]


def test_implementation_factory_cannot_change_identity_or_omit_runner():
    descriptor = KernelDescriptor('cann/example/one', op='example.op', backend=Backend.CANN,
                                  runner=lambda: None)
    definition = ImplementationDefinition(descriptor.kernel_id, descriptor.op, descriptor.backend,
                                          lambda: descriptor)
    assert definition.bind() is descriptor
    with pytest.raises(SchemaError, match='identity'):
        replace(definition, factory=lambda: replace(descriptor, op='other.op')).bind()
    with pytest.raises(SchemaError, match='runner'):
        replace(definition, factory=lambda: replace(descriptor, runner=None)).bind()


def test_common_matmul_metadata_never_reads_an_ascendc_contract(monkeypatch):
    from types import SimpleNamespace
    from flashinfer._matmul import metadata_reason
    monkeypatch.setattr('flashinfer.ascendc.load_contract', lambda _: pytest.fail('backend coupling'))
    tensor = lambda shape: SimpleNamespace(shape=shape, dtype='bfloat16', device='meta')
    assert metadata_reason(tensor((16, 32)), tensor((32, 64)), tensor((16, 64))) == 'supported'
    assert metadata_reason(tensor((16, 31)), tensor((32, 64)), tensor((16, 64))) != 'supported'


def test_logical_call_can_use_another_backend_without_loading_a_native_recipe(monkeypatch):
    import torch
    from flashinfer.operators.catalog import definition
    from flashinfer.operators import runtime

    operator = definition('gemm.grouped_matmul_swiglu_quant')
    observed = []
    def runner(x, weight, scale, x_scale, assistance, groups, out, out_scale, *, swiglu_limit):
        observed.append((assistance, groups, swiglu_limit))
        out.zero_()
        out_scale.fill_(1)
    monkeypatch.setattr('flashinfer.ascendc.load_contract', lambda _: pytest.fail('native recipe leak'))
    monkeypatch.setattr(runtime, 'current_hardware', lambda: HardwareSignature('Ascend910B', 20))
    x = torch.ones((2, 16), dtype=torch.int8)
    weight = torch.ones((1, 16, 32), dtype=torch.int8)
    scale = torch.ones((1, 32))
    x_scale = torch.ones(2)
    groups = torch.tensor([2], dtype=torch.int64)
    arguments = operator.bind(x, weight, scale, x_scale, groups, backend='reference', swiglu_limit=.5)
    inference = operator.infer(arguments, {"weight": "ND"}).inference
    capability = CapabilitySpec(operator.op, cases=(CapabilityCase("logical_reference_test",
        compute_dtypes=(inference.compute_dtype,), accumulator_dtypes=(inference.accumulator_dtype,),
        quantization=inference.quantization, fusion_ops=inference.fusion_ops,
        auxiliary_outputs=inference.auxiliary_outputs,
        features=tuple(FeatureConstraint(name, (value,)) for name, value in inference.features)),))
    descriptor = KernelDescriptor('reference/test/grouped', op=operator.op, backend=Backend.REFERENCE,
                                  capability=capability, runner=runner)
    registry = KernelRegistry((descriptor,)); registry.freeze()
    with kernel_registry_scope(registry):
        out, out_scale = runtime.execute(operator.op, arguments)
    assert observed == [(None, groups, .5)]
    assert out.shape == (2, 16) and out_scale.shape == (2,)
    assert bool((out == 0).all()) and bool((out_scale == 1).all())


def test_functional_out_alias_is_rejected_without_loading_a_backend(monkeypatch):
    import torch
    from flashinfer.operators.catalog import definition
    from flashinfer.operators import runtime
    operator = definition('gemm.grouped_matmul_swiglu_quant')
    monkeypatch.setattr('flashinfer.ascendc.load_contract', lambda _: pytest.fail('native recipe leak'))
    x = torch.ones((2, 16), dtype=torch.int8)
    weight = torch.ones((1, 16, 32), dtype=torch.int8)
    scale = torch.ones((1, 32))
    x_scale = torch.ones(2)
    groups = torch.tensor([2], dtype=torch.int64)
    out = torch.empty((2, 16), dtype=torch.int8)
    arguments = operator.bind(x, weight, scale, x_scale, groups, out=(out, x_scale))
    with pytest.raises(ValueError, match='overlap'):
        runtime.metadata(operator.op, arguments, check_layouts=True)


def test_every_public_entry_resolves_and_preserves_its_frontend_signature():
    import importlib
    import inspect
    from flashinfer.operators.catalog import definition, entries
    for record in entries():
        operator = definition(record["op"])
        if operator.public_entry is None:
            continue
        module, attribute = operator.public_entry.split(":", 1)
        frontend = getattr(importlib.import_module(module), attribute)
        assert callable(frontend), operator.op
        assert operator.public_signature == inspect.signature(frontend), operator.op
