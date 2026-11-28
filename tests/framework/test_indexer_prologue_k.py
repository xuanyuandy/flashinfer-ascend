"""Rich K mutation schema, static checks and additive alias declaration."""

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import flashinfer
from flashinfer.ascendc import load_contract
from flashinfer.ascendc.attention.indexer_prologue_k.implementation import IMPLEMENTATION
from flashinfer.attention.indexer_prologue import _validate_slots, validate_indexer_prologue_k_static
from flashinfer.operators.attention.indexer_prologue_k.operator import OPERATOR
from flashinfer.runtime import GraphCapability

KERNEL_ID = "ascendc/attention/indexer_prologue_k/native/op_package/Ascend910B"


def arguments(tokens=3, blocks=4, slots=5, cache_strides=None):
    key_stride, slot_stride, scale_block_stride, scale_slot_stride = cache_strides or (
        slots * 128 + 7, 128, slots + 3, 1)
    latent = torch.zeros(tokens, 512, dtype=torch.bfloat16)
    wk = torch.zeros(128, 512, dtype=torch.bfloat16)
    norm = torch.ones(128, dtype=torch.bfloat16)
    cos = torch.ones(tokens, 1, 1, 64)
    sin = torch.zeros(tokens, 1, 1, 64)
    indices = torch.zeros(tokens, 2, dtype=torch.int32)
    indices[0] = torch.tensor([-1, -1], dtype=torch.int32)
    key_storage = torch.zeros(blocks * key_stride + 16, dtype=torch.int8)
    scale_storage = torch.zeros(blocks * scale_block_stride + 8, dtype=torch.float16)
    key = torch.as_strided(key_storage, (blocks, slots, 1, 128),
                           (key_stride, slot_stride, 128, 1), 3)
    scale = torch.as_strided(scale_storage, (blocks, slots, 1, 1),
                             (scale_block_stride, scale_slot_stride, 1, 1), 2)
    return dict(latent=latent, wk=wk, norm_weight=norm, rope_cos=cos, rope_sin=sin,
                indices=indices, k_cache=key, k_scale_cache=scale)


def test_exports_and_mutation_schema():
    import flashinfer.attention

    assert flashinfer.indexer_prologue_k is flashinfer.attention.indexer_prologue_k
    assert inspect.signature(flashinfer.indexer_prologue_k).parameters["norm_eps"].default == 1e-6
    signature = OPERATOR.torch_schema()[0] if OPERATOR.schema else None
    from flashinfer.operators.attention.indexer_prologue_k.operator import torch_declarations
    declaration = torch_declarations()[0]
    assert "Tensor(a!) k_cache" in declaration.schema
    assert "Tensor(b!) k_scale_cache" in declaration.schema
    assert declaration.schema.endswith("-> ()")
    assert declaration.mutates_args == ("k_cache", "k_scale_cache")


def test_fake_tensor_returns_none_without_reading_slots():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from flashinfer.aclnn._ops import ensure_registered

    ensure_registered()
    args = arguments()
    with FakeTensorMode():
        fake = {name: torch.empty(tuple(tensor.shape), dtype=tensor.dtype)
                for name, tensor in args.items()}
        assert torch.ops.flashinfer.indexer_prologue_k(
            fake["latent"], fake["wk"], fake["norm_weight"], fake["rope_cos"],
            fake["rope_sin"], fake["indices"], fake["k_cache"], fake["k_scale_cache"]) is None


def test_static_contract_accepts_strided_caches_and_rejects_bad_geometry():
    args = arguments()
    tokens, blocks, slots = validate_indexer_prologue_k_static(
        args["latent"], args["wk"], args["norm_weight"], args["rope_cos"],
        args["rope_sin"], args["indices"], args["k_cache"], args["k_scale_cache"], 1e-6,
        check_device=False)
    assert (tokens, blocks, slots) == (3, 4, 5)
    with pytest.raises(ValueError, match="contiguous last axis"):
        bad = torch.as_strided(torch.zeros(8192, dtype=torch.int8),
                               (4, 5, 1, 128), (647, 129, 128, 2), 3)
        validate_indexer_prologue_k_static(
            args["latent"], args["wk"], args["norm_weight"], args["rope_cos"],
            args["rope_sin"], args["indices"], bad, args["k_scale_cache"], 1e-6,
            check_device=False)
    with pytest.raises(ValueError, match="positive strides on non-singleton"):
        bad = torch.as_strided(torch.zeros(8192, dtype=torch.int8),
                               (4, 5, 1, 128), (0, 128, 128, 1), 3)
        validate_indexer_prologue_k_static(
            args["latent"], args["wk"], args["norm_weight"], args["rope_cos"],
            args["rope_sin"], args["indices"], bad, args["k_scale_cache"], 1e-6,
            check_device=False)
    with pytest.raises(ValueError, match="norm_eps"):
        validate_indexer_prologue_k_static(
            args["latent"], args["wk"], args["norm_weight"], args["rope_cos"],
            args["rope_sin"], args["indices"], args["k_cache"], args["k_scale_cache"], 0.0,
            check_device=False)


def test_singleton_zero_stride_is_accepted_and_non_singleton_zero_is_rejected():
    args = arguments()
    key = torch.as_strided(torch.zeros(256, dtype=torch.int8), (1, 1, 1, 128),
                           (0, 0, 128, 1), 0)
    scale = torch.as_strided(torch.zeros(8, dtype=torch.float16), (1, 1, 1, 1),
                             (0, 0, 1, 1), 0)
    validate_indexer_prologue_k_static(
        args["latent"][:1], args["wk"], args["norm_weight"], args["rope_cos"][:1],
        args["rope_sin"][:1], torch.zeros(1, 2, dtype=torch.int32), key, scale,
        1e-6, check_device=False)
    bad = torch.as_strided(torch.zeros(512, dtype=torch.int8), (2, 1, 1, 128),
                           (0, 0, 128, 1), 0)
    with pytest.raises(ValueError, match="non-singleton"):
        validate_indexer_prologue_k_static(
            args["latent"][:1], args["wk"], args["norm_weight"], args["rope_cos"][:1],
            args["rope_sin"][:1], torch.zeros(1, 2, dtype=torch.int32), bad, scale,
            1e-6, check_device=False)


def test_tensor_adapter_singleton_zero_stride_and_empty_span():
    from flashinfer.jit.tensor_adapter import _reject_output_overlap, _storage_span

    parameter = SimpleNamespace(role="input", alias_of=None, allow_strided=True)
    singleton = torch.as_strided(torch.zeros(256, dtype=torch.int8), (1, 1, 1, 128),
                                 (0, 0, 128, 1), 0)
    _storage_span(parameter, singleton, "ND")
    bad = torch.as_strided(torch.zeros(512, dtype=torch.int8), (2, 1, 1, 128),
                           (0, 0, 128, 1), 0)
    with pytest.raises(ValueError, match="non-singleton"):
        _storage_span(parameter, bad, "ND")

    empty = torch.empty(0)
    roles = SimpleNamespace(role="output", alias_of=None)
    _reject_output_overlap((roles, roles), (empty, empty))
    storage = torch.zeros(130)
    strided = torch.as_strided(storage, (2, 2), (64, 1))
    with pytest.raises(ValueError, match="must not overlap"):
        _reject_output_overlap(
            (SimpleNamespace(role="output", alias_of=None),
             SimpleNamespace(role="input", alias_of=None)),
            (strided, storage),
        )


def test_k_oracle_rejects_changes_to_untouched_key_gaps():
    from flashinfer.operators.attention.indexer_prologue_k import cases

    cpu, expected = cases.make_inputs(torch, tokens=4)
    key_cpu, scale_cpu = cpu["k_cache"], cpu["k_scale_cache"]
    key_storage, scale_storage = expected[0].clone(), expected[1].clone()
    key = torch.as_strided(key_storage, key_cpu.shape, key_cpu.stride(),
                           key_cpu.storage_offset())
    scale = torch.as_strided(scale_storage, scale_cpu.shape, scale_cpu.stride(),
                             scale_cpu.storage_offset())
    args = (cpu["latent"], cpu["wk"], cpu["norm_weight"], cpu["rope_cos"],
            cpu["rope_sin"], cpu["indices"], key, scale)
    cases.verify_mutation(torch, args, expected)
    key_storage[0] = key_storage[0] ^ 1  # Before storage_offset=3: untouched canary.
    with pytest.raises(AssertionError):
        cases.verify_mutation(torch, args, expected)


def test_overlapping_slot_geometry_is_rejected_without_indices_uniqueness():
    args = arguments()
    overlapping = torch.as_strided(torch.zeros(8192, dtype=torch.int8),
                                   (4, 5, 1, 128), (256, 1, 128, 1), 0)
    with pytest.raises(ValueError, match="overlap distinct physical slots"):
        validate_indexer_prologue_k_static(
            args["latent"], args["wk"], args["norm_weight"], args["rope_cos"],
            args["rope_sin"], args["indices"], overlapping, args["k_scale_cache"],
            1e-6, check_device=False)
    # Duplicate indices remain outside the uniqueness contract: only padding
    # and bounds are validated.
    duplicate = torch.tensor([[0, 0], [0, 0]], dtype=torch.int32)
    _validate_slots(duplicate, 4, 5)


@pytest.mark.parametrize("eps", [1e-50, 1e-46, 0.0, -1e-6])
def test_norm_eps_must_remain_positive_after_binary32_narrowing(eps):
    args = arguments()
    with pytest.raises(ValueError, match="norm_eps"):
        validate_indexer_prologue_k_static(
            args["latent"], args["wk"], args["norm_weight"], args["rope_cos"],
            args["rope_sin"], args["indices"], args["k_cache"], args["k_scale_cache"],
            eps, check_device=False)


def test_slot_values_preserve_padding_and_reject_other_negatives():
    indices = torch.tensor([[-1, -1], [0, 2]], dtype=torch.int32)
    _validate_slots(indices, 4, 5)
    for row in ([-1, 0], [0, -1], [-2, 1], [4, 0], [0, 5]):
        with pytest.raises(ValueError, match="unsupported negative|outside"):
            _validate_slots(torch.tensor([row], dtype=torch.int32), 4, 5)


def test_native_declaration_has_exact_alias_pairs_and_no_graph_claim():
    recipe = load_contract(KERNEL_ID)[1]
    assert recipe.tensor_parameter_names[-2:] == ("k_cache_out", "k_scale_cache_out")
    assert recipe.alias_pairs == ((6, 8), (7, 9))
    assert IMPLEMENTATION.bind().graph_capability is GraphCapability.NOT_CAPTURABLE


def test_build_tables_never_write_past_the_128_entry_buffer():
    source = (Path(__file__).resolve().parents[2]
              / "csrc/kernels/attention/indexer_prologue_k/op_kernel/indexer_prologue_k.cpp").read_text()
    assert "InitBuffer(bufTables, 128 * sizeof(uint32_t))" in source
    assert "CopyAddRows(itOff[32], itOff, 64, ROPE_HALF);" in source
    assert "CopyAddRows(itOff[32], itOff, 64, ROPE_DIM);" not in source


def test_declared_torch_schema_is_detected_as_void_mutation():
    from flashinfer.operators.attention.indexer_prologue_k import cases as k_cases
    from flashinfer.operators.attention.indexer_prologue_k.operator import torch_declarations
    from flashinfer.operators.testing import is_void_mutation

    declaration = torch_declarations()[0]
    assert declaration.schema == OPERATOR.torch_declarations()[0].schema
    assert declaration.mutates_args == ("k_cache", "k_scale_cache")
    assert is_void_mutation(OPERATOR, k_cases)


def test_rich_capability_fingerprint_covers_torch_schema_and_mutation_set():
    from dataclasses import replace
    from flashinfer.ascendc.attention.indexer_prologue_k.implementation import IMPLEMENTATION
    from flashinfer.jit.torch_library import AscendCOp
    from flashinfer.operators.attention.indexer_prologue_k.operator import OPERATOR
    from flashinfer.operators.capability import direct_operator_binding, validate_direct_operator_binding

    descriptor = IMPLEMENTATION.bind()
    base = direct_operator_binding(OPERATOR, descriptor.capability, "ascendc", descriptor.kernel_id)
    original = OPERATOR.torch_declarations()[0]

    def declaration(schema, mutates):
        return lambda: (AscendCOp(original.name, schema, original.device_impl,
                                  original.meta_impl, mutates),)

    changed_schema = replace(OPERATOR, torch_declarations=declaration(
        original.schema.replace("Tensor latent", "Tensor latent_changed"), original.mutates_args))
    changed_mutation = replace(OPERATOR, torch_declarations=declaration(
        original.schema, ("k_cache",)))
    assert direct_operator_binding(changed_schema, descriptor.capability, "ascendc",
                                   descriptor.kernel_id).profile_fingerprint != base.profile_fingerprint
    assert direct_operator_binding(changed_mutation, descriptor.capability, "ascendc",
                                   descriptor.kernel_id).profile_fingerprint != base.profile_fingerprint
    changed_binding = direct_operator_binding(changed_schema, descriptor.capability,
                                              "ascendc", descriptor.kernel_id)
    mismatch = replace(descriptor, capability_binding=changed_binding)
    with pytest.raises(Exception, match="capability"):
        validate_direct_operator_binding(mismatch)


def test_public_facade_accepts_generic_harness_routing_options():
    parameters = inspect.signature(flashinfer.indexer_prologue_k).parameters
    assert "kernel_id" in parameters and "backend" in parameters


def test_tiling_uses_stride_metadata_without_requiring_view_flag():
    source = (Path(__file__).resolve().parents[2]
              / "csrc/kernels/attention/indexer_prologue_k/op_host/arch22/indexer_prologue_k_tiling.cpp").read_text()
    assert "GetInputStride(6)" in source
    assert "GetInputStride(7)" in source
    assert "stride metadata is required" in source
    assert "if (context->InputIsView(6))" not in source
    assert "if (context->InputIsView(7))" not in source
