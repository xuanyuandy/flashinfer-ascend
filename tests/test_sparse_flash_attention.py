"""Public semantics, native admission and A2 sparse-attention acceptance."""

import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from flashinfer.ascendc.attention.sparse_flash_attention.implementation import IMPLEMENTATION
from flashinfer.ascendc.attention.sparse_flash_attention.runner import SparseFlashAttentionRunner
from flashinfer.operators.attention.sparse_flash_attention.cases import golden
from flashinfer.operators.attention.sparse_flash_attention.operator import OPERATOR, validate_values
from flashinfer.operators.runtime import fake, signature
from flashinfer.runtime import GraphCapability, HardwareSignature, dispatch

KERNEL_ID = "ascendc/attention/sparse_flash_attention/native/Ascend910B"


def arguments(*, layout="BSND", lse=True, paged=False, rope=True, dtype=torch.float16):
    qshape = (2, 3, 4, 512) if layout == "BSND" else (6, 4, 512)
    kshape = (4, 16, 1, 512) if paged else ((2, 7, 1, 512) if layout == "BSND" else (14, 1, 512))
    q, k, v = (torch.zeros(shape, dtype=dtype) for shape in (qshape, kshape, kshape))
    indices = torch.zeros(qshape[:-2] + (1, 3), dtype=torch.int32)
    options = dict(layout_query=layout, layout_kv="PA_BSND" if paged else layout,
                   return_softmax_lse=lse)
    if layout == "TND":
        options["actual_seq_lengths_query"] = torch.tensor([3, 6], dtype=torch.int32)
    if layout == "TND" or paged:
        options["actual_seq_lengths_kv"] = torch.tensor([7, 7] if paged else [7, 14], dtype=torch.int32)
    if paged:
        options["block_table"] = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    if rope:
        options.update(query_rope=torch.zeros(qshape[:-1] + (64,), dtype=dtype),
                       key_rope=torch.zeros(kshape[:-1] + (64,), dtype=dtype))
    return OPERATOR.bind(q, k, v, indices, 0.125, **options)


def test_public_exports_share_the_declared_interface():
    import flashinfer
    import flashinfer.attention
    assert flashinfer.sparse_flash_attention is flashinfer.attention.sparse_flash_attention
    assert inspect.signature(flashinfer.sparse_flash_attention) == OPERATOR.public_signature
    assert OPERATOR.torch_schema()[0].endswith("-> (Tensor, Tensor, Tensor)")
    assert "Tensor(a!) attention_out" in OPERATOR.torch_schema(out=True)[0]


@pytest.mark.parametrize("layout", ["BSND", "TND"])
@pytest.mark.parametrize("lse", [False, True])
@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("rope", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fake_shapes_dispatch_and_native_contract(layout, lse, paged, rope, dtype):
    args = arguments(layout=layout, lse=lse, paged=paged, rope=rope, dtype=dtype)
    outputs = fake(OPERATOR.op, args)
    assert outputs[0].shape == args["query"].shape and outputs[0].dtype == dtype
    aux = ((2, 1, 3, 4) if layout == "BSND" else (1, 6, 4)) if lse else (0,)
    assert all(t.shape == aux and t.dtype == torch.float32 for t in outputs[1:])
    request, _, _ = signature(OPERATOR.op, args)
    assert dispatch(request, HardwareSignature("Ascend910B", 20)).kernel_id == KERNEL_ID
    validate_values(args)
    with pytest.raises(RuntimeError):
        dispatch(request, HardwareSignature("Ascend950", 32), kernel_id=KERNEL_ID)


@pytest.mark.parametrize("change,match", [
    ({"sparse_mode": 1}, "sparse_mode"),
    ({"sparse_block_size": 3}, "power of two"),
    ({"sparse_block_size": 256}, "power of two"),
    ({"attention_mode": 0}, "attention_mode"),
    ({"pre_tokens": 3}, "INT64_MAX"),
    ({"next_tokens": 3}, "INT64_MAX"),
    ({"scale_value": float("nan")}, "scale_value"),
    ({"scale_value": float("inf")}, "scale_value"),
    ({"layout_kv": "TND"}, "layouts"),
    ({"query_rope": None}, "supplied together"),
])
def test_invalid_attributes_are_rejected_before_aot(change, match):
    args = dict(arguments(), **change)
    with pytest.raises(ValueError, match=match):
        fake(OPERATOR.op, args)


def test_invalid_tensor_metadata_and_out_alias_are_rejected():
    args = arguments()
    for key, tensor in (("value", torch.zeros(2, 8, 1, 512, dtype=torch.float16)),
                        ("query", torch.zeros(2, 3, 65, 512, dtype=torch.float16)),
                        ("sparse_indices", torch.zeros(2, 3, 1, 3, dtype=torch.int64))):
        with pytest.raises(ValueError):
            fake(OPERATOR.op, dict(args, **{key: tensor}))
    outputs = fake(OPERATOR.op, args)
    with pytest.raises(ValueError, match="overlap"):
        signature(OPERATOR.op, dict(args, out=(args["query"], *outputs[1:])))
    with pytest.raises(ValueError, match="out shape/dtype"):
        fake(OPERATOR.op, dict(args, out=(outputs[0], torch.zeros(0), outputs[2])))


@pytest.mark.parametrize("indices", [(-2, -1, -1), (7, -1, -1), (-1, 0, -1)])
def test_invalid_sparse_addresses_fail_before_launch(indices):
    args = arguments()
    args["sparse_indices"][:] = torch.tensor(indices, dtype=torch.int32)
    with pytest.raises(ValueError, match="sparse_indices"):
        validate_values(args)


def test_invalid_prefixes_and_physical_page_addresses_fail_before_launch():
    args = arguments(layout="TND")
    args["actual_seq_lengths_query"][:] = torch.tensor([7, 6], dtype=torch.int32)
    with pytest.raises(ValueError, match="prefix sums"):
        validate_values(args)
    args = arguments(paged=True)
    args["block_table"][1, 0] = 4
    with pytest.raises(ValueError, match="physical KV block"):
        validate_values(args)
    args = arguments()
    args["actual_seq_lengths_kv"] = torch.tensor([8, 7], dtype=torch.int32)
    with pytest.raises(ValueError, match="capacity"):
        validate_values(args)


def test_disabled_native_auxiliary_buffers_do_not_replace_public_out_tensors():
    args = arguments(lse=False)
    invocation = OPERATOR.infer(args, None)
    outputs = fake(OPERATOR.op, args)
    observed = []
    runner = SparseFlashAttentionRunner(lambda *tensors, **attrs: observed.append((tensors, attrs)))
    runner(*invocation.inputs, *outputs, **invocation.attributes)
    tensors, attrs = observed[0]
    assert tensors[9] is outputs[0]
    assert outputs[1].shape == outputs[2].shape == (0,)
    assert tensors[10].shape == tensors[11].shape == (1,)
    assert tensors[10].data_ptr() != tensors[11].data_ptr()
    assert attrs["return_softmax_lse"] is False


def test_runner_preserves_shared_kv_storage_and_supplied_auxiliary_outputs():
    args = arguments()
    args["value"] = args["key"]
    invocation = OPERATOR.infer(args, None)
    outputs = fake(OPERATOR.op, args)
    observed = []
    runner = SparseFlashAttentionRunner(lambda *tensors, **attrs: observed.append((tensors, attrs)))
    runner(*invocation.inputs, *outputs, **invocation.attributes)
    tensors, attrs = observed[0]
    assert tensors[1] is tensors[2] is args["key"]
    assert all(tensors[9 + i] is tensor for i, tensor in enumerate(outputs))


def test_torch_functional_and_out_fake_registration():
    import flashinfer
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        q = torch.empty(1, 3, 4, 512)
        k = torch.empty(1, 7, 1, 512)
        indices = torch.empty(1, 3, 1, 8, dtype=torch.int32)
        outputs = flashinfer.sparse_flash_attention(q.half(), k.half(), k.half(), indices, .125)
        assert outputs[0].shape == q.shape and outputs[1].shape == outputs[2].shape == (0,)
        supplied = tuple(torch.empty_like(t) for t in outputs)
        returned = flashinfer.attention.sparse_flash_attention(q.half(), k.half(), k.half(), indices,
                                                              .125, out=supplied)
        assert all(a is b for a, b in zip(returned, supplied))


def test_cpu_oracle_checks_sparse_blocks_causal_rope_and_auxiliary_values():
    q = torch.ones(1, 2, 1, 512, dtype=torch.float16)
    k = torch.zeros(1, 4, 1, 512, dtype=torch.float16)
    v = torch.arange(4, dtype=torch.float16).view(1, 4, 1, 1).expand_as(k)
    indices = torch.tensor([[[[0, 1]], [[0, 1]]]], dtype=torch.int32)
    qr = torch.ones(1, 2, 1, 64, dtype=torch.float16)
    kr = torch.ones(1, 4, 1, 64, dtype=torch.float16)
    output, maximum, summation = golden(torch, q, k, v, indices, 0.5, sparse_block_size=2,
                                       query_rope=qr, key_rope=kr, return_softmax_lse=True)
    torch.testing.assert_close(output[0, :, 0, 0], torch.tensor([1., 1.5], dtype=torch.float16))
    torch.testing.assert_close(maximum.flatten(), torch.tensor([32., 32.]))
    torch.testing.assert_close(summation.flatten(), torch.tensor([3., 4.]))


def test_arch22_source_closure_aot_only_and_graph_claim():
    prepared = IMPLEMENTATION.prepare_outputs(Path(__file__).resolve().parents[1])
    descriptor, recipe = prepared["descriptor"], prepared["recipe"]
    assert descriptor.graph_capability == GraphCapability.NOT_CAPTURABLE
    assert recipe.allow_jit_fallback is False and recipe.target_soc == "Ascend910B"
    assert recipe.package_op_type == "FiSparseFlashAttention"
    assert len(recipe.tensor_parameters) == 12 and len(recipe.attributes) == 9
    assert any("op_host/arch22/" in ref.locator for ref in recipe.headers)
    assert any("op_kernel/arch22/" in ref.locator for ref in recipe.headers)
    assert not any("arch35" in ref.locator for ref in recipe.headers)
    assert descriptor.capability_binding.domain == "operator"


def test_direct_attention_binding_validates_without_claiming_attention_plan_abi():
    from dataclasses import replace
    from flashinfer.attention.capability import validate_attention_kernel_bindings
    from flashinfer.runtime import SchemaError
    descriptor = IMPLEMENTATION.prepare_outputs(Path(__file__).resolve().parents[1])["descriptor"]
    assert validate_attention_kernel_bindings((), (descriptor,)) == 0
    with pytest.raises(SchemaError, match="requires a capability binding"):
        validate_attention_kernel_bindings((), (replace(descriptor, capability_binding=None),))
    broken = replace(descriptor.capability_binding, profile_fingerprint="0" * 64)
    with pytest.raises(SchemaError, match="binding mismatch"):
        validate_attention_kernel_bindings((), (replace(descriptor, capability_binding=broken),))
    wrong_mode = replace(descriptor, op="attention.single_prefill")
    with pytest.raises(SchemaError):
        validate_attention_kernel_bindings((), (wrong_mode,))


def test_framework_allocates_fresh_nd_outputs_without_replacing_supplied_out(monkeypatch):
    import torch_npu
    from flashinfer.operators import OutputSpec
    from flashinfer.operators.runtime import allocate_output
    calls = []
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: type("Output", (), {
        "device": torch.device("npu:4")})())
    allocated = object()
    monkeypatch.setattr(torch_npu, "npu_format_cast", lambda tensor, fmt: calls.append(fmt) or allocated)
    assert allocate_output(OutputSpec((2, 3, 4, 512), "float16", "npu:4")) is allocated
    assert calls == [2]


# The common test_operator_packages_real matrix discovers this operator's
# cases.py automatically: numerical/out/stream, missing/corrupt AOT and all
# three TASK_QUEUE_ENABLE settings need no additional registration here.


@pytest.fixture(scope="module")
def sparse_release(tmp_path_factory):
    from flashinfer.operators.pytest_support import prepare_release
    return prepare_release("sparse_flash_attention", tmp_path_factory.mktemp("sparse-extra-aot"))


@pytest.mark.real_compile
@pytest.mark.parametrize("configuration,pattern", [
    ({}, "unsorted_causal"),
    ({}, "repeated"),
    ({"empty": True}, "original"),
    ({"qlen": 5, "klen": 3}, "original"),
    ({"paged": True, "lse": True}, "original"),
])
def test_a2_sparse_addressing_empty_rows_and_paged_lse(sparse_release, configuration, pattern, tmp_path):
    script = r'''
import json,os,sys
import torch,torch_npu
from unittest.mock import patch
from flashinfer.operators.attention.sparse_flash_attention import cases
from flashinfer.operators.runtime import public_function
torch.npu.set_device(int(os.environ['FLASHINFER_TEST_NPU_DEVICE']))
configuration,pattern=json.loads(sys.argv[1])
args,options,expected=cases.sample(torch,torch_npu,**configuration)
if pattern != 'original':
    values = [18,0,17,2,-1,-1,-1,-1] if pattern == 'unsorted_causal' else [0,0,2,2,-1,-1,-1,-1]
    indices=torch.tensor(values,dtype=torch.int32).expand_as(args[3].cpu()).clone()
    args=(*args[:3],torch_npu.npu_format_cast(indices.npu(),2),args[4])
    expected=cases.golden(torch,*(t.cpu() for t in args[:4]),args[4],
        query_rope=options['query_rope'].cpu(),key_rope=options['key_rope'].cpu(),
        return_softmax_lse=True)
def forbidden(*args,**kwargs):
    raise AssertionError('AOT addressing acceptance attempted compilation')
with patch('flashinfer.jit.AscendCToolchain.discover',forbidden),patch('flashinfer.jit.AscendCBackend.compile',forbidden):
    outputs=public_function('sparse_flash_attention')(*args,**options)
    for actual,reference,(atol,rtol) in zip(outputs,expected,cases.TOLERANCES):
        assert torch_npu.get_npu_format(actual)==2
        torch.testing.assert_close(actual.cpu().double(),reference.double(),atol=atol,rtol=rtol)
print('EXTRA_SPARSE_A2_PASS',flush=True)
'''
    environment = dict(os.environ, **sparse_release, FLASHINFER_NPU_JIT_DISABLE="1")
    result = subprocess.run([sys.executable, "-B", "-c", script, json.dumps([configuration, pattern])],
                            env=environment, capture_output=True, text=True, timeout=300)
    (tmp_path / "sparse-edge.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "EXTRA_SPARSE_A2_PASS" in result.stdout
