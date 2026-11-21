"""Sparse attention AOT-only execution, compiler and storage-format guards."""

import json
import os
import subprocess
import sys

import pytest
from flashinfer.operators.pytest_support import aot_params, prepare_release


@pytest.fixture(scope="module", params=aot_params("attention.sparse_flash_attention"))
def sparse_release(request, tmp_path_factory):
    impl = request.param
    release = prepare_release(impl.op, tmp_path_factory.mktemp("sparse-extra-aot"), kernel_id=impl.kernel_id)
    return impl.kernel_id, release


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
torch.npu.set_device(int(os.environ.get('FLASHINFER_TEST_NPU_DEVICE', '0')))
torch.npu.config.allow_internal_format=True
configuration,pattern,kernel_id=json.loads(sys.argv[1])
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
    outputs=public_function('sparse_flash_attention')(*args,kernel_id=kernel_id,**options)
    for actual,reference,(atol,rtol) in zip(outputs,expected,cases.TOLERANCES):
        assert torch_npu.get_npu_format(actual)==2
        torch.testing.assert_close(actual.cpu().double(),reference.double(),atol=atol,rtol=rtol)
print('EXTRA_SPARSE_A2_PASS',flush=True)
'''
    kernel_id, release = sparse_release
    environment = dict(os.environ, **release, FLASHINFER_NPU_JIT_DISABLE="1")
    result = subprocess.run([sys.executable, "-B", "-c", script, json.dumps([configuration, pattern, kernel_id])],
                            env=environment, capture_output=True, text=True, timeout=300)
    (tmp_path / "sparse-edge.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "EXTRA_SPARSE_A2_PASS" in result.stdout
