"""ACLNN frontends are metadata-safe and keep native failures explicit."""

from pathlib import Path
import os
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run(code):
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                            env=dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0"),
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_aclnn_import_never_loads_provider_or_compiler():
    run('''
import importlib.abc,subprocess,sys
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'torch','torch_npu','acl','triton'}: raise AssertionError(name)
sys.meta_path.insert(0,Deny())
def forbidden(*args,**kwargs): raise AssertionError('compiler/subprocess')
subprocess.Popen=forbidden
import flashinfer.aclnn
assert flashinfer.aclnn.__all__ == []
assert not hasattr(flashinfer.aclnn, 'enable_aclnn')
import flashinfer.aclnn._native
import flashinfer.aclnn.build
import flashinfer.aclnn._ops
''')


def test_aclnn_registration_and_fake_execution_do_not_load_native():
    run('''
import sys,torch
import flashinfer.norm
from torch._subclasses.fake_tensor import FakeTensorMode
from flashinfer.aclnn import _native
from flashinfer.aclnn._ops import ensure_registered
def forbidden(): raise AssertionError('native execution from FakeTensor')
_native.load_adapter=forbidden
ensure_registered();ensure_registered()
assert flashinfer.norm._torch_library_ops is None, 'production registration must not reroute unrelated norm ops'
assert 'torch_npu' not in sys.modules
with FakeTensorMode():
 a=torch.empty((16,48),dtype=torch.bfloat16)
 b=torch.empty((48,32),dtype=torch.bfloat16)
 bias=torch.empty((32,),dtype=torch.bfloat16)
 result=torch.ops.flashinfer.matmul_bias_silu(a,b,bias)
 assert result.shape==(16,32) and result.dtype==a.dtype
 assert torch.ops.flashinfer.matmul_bias_silu_out(a,b,bias,result) is None
 for dtype in (torch.float16,torch.bfloat16):
  q=torch.empty((128,2,128),dtype=dtype)
  kv=torch.empty((256,2,128),dtype=dtype)
  out=torch.ops.flashinfer.single_prefill_with_kv_cache_catlass(q,kv,kv)
  assert out.shape==q.shape and out.dtype==q.dtype
  assert torch.ops.flashinfer.single_prefill_with_kv_cache_catlass_out(q,kv,kv,out) is None
  try: torch.ops.flashinfer.single_prefill_with_kv_cache_catlass(q,kv,kv,causal=True)
  except (ValueError,RuntimeError): pass
  else: raise AssertionError('unsupported causal mode accepted')
  other=torch.bfloat16 if dtype==torch.float16 else torch.float16
  try: torch.ops.flashinfer.single_prefill_with_kv_cache_catlass(q,kv.to(other),kv)
  except (ValueError,RuntimeError): pass
  else: raise AssertionError('mixed dtype accepted by fake implementation')
assert 'torch_npu' not in sys.modules
''')


def test_public_call_registers_aclnn_automatically():
    run('''
import torch,flashinfer
from torch._subclasses.fake_tensor import FakeTensorMode
from flashinfer.aclnn import _native,_ops
assert not _ops._registered
def forbidden(): raise AssertionError('FakeTensor must not load the native bridge')
_native.load_adapter=forbidden
with FakeTensorMode():
 a=torch.empty((16,16),dtype=torch.bfloat16)
 bias=torch.empty((16,),dtype=torch.bfloat16)
 assert flashinfer.matmul_bias_silu(a,a,bias).shape==(16,16)
 assert _ops._registered
 out=torch.empty_like(a)
 assert flashinfer.matmul_bias_silu(a,a,bias,out=out) is out
''')


def test_aclnn_has_no_cpu_numerical_fallback():
    run('''
import torch
from flashinfer.aclnn._ops import ensure_registered
ensure_registered()
try:
 torch.ops.flashinfer.matmul_bias_silu(torch.zeros((16,16),dtype=torch.bfloat16),torch.zeros((16,16),dtype=torch.bfloat16),torch.zeros(16,dtype=torch.bfloat16))
except NotImplementedError: pass
else: raise AssertionError('CPU fallback must not run')
''')


def test_present_native_adapter_digest_failure_is_not_fallback():
    run('''
import json,os,sys,tempfile,types
from pathlib import Path
import torch
from flashinfer.aclnn import _native
sys.modules['torch_npu']=types.SimpleNamespace(__version__='fixture')
with tempfile.TemporaryDirectory() as temporary:
 root=Path(temporary);os.environ['FLASHINFER_ACLNN_ADAPTER_PATH']=temporary
 (root/'libflashinfer_aclnn.so').write_bytes(b'invalid fixture ELF')
 (root/'manifest.json').write_text(json.dumps(dict(schema_version=1,library='libflashinfer_aclnn.so',torch_version=torch.__version__,torch_npu_version='fixture',sha256='0'*64)))
 assert _native.available()
 try: _native.load_adapter()
 except RuntimeError as error: assert 'digest mismatch' in str(error)
 else: raise AssertionError('corrupt native adapter accepted')
''')


def test_selected_missing_native_adapter_fails_explicitly():
    run('''
import os,sys,tempfile,types
from flashinfer.aclnn import _native
sys.modules['torch_npu']=types.SimpleNamespace(__version__='fixture')
os.environ.pop('FLASHINFER_ACLNN_ADAPTER_PATH',None)
assert _native.available(), 'the native Torch bridge is the only execution path'
with tempfile.TemporaryDirectory() as temporary:
 os.environ['FLASHINFER_ACLNN_ADAPTER_PATH']=temporary
 try: _native.load_adapter()
 except RuntimeError as error: assert 'not bundled' in str(error)
 else: raise AssertionError('missing native adapter accepted')
''')


def test_python_facades_register_and_use_the_same_torch_fake_operators():
    run('''
import sys,torch,flashinfer
from torch._subclasses.fake_tensor import FakeTensorMode
from flashinfer.aclnn import _native
def forbidden(*args,**kwargs): raise AssertionError('fake execution loaded native code')
_native.load_adapter=forbidden
assert not hasattr(torch.ops.flashinfer,'matmul_bias_silu')
with FakeTensorMode(allow_fallback_kernels=False):
 a=torch.empty((16,48),dtype=torch.bfloat16)
 b=torch.empty((48,32),dtype=torch.bfloat16)
 bias=torch.empty(32,dtype=torch.bfloat16)
 result=flashinfer.matmul_bias_silu(a,b,bias)
 direct=torch.ops.flashinfer.matmul_bias_silu(a,b,bias)
 assert result.shape==direct.shape==(16,32)
 assert result.dtype==direct.dtype==torch.bfloat16
 assert flashinfer.matmul_bias_silu(a,b,bias,out=result) is result
 for dtype in (torch.float16,torch.bfloat16):
  q=torch.empty((128,2,128),dtype=dtype)
  kv=torch.empty((256,2,128),dtype=dtype)
  result=flashinfer.single_prefill_with_kv_cache_catlass(q,kv,kv)
  direct=torch.ops.flashinfer.single_prefill_with_kv_cache_catlass(q,kv,kv)
  assert result.shape==direct.shape==q.shape and result.dtype==direct.dtype==dtype
  assert flashinfer.single_prefill_with_kv_cache_catlass(q,kv,kv,out=result) is result
assert 'torch_npu' not in sys.modules
assert flashinfer.norm._torch_library_ops is None
''')


def test_python_facade_has_no_cpu_numerical_fallback():
    run('''
import torch,flashinfer
for call,args in (
 (flashinfer.matmul_bias_silu,(torch.zeros((16,16),dtype=torch.bfloat16),)*2+(torch.zeros(16,dtype=torch.bfloat16),)),
 (flashinfer.single_prefill_with_kv_cache_catlass,(torch.zeros((128,1,128),dtype=torch.float16),)*3),
):
 try: call(*args)
 except NotImplementedError: pass
 else: raise AssertionError('Python facade bypassed the Torch device dispatcher')
''')


def test_prebuilt_bridge_bundling_is_compiler_free_and_keeps_source_identity(tmp_path, monkeypatch):
    import hashlib
    import json
    from flashinfer.aclnn import _native
    from flashinfer.aclnn.build import bundle_adapter

    source = ROOT / "csrc/aclnn_torch_adapter/adapter.cpp"
    supplied = tmp_path / "supplied"
    supplied.mkdir()
    library = supplied / "libflashinfer_aclnn.so"
    manifest = supplied / "manifest.json"
    library.write_bytes(b"verified fixture bytes")
    manifest.write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())))
    monkeypatch.setattr(_native, "verified_adapter", lambda directory: (library, manifest))

    def forbidden(*args, **kwargs):
        raise AssertionError("prebuilt bundling invoked a compiler or loaded a library")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    import ctypes
    monkeypatch.setattr(ctypes, "CDLL", forbidden)
    outputs = bundle_adapter(tmp_path / "bundled", directory=supplied)
    assert {path.name for path in outputs} == {library.name, manifest.name}
    assert all(path.read_bytes() == (supplied / path.name).read_bytes() for path in outputs)
    manifest.write_text(json.dumps(dict(source_sha256="0" * 64)))
    with pytest.raises(RuntimeError, match="source identity mismatch"):
        bundle_adapter(tmp_path / "rejected", directory=supplied)
    assert not (tmp_path / "rejected").exists()
