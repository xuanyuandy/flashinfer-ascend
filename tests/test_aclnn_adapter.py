"""ACLNN frontends are metadata-safe and keep native failures explicit."""

from pathlib import Path
import os
import subprocess
import sys


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
import flashinfer.aclnn._native
import flashinfer.aclnn.build
import flashinfer.aclnn._ops
''')


def test_aclnn_registration_and_fake_execution_do_not_load_native():
    run('''
import sys,torch
from torch._subclasses.fake_tensor import FakeTensorMode
from flashinfer.aclnn import enable_aclnn,_native
def forbidden(): raise AssertionError('native execution from FakeTensor')
_native.load_adapter=forbidden
enable_aclnn();enable_aclnn()
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


def test_aclnn_has_no_cpu_numerical_fallback():
    run('''
import torch
from flashinfer.aclnn import enable_aclnn
enable_aclnn()
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
assert not _native.available(), 'bundled native library must not change the default Python path'
with tempfile.TemporaryDirectory() as temporary:
 os.environ['FLASHINFER_ACLNN_ADAPTER_PATH']=temporary
 _native.require_native()
 try: _native.load_adapter()
 except RuntimeError as error: assert 'not bundled' in str(error)
 else: raise AssertionError('missing native adapter accepted')
''')
