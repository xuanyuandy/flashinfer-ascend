"""Compiler-free loading and calls into the verified Torch/ACLNN adapter."""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from threading import RLock

_lock = RLock()
_owners = {}
_runners = {}


def adapter_directory():
    return Path(os.environ.get("FLASHINFER_ACLNN_ADAPTER_PATH",
                              Path(__file__).resolve().parents[1] / "data/aclnn"))


def available():
    """Compatibility query: native Torch execution is always required.

    This does not probe whether the library is present; loading validates it.
    """
    return True


def verified_adapter(directory=None):
    """Verify prebuilt bytes and build identity without loading the library."""
    import torch
    import torch_npu

    directory = adapter_directory() if directory is None else Path(directory)
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        raise RuntimeError("ACLNN Torch adapter is not bundled; build a normal AOT wheel or run "
                           "python -m flashinfer.aclnn.build --output-dir flashinfer/data/aclnn")
    if manifest.is_symlink():
        raise RuntimeError("ACLNN Torch adapter manifest must not be a symlink")
    record = json.loads(manifest.read_text())
    if (record.get("schema_version") != 1 or record.get("library") != "libflashinfer_aclnn.so"
            or record.get("torch_version") != torch.__version__
            or record.get("torch_npu_version") != torch_npu.__version__):
        raise RuntimeError("ACLNN Torch adapter Torch/torch_npu build identity mismatch")
    library = directory / record["library"]
    if library.is_symlink():
        raise RuntimeError("ACLNN Torch adapter library must not be a symlink")
    if hashlib.sha256(library.read_bytes()).hexdigest() != record["sha256"]:
        raise RuntimeError("ACLNN Torch adapter library digest mismatch")
    return library, manifest


@lru_cache(maxsize=1)
def load_adapter():
    import ctypes as C
    import torch

    library, _ = verified_adapter()
    torch.ops.load_library(str(library))
    api = C.CDLL(str(library))
    api.fi_aclnn_register.argtypes = [C.c_char_p, C.c_char_p, C.c_void_p, C.c_void_p,
                                    C.c_uint64, C.c_uint64]
    api.fi_aclnn_register.restype = C.c_int
    api.fi_aclnn_last_error.argtypes = []
    api.fi_aclnn_last_error.restype = C.c_char_p
    if not hasattr(api, "fi_aclnn_last_workspace_size"):
        raise RuntimeError("ACLNN Torch adapter is outdated; rebuild the adapter or the AOT wheel")
    api.fi_aclnn_last_workspace_size.argtypes = [C.c_char_p, C.POINTER(C.c_uint64)]
    api.fi_aclnn_last_workspace_size.restype = C.c_int
    return api


def prepare(module, recipe):
    """Bind one verified module once; invocation only supplies Tensor lists."""
    import ctypes as C
    import torch

    inputs = sum(parameter.role == "input" for parameter in recipe.tensor_parameters)
    outputs = len(recipe.tensor_parameters) - inputs
    token = recipe.fingerprint + ":" + str(id(module.api))
    with _lock:
        if token not in _runners:
            api = load_adapter()
            workspace = C.cast(getattr(module.api, recipe.host_entry + "GetWorkspaceSize"), C.c_void_p)
            launch = C.cast(getattr(module.api, recipe.host_entry), C.c_void_p)
            status = api.fi_aclnn_register(token.encode(), recipe.host_entry.encode(),
                                           workspace, launch, inputs, outputs)
            if status:
                raise RuntimeError(api.fi_aclnn_last_error().decode())
            _owners[token] = module
            operation = torch.ops.flashinfer_aclnn._execute

            def run(tensors):
                return operation(token, list(tensors[:inputs]), list(tensors[inputs:]))

            _runners[token] = run
        return _runners[token]


def execute(module, recipe, tensors):
    return prepare(module, recipe)(tensors)


def last_workspace_size(module, recipe):
    """Observe the last real GetWorkspaceSize result of a prepared binding."""
    import ctypes as C

    api = load_adapter()
    size = C.c_uint64()
    token = recipe.fingerprint + ":" + str(id(module.api))
    if api.fi_aclnn_last_workspace_size(token.encode(), C.byref(size)):
        raise RuntimeError(api.fi_aclnn_last_error().decode())
    return size.value
