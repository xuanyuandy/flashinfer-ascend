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
    from flashinfer._resources import data_root

    return Path(os.environ.get("FLASHINFER_ACLNN_ADAPTER_PATH", data_root() / "aclnn"))


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
        raise RuntimeError(
            "ACLNN Torch adapter is not bundled; build a normal AOT wheel or run "
            "python -m flashinfer.aclnn.build --output-dir flashinfer/data/aclnn"
        )
    if manifest.is_symlink():
        raise RuntimeError("ACLNN Torch adapter manifest must not be a symlink")
    record = json.loads(manifest.read_text())
    if (
        record.get("schema_version") != 1
        or record.get("library") != "libflashinfer_aclnn.so"
        or record.get("torch_version") != torch.__version__
        or record.get("torch_npu_version") != torch_npu.__version__
    ):
        raise RuntimeError(
            "ACLNN Torch adapter Torch/torch_npu build identity mismatch"
        )
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
    api.fi_aclnn_register.argtypes = [
        C.c_char_p,
        C.c_char_p,
        C.c_void_p,
        C.c_void_p,
        C.c_uint64,
        C.c_uint64,
    ]
    api.fi_aclnn_register.restype = C.c_int
    api.fi_aclnn_last_error.argtypes = []
    api.fi_aclnn_last_error.restype = C.c_char_p
    if not hasattr(api, "fi_aclnn_last_workspace_size"):
        raise RuntimeError(
            "ACLNN Torch adapter is outdated; rebuild the adapter or the AOT wheel"
        )
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
            workspace = C.cast(
                getattr(module.api, recipe.host_entry + "GetWorkspaceSize"), C.c_void_p
            )
            launch = C.cast(getattr(module.api, recipe.host_entry), C.c_void_p)
            if recipe.extended_aclnn:
                if not hasattr(api, "fi_aclnn_register_extended"):
                    raise RuntimeError(
                        "ACLNN Torch adapter lacks the declared extended ABI; rebuild the adapter"
                    )
                integer_pointer = C.POINTER(C.c_int64)
                api.fi_aclnn_register_extended.argtypes = [
                    C.c_char_p,
                    C.c_char_p,
                    C.c_void_p,
                    C.c_void_p,
                    C.c_uint64,
                    C.c_uint64,
                    integer_pointer,
                    integer_pointer,
                    integer_pointer,
                    integer_pointer,
                    C.c_uint64,
                ]
                api.fi_aclnn_register_extended.restype = C.c_int
                array = lambda values: (C.c_int64 * len(values))(*values)
                formats = array(
                    [
                        sum({"ND": 1, "NZ": 2}[f] for f in p.physical_layouts)
                        for p in recipe.tensor_parameters
                    ]
                )
                conversions = array(
                    [
                        next(
                            (
                                {"int4": 1, "uint64": 2}[c.native_dtype]
                                for c in p.native_conversions
                            ),
                            0,
                        )
                        for p in recipe.tensor_parameters
                    ]
                )
                optional = array([p.optional for p in recipe.tensor_parameters])
                has_strings = any(p.dtype == "string" for p in recipe.attributes)
                if has_strings and not hasattr(
                    torch.ops.flashinfer_aclnn, "_execute_with_strings"
                ):
                    raise RuntimeError(
                        "ACLNN Torch adapter lacks string attributes; rebuild the adapter"
                    )
                attribute_types = array(
                    [
                        {"bool": 0, "int64": 1, "float64": 2, "string": 3}[p.dtype]
                        for p in recipe.attributes
                    ]
                )
                status = api.fi_aclnn_register_extended(
                    token.encode(),
                    recipe.host_entry.encode(),
                    workspace,
                    launch,
                    inputs,
                    outputs,
                    formats,
                    conversions,
                    optional,
                    attribute_types,
                    len(recipe.attributes),
                )
                operation = (
                    torch.ops.flashinfer_aclnn._execute_with_strings
                    if has_strings
                    else torch.ops.flashinfer_aclnn._execute_extended
                )
            else:
                status = api.fi_aclnn_register(
                    token.encode(),
                    recipe.host_entry.encode(),
                    workspace,
                    launch,
                    inputs,
                    outputs,
                )
                operation = torch.ops.flashinfer_aclnn._execute
            if status:
                raise RuntimeError(api.fi_aclnn_last_error().decode())
            _owners[token] = module

            def run(tensors, attributes=()):
                args = (token, list(tensors[:inputs]), list(tensors[inputs:]))
                return (
                    operation(*args, list(attributes))
                    if recipe.extended_aclnn
                    else operation(*args)
                )

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
