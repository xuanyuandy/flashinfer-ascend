"""Execute an exported standard op-package via ACLNN, optionally denying exec.

No torch, CATLASS, source tree, JIT cache or compiler is needed by this process.
The fixed 16x16 FP32 probe checks all 256 values against an independent CPU GEMM.
"""

import argparse
import ctypes as C
import json
import os
from pathlib import Path
import sys


def deny_compilers(cann_root):
    """Linux seccomp: deny *all* execve/execveat, including absolute compilers.

    This applies only to this test process and its children, needs no privilege,
    and changes neither the SDK nor the system's mount/filesystem state.
    """
    import errno
    import shutil
    import subprocess

    os.environ["PATH"] = "/usr/bin:/bin"
    assert shutil.which("bisheng") is None and shutil.which("ccec") is None
    seccomp = C.CDLL("libseccomp.so.2")
    seccomp.seccomp_init.argtypes, seccomp.seccomp_init.restype = [C.c_uint32], C.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [C.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype = C.c_int
    seccomp.seccomp_rule_add.argtypes = [C.c_void_p, C.c_uint32, C.c_int, C.c_uint]
    seccomp.seccomp_rule_add.restype = C.c_int
    seccomp.seccomp_load.argtypes, seccomp.seccomp_load.restype = [C.c_void_p], C.c_int
    seccomp.seccomp_release.argtypes, seccomp.seccomp_release.restype = [C.c_void_p], None
    context = seccomp.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
    if not context:
        raise RuntimeError("seccomp_init failed")
    try:
        for syscall in (b"execve", b"execveat"):
            number = seccomp.seccomp_syscall_resolve_name(syscall)
            if number < 0 or seccomp.seccomp_rule_add(context, 0x50000 | errno.EPERM, number, 0):
                raise RuntimeError("could not deny " + syscall.decode())
        if seccomp.seccomp_load(context):
            raise RuntimeError("seccomp_load failed")
    finally:
        seccomp.seccomp_release(context)
    try:
        subprocess.run([str(cann_root / "bin/bisheng"), "--version"], check=True, capture_output=True)
    except PermissionError as error:
        assert error.errno == errno.EPERM
        print("COMPILER_BLOCKED: absolute bisheng --version -> PermissionError errno=1", flush=True)
    else:
        raise AssertionError("compiler was executable under the guard")

    def forbidden(*args, **kwargs):
        raise AssertionError("AOT runtime attempted compiler discovery/compilation/subprocess")

    subprocess.Popen = forbidden
    from flashinfer.jit import AscendCBackend, AscendCToolchain

    AscendCToolchain.discover = forbidden
    AscendCBackend.compile = forbidden
    print("COMPILER_GUARD: PATH bisheng=None ccec=None; execve/execveat=EPERM; discovery/JIT=forbidden", flush=True)


def execute(module, cann_root):
    acl = C.CDLL(str(cann_root / "lib64/libascendcl.so"), mode=C.RTLD_GLOBAL)
    prototypes = {
        "aclInit": [C.c_char_p], "aclFinalize": [],
        "aclrtSetDevice": [C.c_int], "aclrtResetDevice": [C.c_int],
        "aclrtCreateStream": [C.POINTER(C.c_void_p)], "aclrtDestroyStream": [C.c_void_p],
        "aclrtSynchronizeStream": [C.c_void_p],
        "aclrtMalloc": [C.POINTER(C.c_void_p), C.c_size_t, C.c_int], "aclrtFree": [C.c_void_p],
        "aclrtMemcpy": [C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t, C.c_int],
    }
    for name, arguments in prototypes.items():
        function = getattr(acl, name)
        function.argtypes, function.restype = arguments, C.c_int

    def call(name, *arguments):
        status = getattr(acl, name)(*arguments)
        if status:
            raise RuntimeError("%s returned %s" % (name, status))

    base = C.CDLL(str(cann_root / "lib64/libnnopbase.so"), mode=C.RTLD_GLOBAL)
    base.aclCreateTensor.argtypes = [C.POINTER(C.c_int64), C.c_uint64, C.c_int,
        C.POINTER(C.c_int64), C.c_int64, C.c_int, C.POINTER(C.c_int64), C.c_uint64, C.c_void_p]
    base.aclCreateTensor.restype = C.c_void_p
    base.aclDestroyTensor.argtypes, base.aclDestroyTensor.restype = [C.c_void_p], C.c_int
    stream, workspace = C.c_void_p(), C.c_void_p()
    allocations, tensors = [], []
    call("aclInit", None)
    try:
        call("aclrtSetDevice", 0)
        call("aclrtCreateStream", C.byref(stream))
        array = C.c_float * 256
        ha = array(*(float(i % 7 - 3) for i in range(256)))
        hb = array(*(float(i % 5 - 2) for i in range(256)))
        hc = array(*([float("nan")] * 256))
        dims, strides = (C.c_int64 * 2)(16, 16), (C.c_int64 * 2)(16, 1)
        for host in (ha, hb, hc):
            pointer = C.c_void_p()
            call("aclrtMalloc", C.byref(pointer), C.sizeof(host), 0)
            allocations.append(pointer)
            call("aclrtMemcpy", pointer, C.sizeof(host), C.cast(host, C.c_void_p), C.sizeof(host), 1)
            tensor = base.aclCreateTensor(dims, 2, 0, strides, 0, 2, dims, 2, pointer)
            if not tensor:
                raise RuntimeError("aclCreateTensor failed")
            tensors.append(tensor)
        workspace_bytes, executor = module.get_workspace_size(*tensors)
        print("GetWorkspaceSize=0 bytes=" + str(workspace_bytes), flush=True)
        if workspace_bytes:
            call("aclrtMalloc", C.byref(workspace), workspace_bytes, 0)
        module.launch(workspace, workspace_bytes, executor, stream)
        print(module.recipe.host_entry + "=0", flush=True)
        call("aclrtSynchronizeStream", stream)
        call("aclrtMemcpy", C.cast(hc, C.c_void_p), C.sizeof(hc), allocations[2], C.sizeof(hc), 2)
        expected = [sum(ha[m * 16 + k] * hb[k * 16 + n] for k in range(16))
                    for m in range(16) for n in range(16)]
        if not all(hc[i] == expected[i] for i in range(256)):
            raise AssertionError("ACLNN NPU output differs from independent CPU GEMM")
        return dict(shape=[16, 16, 16], device=0, stream_sync=0,
            max_abs_error=max(abs(hc[i] - expected[i]) for i in range(256)), first8=list(hc)[:8])
    finally:
        for tensor in reversed(tensors):
            if base.aclDestroyTensor(tensor):
                raise RuntimeError("aclDestroyTensor failed")
        if workspace.value:
            call("aclrtFree", workspace)
        for pointer in reversed(allocations):
            call("aclrtFree", pointer)
        if stream.value:
            call("aclrtDestroyStream", stream)
        call("aclrtResetDevice", 0)
        call("aclFinalize")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True, help="verified exported generation directory")
    parser.add_argument("--soc", default="ascend950")
    parser.add_argument("--cann-root", type=Path, default=os.environ.get("ASCEND_HOME_PATH"))
    parser.add_argument("--deny-compilers", action="store_true")
    args = parser.parse_args()
    if args.deny_compilers:
        deny_compilers(args.cann_root)
    from flashinfer.jit.op_package import load_op_package

    module = load_op_package(args.package, target_soc=args.soc)
    print("LOAD_OK: ascendc_op_package entry=" + module.recipe.host_entry, flush=True)
    result = execute(module, args.cann_root)
    assert "torch_npu" not in sys.modules and "triton" not in sys.modules
    print("PASS", json.dumps(dict(implementation=module.recipe.implementation,
        compilers_blocked=args.deny_compilers, **result), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
