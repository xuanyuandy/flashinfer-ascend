"""Private, explicitly invoked compiler worker. Never imported by flashinfer."""

import hashlib
import json
import linecache
from pathlib import Path
import shutil
import sys
from types import ModuleType


def main(request_path: Path) -> None:
    # All optional imports and driver/compiler discovery happen inside the worker.
    import torch
    import torch_npu
    import triton
    from triton.compiler import ASTSource
    from triton.runtime import driver

    from flashinfer.jit.core import JitSpec
    from flashinfer.runtime.hardware import _normalize_soc_model
    from flashinfer.jit._triton_abi import triton_binary_mode, verify_triton_symbols

    directory = request_path.parent
    request = json.loads(request_path.read_text())
    spec = JitSpec.from_dict(request["spec"])
    config = request["config"]
    if hashlib.sha256((directory / "source.py").read_bytes()).hexdigest() != spec.source.digest:
        raise ValueError("source snapshot digest mismatch")
    target = driver.active.get_current_target()
    if target.backend != "npu" or spec.target_soc not in (target.arch, _normalize_soc_model(target.arch)):
        raise ValueError("declared SOC %s does not match active target %s" % (spec.target_soc, target))
    # Stable source locations keep diagnostic text IR reproducible across
    # private staging directories. inspect (used by @triton.jit) reads this
    # exact verified snapshot from the worker-local linecache, never a mutable
    # file at the original source path.
    filename = "/__flashinfer_jit__/%s/%s" % (spec.source.digest, spec.source.locator)
    text = (directory / "source.py").read_text(encoding="utf-8")
    linecache.cache[filename] = (len(text), None, text.splitlines(keepends=True), filename)
    module = ModuleType("_flashinfer_triton_source")
    module.__file__ = filename
    sys.modules[module.__name__] = module
    exec(compile(text, filename, "exec"), module.__dict__)
    entry = spec.entry_points[0]
    function = getattr(module, entry, None)
    if not isinstance(function, triton.runtime.JITFunction):
        raise ValueError("entry symbol %r is not a @triton.jit function in %s" % (entry, spec.source.locator))
    signature = dict(config["signature"])
    constants = config.get("constexprs", {})
    if set(signature) & set(constants) or set(signature) | set(constants) != set(function.arg_names):
        raise ValueError("signature and constexprs must partition the source function's arguments")
    if {function.arg_names[i] for i in function.constexprs} != set(constants):
        raise ValueError("constexprs must exactly match the function's tl.constexpr arguments")
    signature.update({key: "constexpr" for key in constants})
    # The Ascend launcher consumes dictionary order when generating its Python
    # argument parser. JSON object order is not the kernel's launch ABI.
    signature = {name: signature[name] for name in function.arg_names}
    source = ASTSource(function, signature=signature, constexprs=constants)
    kernel = triton.compile(source, target=target, options=config.get("options", {}))
    if kernel.name != entry or kernel.metadata.kernel_name != entry:
        raise ValueError("compiler exported %r/%r instead of declared entry %r"
                         % (kernel.name, kernel.metadata.kernel_name, entry))
    binary_mode = triton_binary_mode(config)
    if ((binary_mode == "mix") != (kernel.metadata.mix_mode == "mix")):
        raise ValueError("Triton binary declaration does not match compiler mix_mode")
    verify_triton_symbols(kernel.kernel, entry, binary_mode)
    # Prebuild both CPU runtime extensions under the same per-spec build lock.
    # This loads the device kernel for validation, without launching it.
    kernel._init_handles()
    if not kernel.module or not kernel.function:
        raise RuntimeError("Triton did not resolve the compiled device kernel")
    output = directory / spec.output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(kernel.kernel)
    group = {name: str(Path(path).relative_to(directory))
             for name, path in kernel.metadata_group.items()}
    # Keep native filenames for CompiledKernel (it interprets their suffixes).
    # The declared output may have any normalized relative path/suffix.
    # MLIR bytecode serialization is nondeterministic
    # on this toolchain (verified by two real builds with identical text IR and
    # npubin). It is a compiler intermediate, never a loader input. Discard it
    # after compilation rather than changing the immutable cache-record rules.
    for name in tuple(group):
        if name.endswith(".mlirbc"):
            (directory / group.pop(name)).unlink()
    utils_path = driver.active.utils.get_so_path()
    launcher_path = kernel.run.get_launcher_so_path()
    versions = {"triton": triton.__version__, "torch": torch.__version__,
                "torch_npu": torch_npu.__version__, "python_abi": sys.implementation.cache_tag}
    (directory / "triton.json").write_text(json.dumps({
        "schema_version": 1, "spec_fingerprint": spec.fingerprint,
        "entry_point": entry, "hash": kernel.hash,
        "binary_mode": binary_mode,
        "target": {"backend": target.backend, "arch": target.arch, "warp_size": target.warp_size},
        "versions": versions, "arg_names": function.arg_names,
        "signature": signature, "constexprs": constants,
        "metadata_group": group,
        "npu_utils": str(Path(utils_path).relative_to(directory)),
        "launcher": str(Path(launcher_path).relative_to(directory)),
    }, sort_keys=True), encoding="utf-8")
    # Triton's group index contains staging-absolute paths. The loader uses the
    # checked relative group above; never retain a dangling native index.
    for path in (directory / "triton-cache").rglob("__grp__*.json"):
        path.unlink()
    shutil.rmtree(directory / "__pycache__", ignore_errors=True)
    print("compiled", spec.kernel_id, "entry=", entry, "target=", target,
          "npubin_bytes=", len(kernel.kernel), "resolved=", bool(kernel.function), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
