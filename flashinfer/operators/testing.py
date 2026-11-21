"""Common numerical/AOT/out/stream/installed acceptance using author-owned cases."""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

from .catalog import cases_module, definition
from .definition import OutputSpec
from .runtime import allocate_output, public_function


def _outputs(result, count):
    outputs = (result,) if count == 1 else tuple(result)
    if len(outputs) != count:
        raise AssertionError("operator returned the wrong number of outputs")
    return outputs


def vendor_directory(match, aliases):
    vendor = match.directory / "packages" / "vendors" / match.recipe.package_vendor
    if "." not in str(vendor):
        return str(vendor)
    # CANN 9.2 truncates binary paths at the first dot, including python3.11.
    alias = aliases / match.spec.fingerprint / match.recipe.package_vendor
    alias.parent.mkdir(parents=True, exist_ok=True)
    try:
        alias.symlink_to(vendor, target_is_directory=True)
    except FileExistsError:
        if not alias.is_symlink() or alias.resolve() != vendor.resolve():
            raise
    return str(alias)


def vendor_search_path(vendors, *, replace=()):
    inherited = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
    paths = [*vendors, *(p for p in inherited if p and Path(p).name not in replace)]
    return os.pathsep.join(dict.fromkeys(paths))


def case_parameters(cases, kernel_id):
    """Use an author's implementation-specific input matrix when it is provided."""
    select = getattr(cases, "parameters_for", None)
    parameters = tuple(select(kernel_id) if kernel_id is not None and select else cases.CASE_PARAMETERS)
    if not parameters:
        raise AssertionError(f"no numerical inputs for {kernel_id or cases.__name__}")
    return parameters


def numerical_cases(torch, torch_npu, name, *, first_only=False, kernel_id=None, backend=None):
    operator = definition(name)
    cases = cases_module(name)
    parameters = case_parameters(cases, kernel_id)
    output_count = len(cases.TOLERANCES)
    if not output_count or (operator.schema and output_count != len(operator.output_names)):
        raise AssertionError("numerical cases and output tolerances must be explicit and nonempty")
    call = public_function(name)
    observations = {}
    for index, configuration in enumerate(parameters[:1] if first_only else parameters):
        args, options, expected = cases.sample(torch, torch_npu, **configuration)
        if kernel_id is not None:
            options = dict(options, kernel_id=kernel_id)
        if backend is not None:
            options = dict(options, backend=backend)
        originals = tuple(t for t in tuple(args) + tuple(options.values()) if isinstance(t, torch.Tensor))
        pointers = tuple(t.data_ptr() for t in originals)
        expected = _outputs(expected, output_count)
        outputs = _outputs(call(*args, **options), output_count)
        if hasattr(cases, "verify_effect"):
            cases.verify_effect(torch, call, args, options, outputs)
        supplied = tuple(allocate_output(OutputSpec(tuple(t.shape), str(t.dtype).removeprefix("torch."), t.device))
                         for t in outputs)
        output_pointers = tuple(t.data_ptr() for t in supplied)
        returned = _outputs(call(*args, out=supplied[0] if len(supplied) == 1 else supplied, **options), len(supplied))
        assert all(a is b for a, b in zip(returned, supplied)), "out identity changed"
        assert output_pointers == tuple(t.data_ptr() for t in supplied), "out storage changed"
        assert pointers == tuple(t.data_ptr() for t in originals), "input storage changed"
        errors = []
        for actual, reference, (atol, rtol) in zip(outputs, expected, cases.TOLERANCES):
            assert actual.shape == reference.shape, "reference output shape differs"
            errors.append(float((actual.cpu().double() - reference.double()).abs().max())
                          if actual.numel() else 0.0)
        print(json.dumps(dict(case=configuration, max_abs=errors), sort_keys=True), flush=True)
        for output_tuple in (outputs, returned):
            for actual, reference, (atol, rtol) in zip(output_tuple, expected, cases.TOLERANCES):
                torch.testing.assert_close(actual.cpu().double(), reference.double(), atol=atol, rtol=rtol)
        observations[str(index)] = dict(case=configuration, max_abs=errors)
    return observations


def selftest_case(torch, name, *, kernel_id=None):
    import torch_npu
    torch.npu.config.allow_internal_format = True
    observations = numerical_cases(torch, torch_npu, name, first_only=True, kernel_id=kernel_id)
    return "author CPU reference matched; max_abs=" + repr(observations["0"]["max_abs"])


def target_aot_implementations(name):
    """All production AOT implementations of the requested operator on this device."""
    from flashinfer.runtime import current_hardware
    from .implementation import implementations

    operator = definition(name)
    soc = current_hardware().soc_model
    return tuple(
        d for d in implementations(backend="ascendc")
        if d.op == operator.op and d.bind().target_soc == soc
    )


def select_aot_implementation(name, kernel_id=None):
    choices = target_aot_implementations(name)
    for implementation in choices:
        if kernel_id == implementation.kernel_id:
            return implementation
    if kernel_id is None and len(choices) == 1:
        return choices[0]
    ids = ", ".join(d.kernel_id for d in choices) or "(none)"
    raise ValueError(f"{name}: select a local --kernel-id from {ids}; requested {kernel_id!r}")


def local_vendor_names(name):
    from flashinfer.ascendc import load_contract
    return {load_contract(impl.kernel_id)[1].package_vendor for impl in target_aot_implementations(name)}


def run(name, mode="matrix", *, kernel_id=None):
    import torch
    import torch_npu
    import flashinfer
    from flashinfer.ascendc import load_contract
    from flashinfer.jit import AscendCBackend, AscendCToolchain
    from flashinfer.jit.discovery import AotCorruptError, AotMissingError, AotPackageIndex
    from flashinfer.runtime import current_hardware

    assert torch.npu.is_available() and torch.npu.device_count() > 0, "missing NPU/runtime is a failure"
    index = int(os.environ.get("FLASHINFER_TEST_NPU_DEVICE", "0"))
    torch.npu.set_device(index)
    kernel_id = select_aot_implementation(name, kernel_id).kernel_id
    descriptor, recipe, spec = load_contract(kernel_id)
    assert current_hardware().soc_model == descriptor.target_soc, "device differs from declared target"
    torch.npu.config.allow_internal_format = True
    if mode == "installed":
        assert Path(flashinfer.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), flashinfer.__file__
    runner = flashinfer.get_global_registry().get(kernel_id).runner
    assert runner.provenance is None, "acceptance requires a fresh runner"
    if mode == "installed":
        from flashinfer._resources import data_root
        runner.index = AotPackageIndex(roots=(data_root() / "op_packages",))
    elif os.environ.get("FLASHINFER_AOT_PATHS"):
        runner.index = AotPackageIndex(roots=os.environ["FLASHINFER_AOT_PATHS"].split(os.pathsep))
    expect_jit = mode == "jit"
    if expect_jit:
        assert recipe.allow_jit_fallback, f"{kernel_id} only accepts a distributed AOT package"
    counts = dict(compiler=0, toolchain=0, jit=0)

    def counted(name, original):
        def called(*args, **kwargs):
            counts[name] += 1
            if not expect_jit:
                raise AssertionError("AOT acceptance attempted " + name)
            return original(*args, **kwargs)
        return called

    with ExitStack() as guards:
        for owner, attribute, counter in ((AscendCToolchain, "discover", "toolchain"),
                                          (AscendCBackend, "compile", "compiler"),
                                          (flashinfer.ascendc, "_jit_backend", "jit")):
            guards.enter_context(patch.object(owner, attribute, counted(counter, getattr(owner, attribute))))
        if mode not in ("missing", "corrupt", "jit"):
            packages = runner.index if runner.index is not None else AotPackageIndex()
            match = packages.find(spec, descriptor=descriptor, target_soc=spec.target_soc)
            # Other variants in a full release can expose the same CANN op type.
            runner.index = AotPackageIndex(roots=(match.directory,))
            aliases = Path(guards.enter_context(TemporaryDirectory(prefix="flashinfer-vendors-")))
            paths = vendor_search_path([vendor_directory(match, aliases)], replace=local_vendor_names(name))
            guards.enter_context(patch.dict(os.environ, {"ASCEND_CUSTOM_OPP_PATH": paths}))
        elif "ASCEND_CUSTOM_OPP_PATH" in os.environ:
            guards.enter_context(patch.dict(os.environ))
            os.environ.pop("ASCEND_CUSTOM_OPP_PATH")
        if mode in ("missing", "corrupt"):
            cases = cases_module(name)
            args, options, _ = cases.sample(torch, torch_npu, **case_parameters(cases, kernel_id)[0])
            expected_error = AotMissingError if mode == "missing" else AotCorruptError
            try:
                public_function(name)(*args, **dict(options, backend="ascendc", kernel_id=kernel_id))
            except expected_error as error:
                if mode == "missing":
                    assert error.spec.fingerprint == spec.fingerprint
                assert runner.provenance is None
                observations = dict(error=type(error).__name__)
            else:
                raise AssertionError("invalid AOT installation was accepted")
        else:
            stream = torch.npu.Stream(device=index)
            with torch.npu.stream(stream):
                observations = numerical_cases(torch, torch_npu, name, first_only=mode in ("one", "jit"),
                                               kernel_id=kernel_id, backend="ascendc")
            assert runner.provenance.source == ("jit" if expect_jit else "aot")
            assert runner.provenance.spec_fingerprint == spec.fingerprint
            observations.update(source=runner.provenance.source, spec=runner.provenance.spec_fingerprint)
    if expect_jit:
        assert counts == dict(compiler=1, toolchain=1, jit=1), counts
    else:
        assert not any(counts.values()), counts
    result = dict(result="PASS", operator=name, kernel_id=kernel_id, target_soc=descriptor.target_soc,
                  mode=mode, compiler_calls=counts,
                  flashinfer=flashinfer.__file__, observations=observations)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operator")
    parser.add_argument("mode", nargs="?", default="matrix", choices=("matrix", "one", "missing", "jit", "corrupt", "installed"))
    parser.add_argument("--kernel-id", help="Exact implementation ID on the selected device")
    args = parser.parse_args()
    run(args.operator, args.mode, kernel_id=args.kernel_id)


if __name__ == "__main__":
    main()
