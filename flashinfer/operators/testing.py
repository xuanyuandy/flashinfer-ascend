"""Common numerical/AOT/out/stream/installed acceptance using author-owned cases."""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

from .catalog import cases_module, definition
from .runtime import public_function


def _outputs(result, count):
    outputs = (result,) if count == 1 else tuple(result)
    if len(outputs) != count:
        raise AssertionError("operator returned the wrong number of outputs")
    return outputs


def numerical_cases(torch, torch_npu, name, *, first_only=False):
    operator = definition(name)
    cases = cases_module(name)
    parameters = cases.CASE_PARAMETERS[:1] if first_only else cases.CASE_PARAMETERS
    if not parameters or len(cases.TOLERANCES) != len(operator.output_names):
        raise AssertionError("numerical cases and output tolerances must be explicit and nonempty")
    call = public_function(name)
    observations = {}
    for index, configuration in enumerate(parameters):
        args, options, expected = cases.sample(torch, torch_npu, **configuration)
        originals = tuple(t for t in tuple(args) + tuple(options.values()) if isinstance(t, torch.Tensor))
        pointers = tuple(t.data_ptr() for t in originals)
        outputs = _outputs(call(*args, **options), len(operator.output_names))
        if hasattr(cases, "verify_effect"):
            cases.verify_effect(torch, call, args, options, outputs)
        supplied = tuple(torch.empty_like(t) for t in outputs)
        output_pointers = tuple(t.data_ptr() for t in supplied)
        returned = _outputs(call(*args, out=supplied[0] if len(supplied) == 1 else supplied, **options), len(supplied))
        assert all(a is b for a, b in zip(returned, supplied)), "out identity changed"
        assert output_pointers == tuple(t.data_ptr() for t in supplied), "out storage changed"
        assert pointers == tuple(t.data_ptr() for t in originals), "input storage changed"
        expected = _outputs(expected, len(outputs))
        errors = []
        for actual, reference, (atol, rtol) in zip(outputs, expected, cases.TOLERANCES):
            errors.append(float((actual.cpu().double() - reference.double()).abs().max()))
        print(json.dumps(dict(case=configuration, max_abs=errors), sort_keys=True), flush=True)
        for output_tuple in (outputs, returned):
            for actual, reference, (atol, rtol) in zip(output_tuple, expected, cases.TOLERANCES):
                torch.testing.assert_close(actual.cpu().double(), reference.double(), atol=atol, rtol=rtol)
        observations[str(index)] = dict(case=configuration, max_abs=errors)
    return observations


def selftest_case(torch, name):
    import torch_npu
    torch.npu.config.allow_internal_format = True
    observations = numerical_cases(torch, torch_npu, name, first_only=True)
    return "author CPU reference matched; max_abs=" + repr(observations["0"]["max_abs"])


def target_aot_implementations(name):
    """Acceptance candidates on this device; callers still require a unique choice."""
    from flashinfer.runtime import current_hardware
    from .implementation import implementations

    operator = definition(name)
    soc = current_hardware().soc_model
    return tuple(
        d for d in implementations(backend="ascendc")
        if d.op == operator.op and d.bind().target_soc == soc
    )


def run(name, mode="matrix"):
    import torch
    import torch_npu
    import flashinfer
    from flashinfer.runtime import current_hardware

    assert torch.npu.is_available() and torch.npu.device_count() > 0, "missing NPU/runtime is a failure"
    index = int(os.environ.get("FLASHINFER_TEST_NPU_DEVICE", "0"))
    torch.npu.set_device(index)
    choices = target_aot_implementations(name)
    assert len(choices) == 1, "acceptance must identify one explicit AOT implementation"
    kernel_id = choices[0].kernel_id
    from flashinfer.ascendc import load_contract
    assert current_hardware().soc_model == load_contract(kernel_id)[0].target_soc, "device differs from declared target"
    torch.npu.config.allow_internal_format = True
    if mode == "installed":
        assert Path(flashinfer.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), flashinfer.__file__
    counts = dict(compiler=0, toolchain=0, jit=0)
    def forbidden(name):
        def called(*args, **kwargs):
            counts[name] += 1
            raise AssertionError("AOT acceptance attempted " + name)
        return called
    with ExitStack() as guards:
        guards.enter_context(patch("flashinfer.jit.AscendCToolchain.discover", side_effect=forbidden("toolchain")))
        guards.enter_context(patch("flashinfer.jit.AscendCBackend.compile", side_effect=forbidden("compiler")))
        guards.enter_context(patch("flashinfer.ascendc._jit_backend", side_effect=forbidden("jit")))
        if mode in ("missing", "corrupt"):
            cases = cases_module(name)
            args, options, _ = cases.sample(torch, torch_npu, **cases.CASE_PARAMETERS[0])
            try:
                public_function(name)(*args, **options)
            except Exception as error:
                from flashinfer.jit.discovery import AotMissingError, AotCorruptError
                assert isinstance(error, AotMissingError if mode == "missing" else AotCorruptError), repr(error)
                observations = dict(error=type(error).__name__)
            else:
                raise AssertionError("invalid AOT installation was accepted")
        else:
            stream = torch.npu.Stream(device=index)
            with torch.npu.stream(stream):
                observations = numerical_cases(torch, torch_npu, name, first_only=mode == "one")
            runner = flashinfer.get_global_registry().get(kernel_id).runner
            assert runner.provenance.source == "aot"
            observations.update(source=runner.provenance.source, spec=runner.provenance.spec_fingerprint)
    assert not any(counts.values()), counts
    result = dict(result="PASS", operator=name, mode=mode, compiler_calls=counts,
                  flashinfer=flashinfer.__file__, observations=observations)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operator")
    parser.add_argument("mode", nargs="?", default="matrix", choices=("matrix", "one", "missing", "corrupt", "installed"))
    args = parser.parse_args()
    run(args.operator, args.mode)


if __name__ == "__main__":
    main()
