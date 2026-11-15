"""Common functional/out/fake execution for declaratively registered operators."""

from dataclasses import replace
from functools import lru_cache
from types import SimpleNamespace

from flashinfer.runtime import GraphCapability, SchemaError, build_signature, current_hardware, dispatch, get_global_registry
from flashinfer.runtime.runner import run_selected
from .catalog import definition
from .definition import Invocation


def allocate_output(specification):
    """Allocate a fresh public output with the declared row-major ND storage."""
    import torch
    output = torch.empty(specification.shape, dtype=getattr(torch, specification.dtype),
                         device=specification.device)
    if output.device.type == torch._C._get_privateuse1_backend_name():
        import torch_npu
        # TorchNPU labels four-dimensional empty/empty_like tensors NCHW,
        # including empty_with_format(..., acl_format=2) on this runtime.
        # Only fresh framework outputs are converted; supplied out stays intact.
        output = torch_npu.npu_format_cast(output, 2)
    return output


def metadata(name, arguments, *, check_layouts=False):
    operator = definition(name)
    if operator.infer is None:
        raise SchemaError("operator uses its existing domain frontend: " + operator.op)
    layouts = None
    if check_layouts:
        import torch
        layouts = {}
        for key, tensor in arguments.items():
            if isinstance(tensor, torch.Tensor):
                if tensor.device.type == torch._C._get_privateuse1_backend_name():
                    import torch_npu
                    layouts[key] = {2: "ND", 29: "NZ"}.get(torch_npu.get_npu_format(tensor), "unknown")
                else:
                    layouts[key] = "ND" if tensor.is_contiguous() else "strided"
    invocation = operator.infer(arguments, layouts)
    if not isinstance(invocation, Invocation) or len(invocation.outputs) != len(operator.output_names):
        raise SchemaError("operator infer hook must return logical inputs and output metadata")
    import torch
    supplied = arguments.get("out")
    if supplied is not None:
        if len(invocation.outputs) == 1 and isinstance(supplied, torch.Tensor):
            supplied = (supplied,)
        if not isinstance(supplied, (tuple, list)) or len(supplied) != len(invocation.outputs):
            raise ValueError("out must supply every declared output Tensor")
        outputs = tuple(supplied)
    else:
        outputs = tuple(SimpleNamespace(shape=tuple(s.shape), dtype=getattr(torch, s.dtype), device=s.device,
                                        is_contiguous=lambda: True) for s in invocation.outputs)
    for output, expected in zip(outputs, invocation.outputs):
        if tuple(output.shape) != tuple(expected.shape) or str(output.dtype).removeprefix("torch.") != expected.dtype:
            raise ValueError("out shape/dtype differs from operator output metadata")
    tensors = invocation.inputs + outputs
    if check_layouts and supplied is not None:
        from flashinfer.jit.tensor_adapter import _reject_output_overlap
        parameters = tuple(SimpleNamespace(role="input" if i < len(invocation.inputs) else "output")
                           for i in range(len(tensors)))
        _reject_output_overlap(parameters, tensors)
    device = next(t.device for t in tensors if t is not None)
    if any(t is not None and t.device != device for t in tensors):
        raise ValueError("operator Tensors must share one device")
    return tensors, invocation


def signature(name, arguments):
    operator = definition(name)
    tensors, invocation = metadata(name, arguments, check_layouts=True)
    facts = dict(invocation.specializations, tensor_contract="supported",
                 tensor_contiguity="".join("x" if t is None else "1" if t.is_contiguous() else "0" for t in tensors))
    facts.update(("attribute." + key, value) for key, value in invocation.attributes.items())
    result = build_signature(operator.op, tensors, inference=invocation.inference, specializations=facts)
    output_count = len(invocation.outputs)
    return replace(result,
        dtypes=tuple("none" if t is None else d for t, d in zip(tensors, result.dtypes)),
        physical_layouts=result.physical_layouts[:-output_count] + ("ND",) * output_count), tensors, invocation


def execute(name, arguments):
    import torch
    operator = definition(name)
    request, tensors, invocation = signature(name, arguments)
    decision = dispatch(request, current_hardware(), backend=arguments.get("backend"), kernel_id=arguments.get("kernel_id"))
    selected = get_global_registry().get(decision.kernel_id)
    if any(t is not None and t.device.type == torch._C._get_privateuse1_backend_name() for t in tensors):
        import torch_npu
        if torch_npu.npu.is_current_stream_capturing() and selected.graph_capability == GraphCapability.NOT_CAPTURABLE:
            raise SchemaError(operator.name + " graph_capability=NOT_CAPTURABLE")
    if operator.validate_values is not None:
        operator.validate_values(arguments)
    outputs = tensors[-len(invocation.outputs):]
    if arguments.get("out") is None:
        outputs = tuple(allocate_output(s) for s in invocation.outputs)
        tensors = invocation.inputs + outputs
    run_selected(decision.kernel_id, *tensors, **invocation.attributes)
    return outputs[0] if len(outputs) == 1 else outputs


def fake(name, arguments):
    import torch
    _, invocation = metadata(name, arguments)
    if arguments.get("out") is not None:
        return None
    outputs = tuple(torch.empty(s.shape, dtype=getattr(torch, s.dtype), device=s.device) for s in invocation.outputs)
    return outputs[0] if len(outputs) == 1 else outputs


@lru_cache(maxsize=None)
def public_function(name):
    from .catalog import _record
    from .definition import schema_arguments, signature as python_signature
    record = _record(name)
    if not record["export"]:
        import importlib
        operator = definition(record["op"])
        if operator.public_entry is None:
            raise SchemaError("operator has no public function: " + operator.op)
        module, attribute = operator.public_entry.split(":", 1)
        return getattr(importlib.import_module(module), attribute)
    name = record["name"]
    # Building the facade reads only generated metadata. Authoring can repair
    # stale source digests; every execution still validates its installed hooks.
    public_signature = python_signature(schema_arguments(record["schema"])[0] + (
        ("object", "out", None, True), ("str?", "backend", None, True),
        ("str?", "kernel_id", None, True)))
    def call(*args, **kwargs):
        import torch
        from flashinfer.aclnn._ops import ensure_registered
        operator = definition(record["op"])
        bound = operator.public_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        out = arguments.pop("out")
        arguments["backend"] = getattr(arguments.get("backend"), "value", arguments.get("backend"))
        ensure_registered()
        if out is None:
            return getattr(torch.ops.flashinfer, name)(**arguments)
        outputs = (out,) if len(operator.output_names) == 1 and isinstance(out, torch.Tensor) else out
        if not isinstance(outputs, (tuple, list)) or len(outputs) != len(operator.output_names):
            raise ValueError("out must supply every declared output Tensor")
        getattr(torch.ops.flashinfer, name + "_out")(**arguments, **dict(zip(operator.output_names, outputs)))
        return outputs[0] if len(outputs) == 1 else tuple(outputs)
    call.__name__ = name
    call.__qualname__ = name
    call.__module__ = "flashinfer." + record["domain"]
    call.__doc__ = record["description"]
    call.__signature__ = public_signature
    return call


@lru_cache(maxsize=1)
def torch_declarations():
    from .catalog import entries
    from flashinfer.jit.torch_library import AscendCOp
    declarations = []
    for record in entries():
        operator = definition(record["op"])
        if not record["export"]:
            import importlib
            provider = getattr(importlib.import_module(record["module"]), "torch_declarations", None)
            if provider is not None:
                declarations.extend(provider())
            continue
        for out in (False, True):
            schema, binding = operator.torch_schema(out=out)
            def wrapper(implementation, name=operator.name, binding=binding, outputs=operator.output_names, out=out):
                def invoke(*args, **kwargs):
                    bound = binding.bind(*args, **kwargs)
                    bound.apply_defaults()
                    arguments = dict(bound.arguments)
                    if out:
                        arguments["out"] = tuple(arguments.pop(n) for n in outputs)
                    result = implementation(name, arguments)
                    return None if out else result
                return invoke
            declarations.append(AscendCOp("flashinfer::" + operator.name + ("_out" if out else ""), schema,
                wrapper(execute), wrapper(fake), tuple(operator.output_names) if out else ()))
    return tuple(declarations)
