"""Generate a typed adapter from one recipe and its CANN-generated ACLNN header.

This build input uses only the standard library and runs after OpDef codegen.
No hand-authored operator switch or SDK probing is involved.
"""

import argparse
import configparser
import hashlib
import json
from pathlib import Path
import re


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode("ascii")).hexdigest()


def _name(value):
    return value.replace("_", "").lower()


def workspace_signature(recipe, headers):
    entry = recipe["host_entry"] + "GetWorkspaceSize"
    matches = []
    for path in sorted(Path(headers).glob("aclnn_*.h")):
        text = re.sub(r"/\*.*?\*/|//[^\n]*", "", path.read_text(), flags=re.S)
        match = re.search(r"\b" + re.escape(entry) + r"\s*\((.*?)\)\s*;", text, re.S)
        if match:
            matches.append((path, match[1]))
    if len(matches) != 1:
        raise ValueError("expected exactly one generated ACLNN header for " + entry)
    header, signature = matches[0]
    arguments = []
    for argument in signature.split(","):
        argument = " ".join(argument.split())
        match = re.fullmatch(r"(.+?[ *])([A-Za-z_][A-Za-z0-9_]*)", argument)
        if match is None:
            raise ValueError("unsupported generated ACLNN argument: " + argument)
        arguments.append((match[1].strip(), match[2]))
    tensors = recipe["tensor_parameters"]
    # CANN's C API appends Optional to optional input names and Out to each
    # output name when there are multiple outputs. Bind these exact aliases,
    # without accepting arbitrary names or reordering identical tensor types.
    expected = [(p["name"] + ("Optional" if p.get("optional", False) else ""), "aclTensor*")
                for p in tensors if p["role"] == "input"]
    types = {"bool": "bool", "int64": "int64_t", "float64": "double", "string": "char*"}
    expected += [(p["name"], types[p["dtype"]]) for p in recipe.get("attributes", ())]
    outputs = [p for p in tensors if p["role"] == "output"]
    expected += [(p["name"] + ("Out" if len(outputs) > 1 else ""), "aclTensor*") for p in outputs]
    expected += [("workspaceSize", "uint64_t*"), ("executor", "aclOpExecutor**")]
    actual = [(_name(name), re.sub(r"\s|\bconst\b", "", dtype)) for dtype, name in arguments]
    wanted = [(_name(name), dtype) for name, dtype in expected]
    # CANN canonicalizes a single output's C parameter name to "out".
    # Its position and type remain exact; multiple outputs are never reordered.
    if len(outputs) == 1 and len(actual) == len(wanted) and actual[-3][0] == "out":
        actual[-3] = (wanted[-3][0], actual[-3][1])
    # Optional string attributes are nullable pointers in CANN's C signature;
    # scalar optional attributes keep their ordinary parameter names.
    inputs = sum(p["role"] == "input" for p in tensors)
    for index, attribute in enumerate(recipe.get("attributes", ()), inputs):
        if (attribute["dtype"] == "string" and index < len(actual)
                and actual[index][0] == wanted[index][0] + "optional"):
            actual[index] = (wanted[index][0], actual[index][1])
    if actual != wanted:
        raise ValueError("generated ACLNN ABI does not match NativeBinding: expected %r, got %r"
                         % (wanted, actual))
    return header, arguments


def validate_device_entry(recipe, ops_info):
    """Bind the author's entry name to OpDef codegen without rewriting C++."""
    info = configparser.ConfigParser(interpolation=None)
    info.optionxform = str
    with Path(ops_info).open() as source:
        info.read_file(source)
    op_type = recipe["package_op_type"]
    actual = info.get(op_type, "opInterface.value", fallback=None)
    if actual != recipe["device_entry"]:
        raise ValueError("OpDef device entry does not match BuildConfig.device_entry: "
                         "op_type=%s, declared=%s, generated=%s"
                         % (op_type, recipe["device_entry"], actual))


def generate(recipe_file, headers, output, *, ops_info=None):
    recipe = json.loads(Path(recipe_file).read_text())
    if recipe.get("schema_version") != 2:
        raise ValueError("typed ACLNN generation requires a v2 recipe")
    if ops_info is not None:
        validate_device_entry(recipe, ops_info)
    header, arguments = workspace_signature(recipe, headers)
    digest = fingerprint(recipe)
    signature = ", ".join(dtype + " " + name for dtype, name in arguments)
    names = ", ".join(name for _, name in arguments)
    entry = recipe["host_entry"]
    source = (
        '// Generated from NativeBinding and the matching CANN OpDef header.\n'
        '#include "' + header.name + '"\n'
        '#define FI_EXPORT extern "C" __attribute__((visibility("default")))\n'
        'FI_EXPORT const char* fi_aclnn_binding_fingerprint() { return "' + digest + '"; }\n'
        'FI_EXPORT aclnnStatus fi_aclnn_workspace(' + signature + ') {\n'
        '  return ' + entry + 'GetWorkspaceSize(' + names + ');\n}\n'
        'FI_EXPORT aclnnStatus fi_aclnn_launch(void* workspace, uint64_t size, '
        'aclOpExecutor* executor, aclrtStream stream) {\n'
        '  return ' + entry + '(workspace, size, executor, stream);\n}\n'
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source)
    metadata = dict(schema_version=1, kernel_id=recipe["kernel_id"],
                    recipe_fingerprint=digest, host_entry=entry,
                    tensor_parameters=recipe["tensor_parameters"],
                    attributes=recipe.get("attributes", []),
                    header_sha256=hashlib.sha256(header.read_bytes()).hexdigest(),
                    source_sha256=hashlib.sha256(source.encode()).hexdigest())
    output.with_suffix(".json").write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
    output.with_suffix(".h").write_bytes(header.read_bytes())
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--headers", required=True)
    parser.add_argument("--ops-info", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    generate(args.recipe, args.headers, args.output, ops_info=args.ops_info)
