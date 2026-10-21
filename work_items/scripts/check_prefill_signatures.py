"""Verify declarations against the fixed upstream object, not moving HEAD."""

import ast
from pathlib import Path
import subprocess

REF = "2895577720741ddfedff118202c89584996b8875"
source = subprocess.check_output([
    "git", "-C", "/home/dy/projects/flashinfer-ori", "show",
    REF + ":flashinfer/prefill.py",
], text=True)


def extract(text):
    result = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name == "single_prefill_with_kv_cache":
            result[node.name] = ast.unparse(node.args)
        if isinstance(node, ast.ClassDef) and node.name in (
            "BatchPrefillWithPagedKVCacheWrapper", "BatchPrefillWithRaggedKVCacheWrapper",
        ):
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name in (
                    "__init__", "plan", "run", "forward", "workspace_size",
                ):
                    result[node.name + "." + method.name] = ast.unparse(method.args)
    return result


expected = extract(source)
actual = extract(Path("flashinfer/prefill.py").read_text())
assert actual == expected, {k: (v, actual.get(k)) for k, v in expected.items() if actual.get(k) != v}
print(f"{len(expected)}/{len(expected)} prefill declarations match {REF}")
