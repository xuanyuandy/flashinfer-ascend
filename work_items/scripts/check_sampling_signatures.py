"""Check all ten declarations against the fixed upstream git object."""

import ast
from pathlib import Path
import subprocess

from flashinfer._sampling import OP_NAMES

upstream = subprocess.check_output([
    "git", "-C", "/home/dy/projects/flashinfer-ori", "show",
    "2895577720741ddfedff118202c89584996b8875:flashinfer/sampling.py",
], text=True)


def signatures(source):
    return {n.name: ast.unparse(n.args) for n in ast.parse(source).body
            if isinstance(n, ast.FunctionDef) and n.name in OP_NAMES}


expected = signatures(upstream)
actual = signatures(Path("flashinfer/sampling.py").read_text())
assert actual == expected, (expected, actual)
print("10/10 signatures match pinned upstream (names, order, defaults, annotations)")
