"""Run the task's package import gate and retain structured evidence."""

import importlib
import json
from pathlib import Path

bad = []
paths = sorted(Path("flashinfer").rglob("*.py"))
for path in paths:
    module = str(path.with_suffix("")).replace("/", ".").removesuffix(".__init__")
    try:
        importlib.import_module(module)
    except Exception as error:
        bad.append(dict(module=module, error=type(error).__name__, message=str(error)))
print(json.dumps(dict(modules=len(paths), failing=len(bad), failures=bad), indent=2))
raise SystemExit(bool(bad))
