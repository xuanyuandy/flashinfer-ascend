"""Inventory Python API usage from tests without importing either hardware backend.

This is static evidence, not proof that a test ran or that an API is supported.
Aliases resolve to declarations; bound methods retain their owning class.
Dynamic lookups and unresolved declarations remain explicit in the output.
"""

import argparse
import ast
from collections import Counter, defaultdict
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess


# Public facade namespaces are supplemented even when an export has no test.
# All other test references remain in the raw inventory, including internals.
PUBLIC_MODULES = (
    "activation attention prefill decode cascade gemm norm rope sampling page "
    "quantization quantization.fp4_quantization quantization.fp8_quantization "
    "fused_moe mla sparse comm moe_ep mamba gdn_decode gdn_prefill gdn2_prefill "
    "gdp_prefill kda kda_decode kda_prefill mhc msa_ops topk topk_varlen "
    "attn_scores logits_processor diffusion_ops grouped_mm concat_ops dsv3_ops "
    "gated_act_mxfp8 qwen_3_5_block parallel_attention"
).split()


def declarations(body):
    """Walk module/class control flow, never enter a function's local scope."""
    for node in body:
        yield node
        if isinstance(node, (ast.If, ast.Try, ast.With)):
            for field in ("body", "orelse", "finalbody"):
                yield from declarations(getattr(node, field, ()))
            for handler in getattr(node, "handlers", ()):
                yield from declarations(handler.body)


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted(node.value)
        return base + "." + node.attr if base else None
    return None


class SourceTree:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.modules = {}
        self.hashes = {}
        self._binding_stack = set()
        for path in sorted((self.root / "flashinfer").rglob("*.py")):
            if path.is_symlink() or "data" in path.relative_to(self.root).parts:
                continue
            stem = path.relative_to(self.root).with_suffix("").parts
            module = ".".join(stem[:-1] if stem[-1] == "__init__" else stem)
            self.modules[module] = path

    @lru_cache(None)
    def tree(self, module):
        path = self.modules[module]
        raw = path.read_bytes()
        self.hashes[path.relative_to(self.root).as_posix()] = hashlib.sha256(raw).hexdigest()
        return ast.parse(raw, filename=str(path))

    def imported(self, module, node):
        if isinstance(node, ast.Import):
            return {a.asname or a.name.split(".")[0]: a.name if a.asname else a.name.split(".")[0]
                    for a in node.names}
        package = module.split(".") if self.modules[module].name == "__init__.py" else module.split(".")[:-1]
        base = package[:len(package) - node.level + 1] if node.level else []
        target = ".".join(base + ((node.module or "").split(".") if node.module else []))
        return {a.asname or a.name: target + "." + a.name for a in node.names if a.name != "*"}

    @lru_cache(None)
    def bindings(self, module):
        if module in self._binding_stack:
            return {}
        self._binding_stack.add(module)
        result = {}
        for node in declarations(self.tree(module).body):
            if isinstance(node, (ast.ImportFrom, ast.Import)):
                result.update(self.imported(module, node))
                if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
                    probe = ast.ImportFrom(module=node.module, names=[ast.alias(name="__star__")], level=node.level)
                    target = self.imported(module, probe)["__star__"].removesuffix(".__star__")
                    if target in self.modules:
                        available = self.bindings(target)
                        names = self.string_constants(target).get("__all__", ()) or {n for n in available if not n.startswith("_")}
                        result.update({n: target + "." + n for n in names})
            elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                result[node.name] = module + "." + node.name
        # Recognize the finite lazy-export pattern: a literal name set,
        # a module import and getattr(module, name). Do not execute __getattr__.
        constants = self.string_constants(module)
        dictionaries = {}
        for node in declarations(self.tree(module).body):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
                pairs = [(k.value, v.value) for k, v in zip(node.value.keys, node.value.values)
                         if isinstance(k, ast.Constant) and isinstance(k.value, str)
                         and isinstance(v, ast.Constant) and isinstance(v.value, str)]
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        dictionaries[target.id] = dict(pairs)
        for function in self.tree(module).body:
            if not isinstance(function, ast.FunctionDef) or function.name != "__getattr__":
                continue
            # A literal export->relative-module dictionary followed by an
            # importlib/getattr dispatch is another finite lazy-export form.
            for node in ast.walk(function):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" and isinstance(node.func.value, ast.Name):
                    for name, target in dictionaries.get(node.func.value.id, {}).items():
                        if target.startswith("."):
                            level = len(target) - len(target.lstrip("."))
                            base = module.split(".")[:len(module.split(".")) - level + 1]
                            target = ".".join(base + target.lstrip(".").split("."))
                        result[name] = target + "." + name
            def lazy_body(body, names=None, local=None):
                local = dict(local or {})
                for node in body:
                    if isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name) and node.test.left.id == "name":
                        test = node.test
                        allowed = set()
                        for expression in test.comparators:
                            if isinstance(expression, ast.Name):
                                allowed.update(constants.get(expression.id, ()))
                            elif isinstance(expression, ast.Constant) and isinstance(expression.value, str):
                                allowed.add(expression.value)
                        if len(test.ops) == 1 and isinstance(test.ops[0], (ast.In, ast.Eq)):
                            lazy_body(node.body, allowed if names is None else names & allowed, local)
                            lazy_body(node.orelse, None if names is None else names - allowed, local)
                        elif len(test.ops) == 1 and isinstance(test.ops[0], ast.NotIn) and any(isinstance(n, ast.Raise) for n in node.body):
                            names = allowed if names is None else names & allowed
                        continue
                    if isinstance(node, (ast.ImportFrom, ast.Import)):
                        local.update(self.imported(module, node))
                    expression = node.value if isinstance(node, (ast.Assign, ast.Return)) else None
                    if names and isinstance(expression, ast.Call) and dotted(expression.func) == "getattr" and len(expression.args) >= 2 and isinstance(expression.args[0], ast.Name) and isinstance(expression.args[1], ast.Name) and expression.args[1].id == "name":
                        target = local.get(expression.args[0].id)
                        if target:
                            result.update({name: target + "." + name for name in names})
                    if names and isinstance(node, ast.Return) and isinstance(expression, ast.Name):
                        target = local.get(expression.id, result.get(expression.id))
                        if target:
                            result.update({name: target for name in names})

            lazy_body(function.body)
        self._binding_stack.remove(module)
        return result

    @lru_cache(None)
    def string_constants(self, module):
        values = {}

        def strings(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return {node.value}
            if isinstance(node, ast.Name):
                return values.get(node.id, set())
            if isinstance(node, (ast.Set, ast.Tuple, ast.List)):
                return set().union(*(strings(n) for n in node.elts))
            if isinstance(node, ast.BinOp):
                return strings(node.left) | strings(node.right)
            if isinstance(node, ast.Call) and dotted(node.func) in {"set", "frozenset", "list", "tuple"} and node.args:
                return strings(node.args[0])
            return set()

        for node in declarations(self.tree(module).body):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        values[target.id] = strings(node.value)
        return values

    def qualify(self, module, node):
        name = dotted(node)
        if not name:
            return None
        first, *rest = name.split(".")
        head = self.bindings(module).get(first, module + "." + first)
        return ".".join([head, *rest])

    @lru_cache(None)
    def resolve(self, symbol, seen=()):
        if symbol in seen:
            return None
        if symbol in self.modules:
            parent, _, name = symbol.rpartition(".")
            if parent in self.modules:
                target = self.bindings(parent).get(name)
                if target and target != symbol:
                    return self.resolve(target, seen + (symbol,))
            return {"canonical": symbol, "kind": "module", "path": str(self.modules[symbol].relative_to(self.root)), "line": 1}
        parts = symbol.split(".")
        split = next((i for i in range(len(parts) - 1, 0, -1) if ".".join(parts[:i]) in self.modules), None)
        if split is None:
            return None
        module, attrs = ".".join(parts[:split]), parts[split:]
        namespace = self.tree(module)
        owner = None
        for offset, name in enumerate(attrs):
            nodes = list(declarations(namespace.body))
            found = [n for n in nodes if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
            if found:
                node = found[-1]  # Ignore @overload stubs preceding the body.
                if offset == len(attrs) - 1:
                    result = {"canonical": module + "." + ".".join(attrs),
                              "kind": "class" if isinstance(node, ast.ClassDef) else ("method" if owner else "function"),
                              "path": str(self.modules[module].relative_to(self.root)), "line": node.lineno}
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        result["signature"] = ast.unparse(node.args)
                    if isinstance(node, ast.ClassDef):
                        methods = {n.name for n in declarations(node.body)
                                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("_")}
                        for base in node.bases:
                            target = self.qualify(module, base)
                            inherited = self.resolve(target, seen + (symbol,)) if target else None
                            methods.update((inherited or {}).get("methods", ()))
                        for child in declarations(node.body):
                            if isinstance(child, ast.Assign):
                                methods.update(t.id for t in child.targets if isinstance(t, ast.Name) and not t.id.startswith("_"))
                        result["methods"] = sorted(methods)
                        result["bases"] = [ast.unparse(b) for b in node.bases]
                        initializer = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
                        result["constructor_signature"] = ast.unparse(initializer.args) if initializer else "inherited or implicit"
                        result["lifecycle_signatures"] = {}
                        for method in ("plan", "run", "begin_forward", "forward"):
                            if method in methods:
                                definition = self.resolve(result["canonical"] + "." + method, seen + (symbol,))
                                if definition and "signature" in definition:
                                    result["lifecycle_signatures"][method] = definition["signature"]
                    return result
                if not isinstance(node, ast.ClassDef):
                    return None
                namespace, owner = node, name
                continue
            if offset == 0 and name in self.bindings(module):
                target = self.bindings(module)[name] + ("." + ".".join(attrs[1:]) if len(attrs) > 1 else "")
                return self.resolve(target, seen + (symbol,))
            for node in nodes:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                    value = node.value
                    if isinstance(value, ast.Call) and dotted(value.func) in {"partial", "partialmethod", "functools.partial", "functools.partialmethod"} and value.args:
                        value = value.args[0]
                    target = self.qualify(module, value)
                    if target:
                        if owner and isinstance(value, ast.Name):
                            target = module + "." + ".".join(attrs[:offset] + [value.id] + attrs[offset + 1:])
                        return self.resolve(target, seen + (symbol,))
                    return {"canonical": symbol, "kind": "value", "path": str(self.modules[module].relative_to(self.root)), "line": node.lineno}
            if owner:
                for base in namespace.bases:
                    target = self.qualify(module, base)
                    result = self.resolve(target + "." + ".".join(attrs[offset:]), seen + (symbol,)) if target else None
                    if result:
                        return result
            return None

    def root_exports(self):
        return {"flashinfer." + name for name, target in self.bindings("flashinfer").items()
                if not name.startswith("_") and target.startswith("flashinfer.")}

    def facade_exports(self):
        exports = set(self.root_exports())
        for suffix in PUBLIC_MODULES:
            module = "flashinfer." + suffix
            if module not in self.modules:
                continue
            constants = self.string_constants(module)
            if "__all__" in constants:
                names = constants["__all__"]
            else:
                nodes = list(declarations(self.tree(module).body))
                definitions = {n.name for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
                imported = {name for n in nodes if isinstance(n, (ast.Import, ast.ImportFrom)) for name in self.imported(module, n)}
                intentional = {a.asname for n in nodes if isinstance(n, ast.ImportFrom) for a in n.names if a.asname == a.name}
                # Public definitions, explicit self-alias reexports, star and
                # finite lazy exports; omit incidental imported decorators.
                names = definitions | intentional | (self.bindings(module).keys() - imported - definitions)
            for name in names:
                if not name.startswith("_"):
                    target = module + "." + name
                    definition = self.resolve(target)
                    if definition and definition["kind"] in {"function", "class"}:
                        exports.add(target)
        return exports


class TestUsage(ast.NodeVisitor):
    def __init__(self, source, path, tree, index):
        self.source, self.path, self.index = source, path, index
        self.env = {}
        self.records = defaultdict(set)
        self.dynamic = []
        self.context = "<module>"
        self.visit(tree)

    def record(self, symbol, node, usage):
        if symbol.startswith("flashinfer."):
            self.records[symbol].add((self.path, node.lineno, usage, self.context))

    def refs(self, node):
        if dotted(node) in self.env:
            return self.env[dotted(node)]
        if isinstance(node, ast.Name):
            return self.env.get(node.id, set())
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {"@literal:" + node.value}
        if isinstance(node, ast.Attribute):
            return {name + "." + node.attr for name in self.refs(node.value)}
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return set().union(*(self.refs(n) for n in node.elts))
        if isinstance(node, ast.IfExp):
            return self.refs(node.body) | self.refs(node.orelse)
        if isinstance(node, ast.Subscript):
            return self.refs(node.value)
        if isinstance(node, ast.Call):
            name = dotted(node.func)
            if name in {"getattr", "hasattr"} and len(node.args) >= 2:
                base = self.refs(node.args[0])
                if isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                    return {value + "." + node.args[1].value for value in base}
                names = {value.removeprefix("@literal:") for value in self.refs(node.args[1]) if value.startswith("@literal:")}
                return {value + "." + attribute for value in base for attribute in names if attribute.isidentifier()}
            if name in {"partial", "functools.partial"} and node.args:
                return self.refs(node.args[0])
            if name in {"importlib.import_module", "pytest.importorskip", "__import__"} and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                return {node.args[0].value}
            results = set()
            for value in self.refs(node.func):
                if value.startswith("flashinfer."):
                    declaration = self.index.resolve(value)
                    if declaration and declaration["kind"] == "class":
                        results.add("@instance:" + value)
            return results
        return set()

    def bind(self, target, values):
        name = dotted(target)
        if name:
            self.env[name] = values
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                self.bind(item, values)

    def visit_Import(self, node):
        for a in node.names:
            value = a.name if a.asname else a.name.split(".")[0]
            self.env[a.asname or a.name.split(".")[0]] = {value}
            self.record(a.name, node, "import")

    def visit_ImportFrom(self, node):
        if node.level or not node.module:
            return
        for a in node.names:
            if a.name != "*":
                value = node.module + "." + a.name
                self.env[a.asname or a.name] = {value}
                self.record(value, node, "import")
            elif node.module.startswith("flashinfer"):
                self.dynamic.append({"path": self.path, "line": node.lineno, "expression": ast.unparse(node), "reason": "star import"})

    def visit_Assign(self, node):
        self.visit(node.value)
        values = self.refs(node.value)
        for target in node.targets:
            self.bind(target, values)

    def visit_AnnAssign(self, node):
        if node.value:
            self.visit(node.value)
            self.bind(node.target, self.refs(node.value))

    def visit_For(self, node):
        self.visit(node.iter)
        self.bind(node.target, self.refs(node.iter))
        for child in node.body + node.orelse:
            self.visit(child)

    def visit_FunctionDef(self, node):
        for decorator in node.decorator_list:
            self.visit(decorator)
        original, context = self.env, self.context
        self.env = dict(original)
        self.context = (context + "." if context != "<module>" else "") + node.name
        for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
            self.env.pop(arg.arg, None)
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call) and (dotted(decorator.func) or "").endswith("parametrize") and len(decorator.args) >= 2:
                names = decorator.args[0]
                if isinstance(names, ast.Constant) and isinstance(names.value, str):
                    parameters = [name.strip() for name in names.value.split(",")]
                    values = decorator.args[1]
                    if len(parameters) > 1 and isinstance(values, (ast.List, ast.Tuple)):
                        columns = {name: set() for name in parameters}
                        for case in values.elts:
                            items = case.elts if isinstance(case, (ast.List, ast.Tuple)) else (case.args if isinstance(case, ast.Call) else ())
                            for name, item in zip(parameters, items):
                                columns[name].update(self.refs(item))
                        self.env.update(columns)
                    else:
                        self.env[parameters[0]] = self.refs(values)
        for child in node.body:
            self.visit(child)
        self.env, self.context = original, context

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Attribute(self, node):
        for name in self.refs(node):
            if not name.startswith("@"):
                self.record(name, node, "reference")
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            for value in self.refs(node):
                if not value.startswith("@"):
                    self.record(value, node, "reference")

    def visit_Call(self, node):
        for value in self.refs(node.func):
            self.record(value.removeprefix("@instance:"), node, "bound-call" if value.startswith("@instance:") else "call")
        if isinstance(node.func, ast.Name) and node.func.id in {"getattr", "hasattr"} and len(node.args) >= 2:
            if any(v.startswith("flashinfer") for v in self.refs(node.args[0])) and not isinstance(node.args[1], ast.Constant):
                self.dynamic.append({"path": self.path, "line": node.lineno, "expression": ast.unparse(node)[:250], "reason": "nonliteral attribute name"})
        self.generic_visit(node)


def call_style(declaration):
    if not declaration:
        return "unresolved"
    methods = set(declaration.get("methods", ()))
    if declaration["kind"] == "class":
        if {"plan", "run"} <= methods:
            return "wrapper-plan-run"
        if {"begin_forward", "forward"} <= methods:
            return "wrapper-begin-forward"
        if "run" in methods:
            return "wrapper-run"
        return "class-or-resource"
    return {"function": "direct-function", "method": "bound-method", "value": "value", "module": "module"}[declaration["kind"]]


def scan(root):
    root = Path(root).resolve()
    index = SourceTree(root)
    observations = defaultdict(set)
    files, dynamic = [], []
    for path in sorted((root / "tests").rglob("*.py")):
        raw = path.read_bytes()
        tree = ast.parse(raw, filename=str(path))
        relative = path.relative_to(root).as_posix()
        visitor = TestUsage(raw.decode("utf-8"), relative, tree, index)
        for symbol, usages in visitor.records.items():
            observations[symbol].update(usages)
        dynamic.extend(visitor.dynamic)
        files.append({"path": relative, "sha256": hashlib.sha256(raw).hexdigest(),
                      "test_definitions": sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test_") for n in ast.walk(tree)),
                      "referenced_symbols": sorted(visitor.records), "dynamic_lookups": len(visitor.dynamic)})
    exports = index.root_exports()
    facade_exports = index.facade_exports()
    rows = []
    for symbol in sorted(set(observations) | facade_exports):
        definition = index.resolve(symbol)
        if definition and definition["kind"] == "module":
            continue
        records = sorted(observations[symbol])
        private = any(part.startswith("_") for part in symbol.split(".")[1:])
        rows.append({"symbol": symbol, "definition": definition, "call_style": call_style(definition),
                     "exposure": "top-level-export" if symbol in exports else ("facade-module-export" if symbol in facade_exports else ("private-test-reference" if private else "module-test-reference")),
                     "seen_in_tests": bool(records), "usages": [dict(path=p, line=n, usage=u, context=c) for p, n, u, c in records]})
    summary = {"python_test_files": len(files), "test_definitions": sum(f["test_definitions"] for f in files),
               "symbols": len(rows), "test_referenced_symbols": sum(r["seen_in_tests"] for r in rows),
               "resolved_symbols": sum(r["definition"] is not None for r in rows),
               "call_styles": dict(Counter(r["call_style"] for r in rows)),
               "dynamic_lookups": len(dynamic)}
    return {"summary": summary, "symbols": rows, "files": files,
            "dynamic_lookups": dynamic, "source_sha256": dict(sorted(index.hashes.items()))}


def write_csv(path, rows, fields):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = scan(args.upstream)
    result["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.upstream, text=True).strip()
    result["dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=args.upstream))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "test-api-inventory.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    flattened = []
    for row in result["symbols"]:
        d = row["definition"] or {}
        flattened.append({"symbol": row["symbol"], "canonical": d.get("canonical", ""),
                          "source": d.get("path", ""), "line": d.get("line", ""),
                          "kind": d.get("kind", "unresolved"), "call_style": row["call_style"],
                          "exposure": row["exposure"], "methods": ";".join(d.get("methods", ())),
                          "test_files": ";".join(sorted({u["path"] for u in row["usages"]})),
                          "test_calls": sum(u["usage"] in {"call", "bound-call"} for u in row["usages"])})
    write_csv(args.output_dir / "test-api-inventory.csv", flattened,
              ["symbol", "canonical", "source", "line", "kind", "call_style", "exposure", "methods", "test_files", "test_calls"])
    write_csv(args.output_dir / "test-file-coverage.csv",
              [{**row, "referenced_symbols": ";".join(row["referenced_symbols"])} for row in result["files"]],
              ["path", "sha256", "test_definitions", "referenced_symbols", "dynamic_lookups"])
    write_csv(args.output_dir / "dynamic-lookups.csv", result["dynamic_lookups"], ["path", "line", "expression", "reason"])
    print(json.dumps({"commit": result["commit"], "dirty": result["dirty"], **result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
