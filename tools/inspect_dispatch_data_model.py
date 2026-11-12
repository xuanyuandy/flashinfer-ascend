#!/usr/bin/env python3
"""Extract real dispatch record definitions and a metadata-only Matmul walkthrough.

The walkthrough copies production descriptors into a private registry, removes
their execution callables, and runs the real dispatcher. No tensor allocation,
kernel compilation or launch occurs; the production registry stays unchanged.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import Field, MISSING, fields, is_dataclass, replace
from enum import Enum
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from typing import get_type_hints


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODEL_SPECS = {
    "InputSignature": ("flashinfer/runtime/signature.py", "InputSignature"),
    "HardwareSignature": ("flashinfer/runtime/hardware.py", "HardwareSignature"),
    "CapabilitySpec": ("flashinfer/runtime/schema.py", "CapabilitySpec"),
    "FeatureDefinition": ("flashinfer/runtime/features.py", "FeatureDefinition"),
    "ShapeRange": ("flashinfer/runtime/schema.py", "ShapeRange"),
    "KernelDescriptor": ("flashinfer/runtime/schema.py", "KernelDescriptor"),
    "ImplementationStack": ("flashinfer/runtime/registry.py", "ImplementationStack"),
    "CandidateReport": ("flashinfer/runtime/dispatch.py", "CandidateReport"),
    "StackReport": ("flashinfer/runtime/dispatch.py", "StackReport"),
    "DispatchDecision": ("flashinfer/runtime/dispatch.py", "DispatchDecision"),
    "_TuningResolution": ("flashinfer/tuning/store.py", "_TuningResolution"),
    "SourceRef": ("flashinfer/runtime/schema.py", "SourceRef"),
    "ArtifactRef": ("flashinfer/runtime/schema.py", "ArtifactRef"),
    "WorkspaceFormula": ("flashinfer/runtime/schema.py", "WorkspaceFormula"),
    "KernelLaunchABI": ("flashinfer/runtime/schema.py", "KernelLaunchABI"),
    "KernelBinaryABI": ("flashinfer/runtime/schema.py", "KernelBinaryABI"),
    "KernelConstraints": ("flashinfer/runtime/schema.py", "KernelConstraints"),
    "KernelCapabilityBinding": ("flashinfer/runtime/schema.py", "KernelCapabilityBinding"),
    "GraphReplaySpec": ("flashinfer/runtime/schema.py", "GraphReplaySpec"),
    "JitSpec": ("flashinfer/jit/core.py", "JitSpec"),
    "BuildRecipe": ("flashinfer/jit/recipe.py", "BuildRecipe"),
    "RuntimeProvenance": ("flashinfer/jit/runtime.py", "RuntimeProvenance"),
    "JitLoadedModule": ("flashinfer/jit/loading.py", "JitLoadedModule"),
    "WorkloadSpec": ("flashinfer/runtime/schema.py", "WorkloadSpec"),
    "DeviceCapability": ("flashinfer/runtime/schema.py", "DeviceCapability"),
    "Plan": ("flashinfer/runtime/plan.py", "Plan"),
    "_WorkloadPlan": ("flashinfer/runtime/plan.py", "_WorkloadPlan"),
}

ROLES = {
    "InputSignature": "facade 创建本次调用的元数据快照；能力过滤、调优查询和决策缓存消费它。不是 Tensor 或产物。",
    "HardwareSignature": "hardware probe 创建 SoC / AI Core / UB 快照；过滤和调优匹配消费它，不包含 Tensor 地址。",
    "CapabilitySpec": "算子作者声明一个实现的支持域；unsupported_reasons(signature, hardware) 对照实际请求。",
    "InferenceSpec": "公共入口声明本次调用的精度、逐 Tensor 量化、Attention/KV、融合、MoE 等语义；参与能力过滤和缓存身份，不代表 kernel 已实现这些能力。",
    "CapabilityCase": "一个连贯的能力组合，内部条件 AND；CapabilitySpec.cases 之间 OR，再与外层条件 AND。不把 dtype、量化和 shape 组合展开为任意笛卡尔积。",
    "FeatureConstraint": "一个有类型的功能键及允许值；真实调用必须显式提供该特性。统一匹配器消费，不按 Attention/GEMM 编写 dispatcher 分支。",
    "FeatureDefinition": "功能目录定义分类、键、类型和允许取值；用于校验，不表示任何 kernel 已支持目录中的功能。",
    "ShapeRange": "CapabilitySpec 的一个 Tensor 轴闭区间；tensor_index/axis 负责轴映射，name 只用于诊断。",
    "KernelDescriptor": "provider/manifest 创建一条实现记录；registry 保存它，dispatcher 选择它的身份，执行层调用其 runner。",
    "ImplementationStack": "registry.stacks 从 descriptor 派生的来源视图；两级选型和解释使用它，不另设栈能力或优先级。",
    "CandidateReport": "过滤阶段创建一个候选的 accepted/reasons；DispatchDecision.reports 保留逐候选结论。",
    "StackReport": "解释阶段创建来源栈资格和选中状态；implementations 关联候选报告。accepted 不代表 selected。",
    "DispatchDecision": "dispatch 的最终输出；facade 读取 kernel_id，随后在 registry 中找到原 runner。它不保存输入 Tensor 或二进制。",
    "_TuningResolution": "调优数据库内部查询结果；record 是字典，source 标识 local/release，rejections 保存记录拒绝原因。",
    "JitSpec": "编译声明/runner 创建的编译单元身份；AOT 匹配和 JIT build/load 消费它，不等于一次调用的 InputSignature。",
    "BuildRecipe": "算子构建与 Tensor/host ABI 配方；冻结、编译、AOT runner 和 Tensor adapter 使用它。此处已展开继承字段。",
    "RuntimeProvenance": "AotKernelRunner 绑定执行资源时记录实际 aot/jit 来源；不改变 descriptor 的静态 compile_mode。",
    "JitLoadedModule": "loader 返回校验凭据和不透明执行对象；runner 使用 opaque_module 执行。",
    "SourceRef": "源码的语言、包内路径和 SHA256 引用；构造记录本身不读源码或编译。",
    "ArtifactRef": "已交付产物的路径、摘要、格式、SoC 与构建身份；加载器校验实际字节与声明一致。",
    "WorkspaceFormula": "规划接口用工作负载求 workspace 上界的公式；当前 Matmul 实际 workspace=0 由 host tiling 给出。",
    "KernelLaunchABI": "声明 host 入口、参数次序、可变参数和 stream 参数位置；连接 descriptor 与构建/加载契约。",
    "KernelBinaryABI": "描述二进制参数、tiling 与错误协议；属于交付/校验元数据，不是每次选择时重新运行的 kernel。",
    "KernelConstraints": "WorkloadSpec/DeviceCapability 规划选择路径的额外约束；与 Tensor dispatcher 的 CapabilitySpec 路径区分。",
    "KernelCapabilityBinding": "声明能力目录的 domain/profile/rule 与指纹身份；属于 descriptor 元数据和校验关系。",
    "GraphReplaySpec": "图执行声明与验证边界；当前 Matmul 保持 NOT_CAPTURABLE，不能从存在该类型推断可入图。",
    "WorkloadSpec": "显式 manifest/规划接口的工作负载模型；build_plan/registry.select 使用，不替代本例 Tensor signature。",
    "DeviceCapability": "显式规划接口的设备能力模型；支持版本、features 等，不等于 HardwareSignature 的三个字段。",
    "Plan": "wrapper 可持有 signature/hardware/decision/workspace；普通 Matmul facade 直接使用 decision，不构造 Plan。",
    "_WorkloadPlan": "build_plan 返回的内部规划对象；绑定 workload/capability/kernel 和两类 workspace 上界。",
}


def class_model(path, name):
    text = (ROOT / path).read_text(encoding="utf-8")
    tree = ast.parse(text)
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    node = classes[name]

    def declarations(current):
        result = {}
        for base in current.bases:
            if isinstance(base, ast.Name) and base.id in classes:
                result.update(declarations(classes[base.id]))
        for item in current.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                result[item.target.id] = {
                    "name": item.target.id, "type": ast.unparse(item.annotation),
                    "default": ast.unparse(item.value) if item.value is not None else None,
                    "declared_in": current.name, "line": item.lineno,
                }
        return result

    declared = declarations(node)
    module = importlib.import_module(path.removesuffix(".py").replace("/", "."))
    assert Path(module.__file__).resolve() == (ROOT / path).resolve(), path
    actual = getattr(module, name)
    assert is_dataclass(actual), name
    assert list(declared) == [field.name for field in fields(actual)], name
    assert actual.__dataclass_params__.frozen, name
    runtime_hints = get_type_hints(actual)
    for item, runtime_field in zip(declared.values(), fields(actual)):
        declared_type = eval(item["type"], vars(module))
        if isinstance(declared_type, str):
            declared_type = eval(declared_type, vars(module))
        assert declared_type == runtime_hints[item["name"]], (name, item["name"], "annotation")
        if item["default"] is None:
            assert runtime_field.default is MISSING and runtime_field.default_factory is MISSING
        else:
            declared_default = eval(item["default"], vars(module))
            if isinstance(declared_default, Field):
                assert declared_default.default == runtime_field.default
                assert declared_default.default_factory == runtime_field.default_factory
                assert declared_default.repr == runtime_field.repr and declared_default.compare == runtime_field.compare
            else:
                assert declared_default == runtime_field.default, (name, item["name"], "default")
    lines = ["@dataclass(frozen=True)", f"class {name}:"]
    for field in declared.values():
        default = "" if field["default"] is None else " = " + field["default"]
        lines.append(f"    {field['name']}: {field['type']}{default}")
    return {"path": path, "line": node.lineno, "kind": "frozen_dataclass",
            "fields": list(declared.values()), "field_count": len(declared),
            "definition": "\n".join(lines), "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
            "definition_note": "Complete field declarations, inherited fields flattened; methods/validation omitted."}


def plain(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    return value


def inspect_registry():
    path = "flashinfer/runtime/registry.py"
    tree = ast.parse((ROOT / path).read_text())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "KernelRegistry")
    init = next(item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == "__init__")
    members = [ast.unparse(item) for item in init.body if isinstance(item, (ast.Assign, ast.AnnAssign))]
    methods = [{"name": item.name, "signature": f"def {item.name}({ast.unparse(item.args)})", "line": item.lineno}
               for item in node.body if isinstance(item, ast.FunctionDef)]
    return {"path": path, "line": node.lineno, "kind": "mutable_container_class", "members": members, "methods": methods}


def inspect_controls():
    tree = ast.parse((ROOT / "flashinfer/runtime/dispatch.py").read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "dispatch")
    cache = next(item for item in tree.body if isinstance(item, ast.AnnAssign)
                 and isinstance(item.target, ast.Name) and item.target.id == "_decision_cache")
    keys = next(item for item in ast.walk(node) if isinstance(item, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "cache_key" for target in item.targets)
                and isinstance(item.value, ast.Tuple))
    return {"entry_signature": f"def dispatch({ast.unparse(node.args)}) -> {ast.unparse(node.returns)}",
            "policy_kind": "local value: Union[str, Backend], not a BackendPolicy class",
            "cache_container": ast.unparse(cache.annotation),
            "cache_key_components": [ast.unparse(item) for item in keys.value.elts],
            "registry_identity": "outer WeakKeyDictionary key, not another cache tuple field"}


def walkthrough():
    from flashinfer.runtime import HardwareSignature, InputSignature, KernelRegistry, dispatch, get_global_registry
    from flashinfer.runtime.dispatch import _sort_key

    global_registry = get_global_registry()
    before = tuple((d.kernel_id, d.fingerprint) for d in global_registry.descriptors())
    production = global_registry.descriptors(op="gemm.matmul")
    assert len(production) == 3
    from flashinfer.ascendc import load_contract
    recipe = load_contract("ascendc/gemm/matmul/native/op_package/Ascend950")[1]
    recipe.validate_tensor_metadata(
        shapes={"a": (32, 48), "b": (48, 64), "out": (32, 64)},
        dtypes={"a": "bfloat16", "b": "bfloat16", "out": "bfloat16"},
    )
    registry = KernelRegistry(replace(d, runner=None) for d in production)
    registry.freeze()
    for d in registry.descriptors():
        assert d.runner is None and d.fingerprint == global_registry.get(d.kernel_id).fingerprint
    sig = InputSignature("gemm.matmul", dtypes=("bfloat16",) * 3,
                         shapes=((32, 48), (48, 64), (32, 64)), layouts=("",) * 3,
                         physical_layouts=("ND",) * 3, specializations=(("tensor_contract", "supported"),))
    hw = HardwareSignature("Ascend950", 56, 221184)
    first = dispatch(sig, hw, backend="auto", registry=registry)
    second = dispatch(sig, hw, backend="auto", registry=registry)
    assert first.backend.value == "cann" and not first.cache_hit and second.cache_hit
    decisions = {"auto_first": plain(first), "auto_cached": plain(second)}
    for backend in ("cann", "ascendc", "triton"):
        decision = dispatch(sig, hw, backend=backend, registry=registry)
        assert decision.backend.value == backend
        decisions[backend] = plain(decision)
    assert all(report.accepted for report in first.reports)
    assert sum(report.selected for report in first.stack_reports) == 1
    assert tuple((d.kernel_id, d.fingerprint) for d in global_registry.descriptors()) == before
    return {"kind": "REAL_DISPATCH_WITH_PRODUCTION_METADATA_NO_RUNNER_EXECUTION",
            "signature": sig.to_dict(), "hardware": plain(hw),
            "recipe_tensor_metadata_check": "PASS",
            "registry": {"frozen": registry.frozen, "revision": registry.revision,
                         "descriptors": [{"kernel_id": d.kernel_id, "op": d.op, "backend": d.backend.value,
                                          "compile_mode": d.compile_mode.value, "priority": d.priority,
                                          "capability": d.capability.to_dict(), "runner_in_demo": None,
                                          "production_runner": (getattr(d.runner, "__qualname__", None)
                                                                or type(d.runner).__name__)} for d in production]},
            "implementation_stacks": [{"stack_id": stack.stack_id,
                                       "kernel_ids": [d.kernel_id for d in stack.descriptors]}
                                      for stack in registry.stacks(op=sig.op)],
            "sort_keys": {d.kernel_id: list(_sort_key(d)) for d in registry.descriptors()},
            "decisions": decisions, "dispatch_calls": len(decisions),
            "kernel_launches": 0, "tensor_allocations": 0, "global_registry_unchanged": True}


def run():
    settings = {"FLASHINFER_NPU_BACKEND": "auto", "FLASHINFER_NPU_JIT_DISABLE": "0",
                "FLASHINFER_NPU_TUNING_DB": "", "FLASHINFER_NPU_DISABLE_DISPATCH_CACHE": "0",
                "FLASHINFER_NPU_DISPATCH_ORDER": "external,cann,aot,jit,reference"}
    previous = {key: os.environ.get(key) for key in settings}
    os.environ.update(settings)
    try:
        pending = dict(MODEL_SPECS)
        models, enums = {}, {}
        while pending:
            name = next(iter(pending))
            path, symbol = pending.pop(name)
            info = class_model(path, symbol)
            models[name] = info
            owner = importlib.import_module(path.removesuffix(".py").replace("/", "."))
            for field in info["fields"]:
                for node in ast.walk(ast.parse(field["type"], mode="eval")):
                    if not isinstance(node, ast.Name):
                        continue
                    value = getattr(owner, node.id, None)
                    if not isinstance(value, type) or not value.__module__.startswith("flashinfer."):
                        continue
                    relative = value.__module__.replace(".", "/") + ".py"
                    if is_dataclass(value) and value.__name__ not in models:
                        pending.setdefault(value.__name__, (relative, value.__name__))
                    elif issubclass(value, Enum) and value.__name__ not in enums:
                        source = (ROOT / relative).read_text()
                        declaration = next(item for item in ast.parse(source).body
                                           if isinstance(item, ast.ClassDef) and item.name == value.__name__)
                        enums[value.__name__] = {
                            "path": relative, "line": declaration.lineno,
                            "bases": [ast.unparse(base) for base in declaration.bases],
                            "sha256": hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
                            "values": {key: member.value for key, member in value.__members__.items()},
                        }
        trace = walkthrough()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return {"status": "PASS", "models": models, "enums": enums, "registry_class": inspect_registry(),
            "controls": inspect_controls(), "walkthrough": trace,
            "core_model_count": len(MODEL_SPECS), "model_count": len(models),
            "field_count": sum(model["field_count"] for model in models.values()),
            "inspector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def write_reference(data, target):
    reverse = {name: [] for name in (*data["models"], *data["enums"])}
    for name, item in data["models"].items():
        for field in item["fields"]:
            names = {node.id for node in ast.walk(ast.parse(field["type"], mode="eval")) if isinstance(node, ast.Name)}
            for dependency in names & reverse.keys():
                reverse[dependency].append(f"{name}.{field['name']}")
    text = ["# Dispatcher 数据结构定义与关联索引", "",
            "本页从当前源码自动提取，字段逐项与实际 dataclass 核对。代码块列出完整字段（继承字段展开），省略方法、校验和 docstring；它们不是 C++ 内存布局。", "",
            "先读 PDF 的组件章节，再用本页查字段。`InputSignature` 是请求事实；`KernelRegistry` 是保存 `KernelDescriptor` 的容器；`dispatch()` 是函数，输出 `DispatchDecision`。", "",
            "[Matmul 接口与调用链](../matmul-backends.md) · [架构讲解 PDF](../presentations/dispatcher.pdf)", "",
            f"实际校验：{data['model_count']} 个 dataclass、{data['field_count']} 个字段、{len(data['enums'])} 个枚举；5 次真实 dispatch，包含决策缓存命中。样例未申请 Tensor、未执行 kernel，生产 registry 保持不变。", "",
            "## 容器与控制值", "",
            "`KernelRegistry` 是可变 Python 类，内部保存如下成员；内置登记冻结后通过只读查询使用，合法的 external/custom 登记更新 revision。", "", "```python",
            "class KernelRegistry:", "    def __init__(self, descriptors=()):",
            *["        " + line for line in data["registry_class"]["members"]], "```", "",
            "常用操作：`register`/`freeze` 完成初始化；`descriptors(op)` 按逻辑算子取候选；`stacks(op)` 派生来源视图；`get(kernel_id)` 连接最终决策与 runner。", "",
            "`policy` 是 `auto` 或 `Backend` 值，`order` 是档位元组，不存在专门的 BackendPolicy 类。调优 record 是字典，查询的公共返回值为 `Optional[str]` 冠军 ID。", "",
            "决策缓存外层是 `WeakKeyDictionary[KernelRegistry, Dict[tuple, DispatchDecision]]`；内层 key 的实际字段如下：", "", "```python", "cache_key = (",
            *["    " + field + "," for field in data["controls"]["cache_key_components"]], ")", "```", "",
            "## 一条 Matmul 请求的实际数据", "",
            "私有 registry 复制生产 Matmul 的三个 descriptor，并去掉 runner 以隔离执行；能力、身份与排序元数据保持相同。", "", "```json",
            json.dumps({"signature": data["walkthrough"]["signature"], "hardware": data["walkthrough"]["hardware"]}, ensure_ascii=False, indent=2), "```", "",
            "默认 auto 的实际结果为 CANN；三候选均 accepted，只有 cann 栈 selected。相同请求第二次 cache_hit=True；强制 cann/ascendc/triton 分别选择对应来源。", "",
            "## 完整 dataclass 字段", ""]
    for name, info in data["models"].items():
        text += [f"### {name}", "", ROLES.get(name, "该记录作为其他结构的字段使用，关联位置如下；具体校验与行为见源码。"), "",
                 f"源码：[{'%s:%s' % (info['path'], info['line'])}](../../{info['path']})。", "",
                 "被引用位置：" + ("、".join(f"`{value}`" for value in reverse[name]) if reverse[name] else "作为函数输入/输出或容器成员使用，见上方用途与源码。"), "",
                 "```python", info["definition"], "```", ""]
    text += ["## 引用的枚举", ""]
    for name, info in data["enums"].items():
        text += [f"### {name}", "", f"源码：[{info['path']}:{info['line']}](../../{info['path']})。", "",
                 "被引用位置：" + "、".join(f"`{value}`" for value in reverse[name]), "", "```python",
                 f"class {name}({', '.join(info['bases'])}):", *[f"    {key} = {value!r}" for key, value in info["values"].items()], "```", ""]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(text), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/dispatcher-docs/data-models.json"))
    parser.add_argument("--reference", type=Path, default=ROOT / "docs/generated/dispatcher-data-models.md")
    args = parser.parse_args()
    data = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_reference(data, args.reference)
    print(f"PASS: {data['model_count']} dataclass definitions / {data['field_count']} fields match runtime classes.")
    print(f"PASS: {data['walkthrough']['dispatch_calls']} real dispatch calls, including cache hit; no allocation/launch; global registry unchanged.")
    print(f"Evidence: {args.output}")


if __name__ == "__main__":
    main()
