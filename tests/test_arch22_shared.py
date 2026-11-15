"""Shared sources and distribution must preserve strict physical SoC identities."""

import json
import os
from pathlib import Path

import pytest

from flashinfer._wheel import production_kernel_ids
from flashinfer.ascendc import load_contract
from flashinfer.jit.op_package import package_target
from flashinfer.operators.implementation import implementations
from flashinfer.runtime import DispatchError, HardwareSignature, dispatch


@pytest.mark.parametrize("name", ["single_prefill_catlass", "grouped_matmul_swiglu_quant",
                                 "sparse_flash_attention"])
def test_arch22_variants_share_sources_and_runner_but_not_frozen_identity(name):
    variants = {d.native.target_soc: d for d in implementations(backend="ascendc")
                if d.operator.name == name}
    b, c = (variants[soc] for soc in ("Ascend910B", "Ascend910C"))
    assert b.runner_factory is c.runner_factory
    assert b.native.source_dir == c.native.source_dir
    assert b.native.build == c.native.build
    assert b.native.tensors == c.native.tensors
    db, rb, sb = load_contract(b.kernel_id)
    dc, rc, sc = load_contract(c.kernel_id)
    assert rb.source == rc.source and rb.headers == rc.headers
    assert rb.npu_arch == rc.npu_arch == "dav-2201"
    assert db.target_soc == rb.target_soc == sb.target_soc == "Ascend910B"
    assert dc.target_soc == rc.target_soc == sc.target_soc == "Ascend910C"
    assert db.fingerprint != dc.fingerprint and sb.fingerprint != sc.fingerprint
    with pytest.raises(ValueError, match="bind"):
        rb.validate_descriptor(dc)


def test_shared_wheel_inventory_is_exactly_both_physical_targets():
    expected = {d.kernel_id for d in implementations(backend="ascendc")
                if d.native.target_soc in ("Ascend910B", "Ascend910C")}
    assert set(production_kernel_ids("Ascend910B")) == expected
    assert set(production_kernel_ids("Ascend910C")) == expected
    assert len(expected) == 6


@pytest.mark.parametrize("alias", ["Ascend910C", "Ascend910_9362", "Ascend910_9391", "Ascend910_9392"])
def test_a3_compute_unit_is_not_aliased_to_a2(alias):
    assert package_target(alias) == ("ascend910_93", "dav-2201")
    assert package_target("Ascend910B3") == ("ascend910b", "dav-2201")


@pytest.mark.parametrize("soc", ["Ascend910B", "Ascend910C"])
@pytest.mark.parametrize("name", ["grouped_matmul_swiglu_quant", "sparse_flash_attention"])
def test_shared_public_semantics_route_to_only_the_physical_soc(name, soc):
    if name == "grouped_matmul_swiglu_quant":
        from tests.test_grouped_matmul_swiglu_quant import inputs, request
        args, options = inputs()
        signature = request(args, options, layout="NZ")
        prefix = "ascendc/gemm/grouped_matmul_swiglu_quant/native/op_package/"
    else:
        from tests.test_sparse_flash_attention import arguments, OPERATOR
        from flashinfer.operators.runtime import signature as make_signature
        signature, _, _ = make_signature(OPERATOR.op, arguments())
        prefix = "ascendc/attention/sparse_flash_attention/native/"
    hardware = HardwareSignature(soc, 24)
    assert dispatch(signature, hardware, backend="ascendc").kernel_id == prefix + soc
    other = "Ascend910C" if soc == "Ascend910B" else "Ascend910B"
    with pytest.raises(DispatchError):
        dispatch(signature, hardware, kernel_id=prefix + other)


@pytest.mark.real_compile
@pytest.mark.parametrize("name", ["grouped_matmul_swiglu_quant", "sparse_flash_attention"])
def test_a3_shared_arch22_native_package_cross_compile(name, tmp_path):
    from flashinfer.aot.build import build_production_packages
    from flashinfer.jit.cache import verify_build_directory
    from flashinfer.jit.op_package import verify_op_package

    kernel_id = next(d.kernel_id for d in implementations(backend="ascendc")
                     if d.operator.name == name and d.native.target_soc == "Ascend910C")
    release, = build_production_packages(kernel_ids=(kernel_id,), output_dir=tmp_path / "release",
        cache_dir=os.environ.get("FLASHINFER_AOT_CACHE_DIR", tmp_path / "cache"))
    _, recipe, spec = load_contract(kernel_id)
    verify_build_directory(release)
    metadata, packaged_recipe = verify_op_package(release, spec)
    assert packaged_recipe == recipe and metadata["schema_version"] == 2
    binding = json.loads((release / metadata["native_adapter"]).read_text())
    assert binding["kernel_id"] == kernel_id
    assert recipe.target_soc == "Ascend910C"
    assert any("ascend910_93" in str(p) for p in Path(release).rglob("*.json"))


def test_target_mapping_is_part_of_the_frozen_driver_identity(tmp_path, monkeypatch):
    from flashinfer.jit import ascendc

    package = tmp_path / 'flashinfer'
    driver = package / 'jit/ascendc.py'
    target_mapping = package / 'runtime/targets.py'
    driver.parent.mkdir(parents=True)
    target_mapping.parent.mkdir(parents=True)
    driver.write_text('# compiler driver\n')
    driver.with_name('op_package.py').write_text('# package target consumer\n')
    target_mapping.write_text('SOC_TARGETS = {"Ascend910C": "ascend910_93"}\n')
    monkeypatch.setattr(ascendc, '__file__', str(driver))
    before = ascendc._driver_identity()
    target_mapping.write_text('SOC_TARGETS = {"Ascend910C": "changed_compute_unit"}\n')
    assert ascendc._driver_identity() != before
