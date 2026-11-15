"""Named device entries bind source, OpDef codegen and the public build recipe."""

import importlib.util
import json
import os
from pathlib import Path
import re

import pytest

from flashinfer.ascendc import load_contract
from flashinfer.jit import DiskJitBuildStore, load_build_recipes
from flashinfer.operators.implementation import implementations

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('entry_adapter_generator', ROOT / 'csrc/cmake/operator/aclnn_adapter.py')
generator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generator)


def test_current_device_sources_define_their_declared_literal_entry():
    recipes = [d.prepare_outputs(ROOT)['recipe'] for d in implementations(backend='ascendc')]
    recipes.extend(load_build_recipes(ROOT / 'csrc/recipes/matmul_probe.json'))
    names = {}
    for recipe in recipes:
        source = (ROOT / recipe.source.locator).read_text()
        assert 'FI_DEVICE_ENTRY' not in source
        assert re.search(r'\b__global__\s+__aicore__\s+void\s+'
                         + re.escape(recipe.device_entry) + r'\s*\(', source), recipe.kernel_id
        previous = names.setdefault(recipe.device_entry, recipe.source.locator)
        assert previous == recipe.source.locator, 'different device sources share an entry name'


@pytest.mark.parametrize('entry', ['fi_matmul', 'custom_named_entry'])
def test_opdef_entry_validation_accepts_the_authored_name(tmp_path, entry):
    info = tmp_path / 'ops-info.ini'
    info.write_text('[FiMatmul]\nopInterface.value=' + entry + '\n')
    generator.validate_device_entry(dict(package_op_type='FiMatmul', device_entry=entry), info)


@pytest.mark.parametrize('contents', [
    '[FiMatmul]\nopInterface.value=another_entry\n',
    '[AnotherOp]\nopInterface.value=fi_matmul\n',
    '[FiMatmul]\nopFile.value=fi_matmul\n',
])
def test_wrong_or_missing_opdef_entry_is_rejected(tmp_path, contents):
    info = tmp_path / 'ops-info.ini'
    info.write_text(contents)
    with pytest.raises(ValueError, match='device entry does not match'):
        generator.validate_device_entry(dict(package_op_type='FiMatmul', device_entry='fi_matmul'), info)


def test_both_v2_cmake_paths_require_codegen_entry_validation():
    for mode in ('operator', 'op_package'):
        text = (ROOT / 'csrc/cmake' / mode / 'host/CMakeLists.txt').read_text()
        assert '--ops-info "${ASCEND_AUTOGEN_PATH}/aic-${ASCEND_COMPUTE_UNIT}-ops-info.ini"' in text


@pytest.fixture(scope='module')
def entry_packages(tmp_path_factory):
    from flashinfer.aot.build import build_production_packages
    directory = tmp_path_factory.mktemp('named-entries')
    cache = Path(os.environ.get('FLASHINFER_AOT_CACHE_DIR', directory / 'cache'))
    packages = build_production_packages(output_dir=directory / 'release', cache_dir=cache)
    return {json.loads((p / 'manifest.json').read_text())['spec']['kernel_id']: p
            for p in packages}, cache


@pytest.mark.real_compile
@pytest.mark.parametrize('kernel_id', [d.kernel_id for d in implementations(backend='ascendc')])
def test_named_entries_compile_and_bind_their_actual_opdef(kernel_id, entry_packages):
    from flashinfer.jit.cache import verify_build_directory
    from flashinfer.jit.op_package import verify_op_package, package_target
    packages, cache = entry_packages
    release = packages[kernel_id]
    record = verify_build_directory(release)
    _, recipe, spec = load_contract(kernel_id)
    metadata, _ = verify_op_package(release, spec)
    compute = package_target(recipe.target_soc)[0]
    config = release / metadata['vendor_path'] / ('op_impl/ai_core/tbe/config/' + compute
              + '/aic-' + compute + '-ops-info.json')
    assert json.loads(config.read_text())[recipe.package_op_type]['opInterface']['value'] == recipe.device_entry
    log = DiskJitBuildStore(cache).directory(record) / 'build.log'
    assert 'FI_DEVICE_ENTRY' not in log.read_text(errors='replace')


@pytest.mark.real_compile
@pytest.mark.parametrize('soc', ['Ascend910B', 'Ascend910C', 'Ascend950'])
def test_literal_native_probe_builds_for_each_target(soc, tmp_path):
    from flashinfer.aot.build import compile_and_package
    from flashinfer.jit.core import JitSpec
    from flashinfer.jit.op_package import verify_op_package
    release, = compile_and_package(ROOT / 'examples/ascendc_aot_native.json', targets=(soc,),
        cache_dir=os.environ.get('FLASHINFER_AOT_CACHE_DIR', tmp_path / 'cache'),
        output_dir=tmp_path / 'release')
    spec = JitSpec.from_dict(json.loads((release / 'manifest.json').read_text())['spec'])
    _, recipe = verify_op_package(release, spec)
    assert recipe.device_entry == 'fi_matmul_probe_native' and spec.target_soc == soc
