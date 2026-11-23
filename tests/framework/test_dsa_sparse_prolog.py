"""Registration of DsaSparseProlog without launching the device kernel."""

import inspect
from pathlib import Path

import flashinfer
from flashinfer.ascendc.attention.dsa_sparse_prolog.implementation import IMPLEMENTATIONS
from flashinfer.operators.attention.dsa_sparse_prolog.operator import OPERATOR

ROOT = Path(__file__).resolve().parents[2]


def test_public_signature_and_native_identity():
    assert flashinfer.dsa_sparse_prolog is flashinfer.attention.dsa_sparse_prolog
    assert inspect.signature(flashinfer.dsa_sparse_prolog) == OPERATOR.public_signature
    assert tuple(item.native.kernel_id for item in IMPLEMENTATIONS) == (
        "ascendc/attention/dsa_sparse_prolog/native/Ascend910B",
        "ascendc/attention/dsa_sparse_prolog/native/Ascend910C",
    )
    prepared = IMPLEMENTATIONS[0].prepare_outputs(ROOT)
    recipe = prepared["recipe"]
    assert recipe.package_op_type == "DsaSparseProlog"
    assert recipe.device_entry == "dsa_sparse_prolog"
    assert recipe.host_entry == "aclnnDsaSparseProlog"
    assert recipe.tiling_host_entry == "TilingDsaSparseProlog"
