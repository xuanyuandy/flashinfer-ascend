"""Triton implementation declaration for norm.gemma_fused_add_rmsnorm."""
from dataclasses import replace
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.runtime.manifest import descriptor_from_dict
from .runner import _gemma_fused_add_rmsnorm_triton

def descriptor():
    value = descriptor_from_dict({'kernel_id': 'triton/norm/gemma_fused_add_rmsnorm_v1',
     'op': 'norm.gemma_fused_add_rmsnorm',
     'backend': 'triton',
     'constraints': {'supported_socs': [],
                     'dtype_signatures': [],
                     'layout_signatures': [],
                     'required_features': [],
                     'quant_storage_dtypes': [],
                     'deterministic': None},
     'workspace': {'constant_bytes': 0, 'dynamic_coefficients': [], 'alignment': 32},
     'int_workspace': {'constant_bytes': 0, 'dynamic_coefficients': [], 'alignment': 32},
     'artifact': None,
     'launch_abi': None,
     'binary_abi': None,
     'priority': 0,
     'op_schema_version': 1,
     'tiling_schema_version': 1,
     'capability_binding': None,
     'compile_mode': 'jit',
     'capability': {'op': 'norm.gemma_fused_add_rmsnorm',
                    'dtypes': ['float16', 'bfloat16', 'float32'],
                    'layouts': [],
                    'soc_models': ['Ascend910B', 'Ascend950'],
                    'min_aicore': 0,
                    'specializations': []}})
    return replace(value, runner=_gemma_fused_add_rmsnorm_triton)

IMPLEMENTATION = ImplementationDefinition('triton/norm/gemma_fused_add_rmsnorm_v1', 'norm.gemma_fused_add_rmsnorm', "triton", descriptor, origin="flashinfer.triton.norm")
