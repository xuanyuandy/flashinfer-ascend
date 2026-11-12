"""Ascend C authoring inputs. These fields do not belong to logical operators."""

from dataclasses import dataclass

from flashinfer.runtime import CapabilitySpec, SchemaError


@dataclass(frozen=True)
class BuildConfig:
    device: str
    host: tuple
    op_type: str
    tiling: str
    includes: tuple = ('op_kernel', 'op_host')
    cmake: object = None
    implementation: str = 'native'
    dependencies: tuple = ()
    task_type: str = 'aic_aiv'
    sync_policy: str = 'cross_core_flags'
    workspace_bytes: object = 'dynamic'
    version: int = 2
    definitions: tuple = ()
    compile_options: tuple = ()
    link_options: tuple = ()
    device_entry: object = None
    host_entry: object = None
    launch_abi: object = None
    package_vendor: object = None
    recipe_id: object = None
    tiling_mode: str = 'host_dynamic'
    block_dim: int = 1


@dataclass(frozen=True)
class NativeDefinition:
    name: str
    kernel_id: str
    target_soc: str
    source_dir: str
    capability: CapabilitySpec
    build: BuildConfig
    tensors: tuple
    shapes: tuple
    dtype_signatures: tuple = ()
    attributes: tuple = ()
    workspace_bound: int = 0
    allow_jit_fallback: bool = False
    descriptor_metadata: object = None

    def __post_init__(self):
        if (not self.kernel_id.startswith('ascendc/') or not isinstance(self.capability, CapabilitySpec)
                or not isinstance(self.build, BuildConfig) or self.workspace_bound < 0):
            raise SchemaError('invalid Ascend C implementation definition')

    @property
    def op(self):
        return self.capability.op
