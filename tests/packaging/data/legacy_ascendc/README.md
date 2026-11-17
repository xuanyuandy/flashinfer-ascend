# Frozen internal Ascend C probes

These files preserve the seven original v1 recipes and their complete source
closure for compatibility and raw host-launch tests. Public recipes are v2
only. This directory is not included in wheels or source build inputs.

Do not update these recipes' serialization or source digests when editing the
public probes. Their exact fingerprints and v1 rejection rules remain tested.
The copied scripts use private compatibility helpers to execute the original
host_shared/op_package compile, FakeTensor, namespace and numerical checks.
