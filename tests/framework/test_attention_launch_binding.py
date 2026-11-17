"""attention launch binding regression coverage."""


import unittest
from dataclasses import replace

from flashinfer.attention import (
    ATTENTION_AUXILIARY_VIEW_C_ABI,
    ATTENTION_RUN_OPTIONS_C_ABI,
    ATTENTION_TENSOR_VIEW_C_ABI,
    AttentionAddressBinding,
    AttentionAuxiliaryContract,
    AttentionAuxiliaryRole,
    AttentionAuxiliaryTensor,
    AttentionAuxiliaryViewPOD,
    AttentionFrameworkSession,
    AttentionHostBufferLease,
    AttentionKVCacheViewPOD,
    AttentionMode,
    AttentionPlanSpec,
    AttentionRunOptions,
    AttentionRunTensorContract,
    AttentionStorageLease,
    AttentionStorageLifetime,
    AttentionTensorAccessPolicy,
    AttentionTensorRole,
    AttentionTensorViewPOD,
    CustomMaskSpec,
    KVCacheView,
    PagedKVCacheSpec,
    PagedKVMetadata,
    PagedPrefillMetadata,
    PosEncodingMode,
    StreamContext,
    TensorView,
    contiguous_strides,
    dtype_itemsize,
    materialize_attention_auxiliary_view,
    materialize_attention_kv_cache_view,
    materialize_attention_tensor_view,
)
from flashinfer.runtime import SchemaError


def view(
    shape,
    storage_id,
    *,
    dtype="float32",
    writable=False,
    device="npu:0",
):
    shape = tuple(shape)
    return TensorView(
        shape,
        contiguous_strides(shape),
        dtype,
        device,
        storage_id,
        max(1, dtype_itemsize(dtype)) * _numel(shape),
        data_ptr_alignment=64,
        writable=writable,
    )


def _numel(shape):
    result = 1
    for dim in shape:
        result *= dim
    return result


def lease_for(value, address, *, writable=None):
    return AttentionStorageLease(
        "lease:" + value.storage_id,
        value.storage_id,
        value.device,
        address,
        value.storage_nbytes,
        64,
        "allocator:npu0",
        1,
        AttentionStorageLifetime.RUN,
        value.writable if writable is None else writable,
    )


def bind(role, value, address):
    return AttentionAddressBinding(role, value, lease_for(value, address))


def table_lease(name, address, size, *, device="npu:0"):
    return AttentionHostBufferLease(
        "lease:" + name,
        address,
        size,
        64,
        "host-arena",
        1,
        AttentionStorageLifetime.RUN,
        True,
    )


def masked_plan(*, profiler=False, alibi=False):
    spec = AttentionPlanSpec(
        mode=AttentionMode.BATCH_PREFILL_PAGED,
        num_qo_heads=2,
        num_kv_heads=1,
        head_dim_qk=2,
        q_dtype="float32",
        kv_dtype="float32",
        custom_mask=CustomMaskSpec(1, packed=True),
        use_profiler=profiler,
        pos_encoding_mode=(PosEncodingMode.ALIBI if alibi else PosEncodingMode.NONE),
        logits_soft_cap=5.0,
    )
    metadata = PagedPrefillMetadata(
        (0, 1), PagedKVMetadata((0, 1), (0,), (2,), 2)
    )
    return AttentionFrameworkSession(spec.mode).plan(spec, metadata)


class AttentionAuxiliaryContractTests(unittest.TestCase):
    def test_run_options_are_exact_64_byte_pod_and_plan_bound(self):
        options = AttentionRunOptions(2.0, 3.0, 4.0, 1.5)
        payload = options.pack()
        self.assertEqual(len(payload), ATTENTION_RUN_OPTIONS_C_ABI.size_bytes)
        self.assertEqual(len(payload), 64)
        self.assertEqual(AttentionRunOptions.from_bytes(payload), options)
        self.assertNotEqual(options.fingerprint, replace(options, q_scale=2.5).fingerprint)
        capped = masked_plan()
        uncapped = replace(capped, spec=replace(capped.spec, logits_soft_cap=None))
        with self.assertRaisesRegex(SchemaError, "capped plan"):
            options.validate_plan(uncapped)

    def test_auxiliary_roles_validate_shape_dtype_access_and_plan_presence(self):
        plan = masked_plan(profiler=True, alibi=True)
        auxiliary = AttentionAuxiliaryContract(
            (
                AttentionAuxiliaryTensor(
                    AttentionAuxiliaryRole.CUSTOM_MASK,
                    view((1,), "mask", dtype="uint8"),
                ),
                AttentionAuxiliaryTensor(
                    AttentionAuxiliaryRole.Q_SCALE,
                    view((2,), "q-scale"),
                ),
                AttentionAuxiliaryTensor(
                    AttentionAuxiliaryRole.PROFILER,
                    view((8,), "profiler", dtype="uint64", writable=True),
                ),
                AttentionAuxiliaryTensor(
                    AttentionAuxiliaryRole.ALIBI_SLOPES,
                    view((2,), "slopes"),
                ),
            )
        )
        auxiliary.validate_plan(plan, "npu:0")
        self.assertEqual(AttentionAuxiliaryContract.from_dict(auxiliary.to_dict()), auxiliary)
        rebound = AttentionAuxiliaryContract(
            tuple(
                replace(item, view=replace(item.view, storage_id="new:" + item.view.storage_id))
                for item in auxiliary.components
            )
        )
        self.assertEqual(rebound.fingerprint, auxiliary.fingerprint)
        with self.assertRaisesRegex(SchemaError, "canonical order"):
            AttentionAuxiliaryContract(tuple(reversed(auxiliary.components)))
        with self.assertRaisesRegex(SchemaError, "read-only"):
            AttentionAuxiliaryTensor(
                AttentionAuxiliaryRole.Q_SCALE,
                view((2,), "bad", writable=True),
            )
        with self.assertRaisesRegex(SchemaError, "custom-mask"):
            AttentionAuxiliaryContract(
                tuple(item for item in auxiliary.components if item.role != AttentionAuxiliaryRole.CUSTOM_MASK)
            ).validate_plan(plan, "npu:0")


class AttentionPODMaterializationTests(unittest.TestCase):
    def test_tensor_view_materializes_from_lease_and_strictly_round_trips(self):
        value = view((2, 3), "q")
        binding = bind("q", value, 0x1000)
        pod = materialize_attention_tensor_view(
            binding, AttentionTensorRole.Q, device_index=0
        )
        payload = pod.pack()
        self.assertEqual(len(payload), ATTENTION_TENSOR_VIEW_C_ABI.size_bytes)
        self.assertEqual(AttentionTensorViewPOD.from_bytes(payload), pod)
        self.assertEqual(pod.data_ptr, 0x1000)
        values = ATTENTION_TENSOR_VIEW_C_ABI.unpack(payload)
        values.pop("reserved")
        values["shape"] = tuple(values["shape"][:2]) + (1,) + tuple(values["shape"][3:])
        with self.assertRaisesRegex(SchemaError, "trailing"):
            AttentionTensorViewPOD.from_bytes(ATTENTION_TENSOR_VIEW_C_ABI.pack(values))

    def test_dense_kv_descriptor_binds_canonical_component_table(self):
        spec = PagedKVCacheSpec(
            1, 2, 1, 2, 2, "float32", structure="separate", device="npu:0"
        )
        key = view((1, 2, 1, 2), "key")
        value = view((1, 2, 1, 2), "value")
        kv = KVCacheView(spec, key, value)
        bindings = (
            bind("kv.key_storage", key, 0x1000),
            bind("kv.value_storage", value, 0x2000),
        )
        table = table_lease("kv-table", 0x3000, 2 * ATTENTION_TENSOR_VIEW_C_ABI.size_bytes)
        pod = materialize_attention_kv_cache_view(kv, bindings, table, 0)
        self.assertEqual(
            tuple(item.role_code for item in pod.components),
            (
                int(AttentionTensorRole.KV_KEY_STORAGE),
                int(AttentionTensorRole.KV_VALUE_STORAGE),
            ),
        )
        self.assertEqual(
            AttentionKVCacheViewPOD.from_bytes(pod.pack(), pod.component_blob), pod
        )
        with self.assertRaisesRegex(SchemaError, "missing"):
            materialize_attention_kv_cache_view(kv, bindings[:1], table, 0)

    def test_auxiliary_descriptor_handles_empty_and_nonempty_tables(self):
        empty_lease = table_lease("empty", 0, 0)
        empty = materialize_attention_auxiliary_view(
            AttentionAuxiliaryContract(), (), empty_lease, 0
        )
        self.assertEqual(empty.components_ptr, 0)
        self.assertEqual(
            AttentionAuxiliaryViewPOD.from_bytes(empty.pack(), b""), empty
        )

        mask = view((2,), "mask", dtype="uint8")
        auxiliary = AttentionAuxiliaryContract(
            (AttentionAuxiliaryTensor(AttentionAuxiliaryRole.CUSTOM_MASK, mask),)
        )
        table = table_lease("aux-table", 0x5000, ATTENTION_TENSOR_VIEW_C_ABI.size_bytes)
        pod = materialize_attention_auxiliary_view(
            auxiliary, (bind("aux.custom_mask", mask, 0x4000),), table, 0
        )
        self.assertEqual(len(pod.pack()), ATTENTION_AUXILIARY_VIEW_C_ABI.size_bytes)
        self.assertEqual(
            AttentionAuxiliaryViewPOD.from_bytes(pod.pack(), pod.component_blob), pod
        )

    def test_runtime_contract_detects_auxiliary_output_alias(self):
        plan = masked_plan(profiler=True)
        q = view((1, 2, 2), "q")
        spec = PagedKVCacheSpec(
            1, 2, 1, 2, 2, "float32", structure="separate", device="npu:0"
        )
        kv = KVCacheView(
            spec,
            view((1, 2, 1, 2), "key"),
            view((1, 2, 1, 2), "value"),
        )
        profiler = view((2,), "q", dtype="uint64", writable=True)
        auxiliary = AttentionAuxiliaryContract(
            (
                AttentionAuxiliaryTensor(
                    AttentionAuxiliaryRole.CUSTOM_MASK,
                    view((1,), "mask", dtype="uint8"),
                ),
                AttentionAuxiliaryTensor(AttentionAuxiliaryRole.PROFILER, profiler),
            )
        )
        contract = AttentionRunTensorContract(
            q, kv, StreamContext("npu:0", "stream"), auxiliary=auxiliary
        )
        with self.assertRaisesRegex(SchemaError, "output cannot alias"):
            contract.validate(AttentionTensorAccessPolicy(), plan=plan)


from flashinfer.attention import (
    ATTENTION_AUXILIARY_VIEW_C_ABI,
    ATTENTION_KV_CACHE_VIEW_C_ABI,
    ATTENTION_RUN_OPTIONS_C_ABI,
    ATTENTION_TENSOR_VIEW_C_ABI,
    AttentionGraphResourceContract,
    AttentionHostBufferLease,
    AttentionHostBufferRole,
    AttentionLaunchLeaseContract,
    AttentionLaunchPacket,
    AttentionRunOptions,
    AttentionStorageLease,
    AttentionStorageLifetime,
    AttentionStreamBinding,
    AttentionWorkspaceContract,
    ReferenceAttentionExecutor,
    ReferenceBuffer,
    ReferenceTensor,
    build_kernel_execution_identity,
    materialize_attention_launch_packet,
    select_attention_dispatch,
    validate_reference_attention_views,
    AttentionAddressBinding,
)
from tests.framework.test_attention_capability import bound_kernel, functional_profile, group_case, group_plan


def packet_fixture(return_lse=True):
    profile = functional_profile()
    kernel = bound_kernel(profile)
    plan = group_plan()
    receipt = select_attention_dispatch(
        plan, (profile,), (kernel,), profile.environment
    )
    _, case = group_case()
    result = ReferenceAttentionExecutor().execute(
        plan, case.trace.q, case.trace.kv_data, return_lse=return_lse
    )
    out = ReferenceBuffer.zeros(result.output.shape, result.output.dtype)
    lse = (
        ReferenceBuffer.zeros(result.lse.shape, result.lse.dtype)
        if result.lse is not None
        else None
    )
    float_workspace = ReferenceTensor.zeros(
        (receipt.float_workspace_bytes,), dtype="uint8"
    )
    int_workspace = ReferenceTensor.zeros(
        (receipt.int_workspace_bytes,), dtype="uint8"
    )
    workspace = AttentionWorkspaceContract(
        backend=receipt.backend.value,
        device="cpu",
        float_capacity_bytes=receipt.float_workspace_bytes,
        int_capacity_bytes=receipt.int_workspace_bytes,
        required_float_bytes=receipt.float_workspace_bytes,
        required_int_bytes=receipt.int_workspace_bytes,
        plan_generation=plan.generation,
    )
    tensors = validate_reference_attention_views(
        case.trace.q,
        case.trace.kv_data,
        out=out,
        lse=lse,
        workspace_float=float_workspace,
        workspace_int=int_workspace,
        plan=plan,
    )
    graph = AttentionGraphResourceContract.disabled(plan.spec.mode)
    identity = build_kernel_execution_identity(
        plan,
        workspace,
        graph,
        tensors,
        receipt,
        profile,
        kernel,
        profile.environment,
        return_lse=return_lse,
    )
    device_bindings = []
    for index, (role, view) in enumerate(tensors.named_views):
        lease = AttentionStorageLease(
            "device-lease:%s" % role,
            view.storage_id,
            view.device,
            0 if view.storage_nbytes == 0 else 0x100000 + index * 0x10000,
            view.storage_nbytes,
            64,
            "allocator:%s" % role,
            1,
            AttentionStorageLifetime.RUN,
            view.writable,
        )
        device_bindings.append(AttentionAddressBinding(role, view, lease))
    launch_lease = AttentionLaunchLeaseContract(
        identity.fingerprint,
        receipt.fingerprint,
        tensors.stream.device,
        tensors.stream.stream_id,
        tuple(device_bindings),
    )
    sizes = {
        AttentionHostBufferRole.Q_DESCRIPTOR: ATTENTION_TENSOR_VIEW_C_ABI.size_bytes,
        AttentionHostBufferRole.KV_DESCRIPTOR: ATTENTION_KV_CACHE_VIEW_C_ABI.size_bytes,
        AttentionHostBufferRole.KV_COMPONENTS: 6 * ATTENTION_TENSOR_VIEW_C_ABI.size_bytes,
        AttentionHostBufferRole.AUX_DESCRIPTOR: ATTENTION_AUXILIARY_VIEW_C_ABI.size_bytes,
        AttentionHostBufferRole.AUX_COMPONENTS: 0,
        AttentionHostBufferRole.RUN_OPTIONS: ATTENTION_RUN_OPTIONS_C_ABI.size_bytes,
        AttentionHostBufferRole.OUT_DESCRIPTOR: ATTENTION_TENSOR_VIEW_C_ABI.size_bytes,
        AttentionHostBufferRole.PLAN_METADATA: 65536,
    }
    if lse is not None:
        sizes[AttentionHostBufferRole.LSE_DESCRIPTOR] = ATTENTION_TENSOR_VIEW_C_ABI.size_bytes
    host_leases = {}
    for index, role in enumerate(sorted(sizes, key=lambda item: item.value)):
        capacity = sizes[role]
        host_leases[role] = AttentionHostBufferLease(
            "host-lease:%s" % role.value,
            0 if capacity == 0 else 0x10000000 + index * 0x200000,
            capacity,
            64,
            "host-arena",
            1,
            AttentionStorageLifetime.RUN,
            True,
        )
    stream = AttentionStreamBinding(
        "cpu", 0, tensors.stream.stream_id, 0xABC0, "host-test-runtime", 1
    )
    packet = materialize_attention_launch_packet(
        plan,
        tensors,
        identity,
        receipt,
        launch_lease,
        stream,
        host_leases,
    )
    return packet, plan, tensors, host_leases


class AttentionLaunchPacketTests(unittest.TestCase):
    def test_complete_packet_round_trip_freezes_all_thirteen_arguments(self):
        packet, _, _, _ = packet_fixture()
        restored = AttentionLaunchPacket.from_dict(packet.to_dict())
        self.assertEqual(restored, packet)
        self.assertEqual(restored.fingerprint, packet.fingerprint)
        self.assertEqual(
            tuple(
                name
                for name in packet.arguments.__dataclass_fields__
                if name != "schema_version"
            ),
            (
                "q",
                "kv",
                "aux",
                "run_options",
                "out",
                "lse",
                "plan_metadata",
                "plan_metadata_nbytes",
                "float_workspace",
                "float_workspace_nbytes",
                "int_workspace",
                "int_workspace_nbytes",
                "stream",
            ),
        )
        self.assertGreater(packet.arguments.plan_metadata_nbytes, 0)

    def test_optional_lse_uses_null_argument_and_no_host_descriptor(self):
        packet, _, _, _ = packet_fixture(return_lse=False)
        self.assertEqual(packet.arguments.lse, 0)
        self.assertNotIn(
            AttentionHostBufferRole.LSE_DESCRIPTOR,
            {item.role for item in packet.host_buffers},
        )
        self.assertEqual(AttentionLaunchPacket.from_dict(packet.to_dict()), packet)

    def test_scalar_or_metadata_content_mutation_is_rejected(self):
        packet, _, _, _ = packet_fixture()
        buffers = list(packet.host_buffers)
        index = next(
            i
            for i, item in enumerate(buffers)
            if item.role == AttentionHostBufferRole.RUN_OPTIONS
        )
        buffers[index] = replace(
            buffers[index], content=AttentionRunOptions(q_scale=2.0).pack()
        )
        with self.assertRaisesRegex(SchemaError, "run-options fingerprint"):
            replace(packet, host_buffers=tuple(buffers))

        buffers = list(packet.host_buffers)
        index = next(
            i
            for i, item in enumerate(buffers)
            if item.role == AttentionHostBufferRole.PLAN_METADATA
        )
        corrupted = bytearray(buffers[index].content)
        corrupted[-1] ^= 1
        buffers[index] = replace(buffers[index], content=bytes(corrupted))
        with self.assertRaises(SchemaError):
            replace(packet, host_buffers=tuple(buffers))

    def test_component_table_address_and_workspace_pointer_cannot_be_partially_rebound(self):
        packet, _, _, _ = packet_fixture()
        buffers = list(packet.host_buffers)
        index = next(
            i
            for i, item in enumerate(buffers)
            if item.role == AttentionHostBufferRole.KV_COMPONENTS
        )
        buffers[index] = replace(
            buffers[index],
            lease=replace(
                buffers[index].lease,
                base_address=buffers[index].lease.base_address + 0x1000,
                allocation_generation=2,
            ),
        )
        with self.assertRaisesRegex(SchemaError, "component table pointer"):
            replace(packet, host_buffers=tuple(buffers))

        changed_args = replace(
            packet.arguments,
            float_workspace=0xDEAD00,
            float_workspace_nbytes=1,
        )
        with self.assertRaisesRegex(SchemaError, "workspace_float"):
            replace(packet, arguments=changed_args)

    def test_host_and_device_memory_domains_are_not_interchangeable(self):
        packet, plan, tensors, host_leases = packet_fixture()
        device_lease = packet.launch_lease.bindings[0].lease
        bad = dict(host_leases)
        bad[AttentionHostBufferRole.KV_COMPONENTS] = device_lease
        with self.assertRaises(TypeError):
            materialize_attention_launch_packet(
                plan,
                tensors,
                packet.execution_identity,
                packet.dispatch_receipt,
                packet.launch_lease,
                packet.stream_binding,
                bad,
            )

    def test_stream_generation_and_host_allocation_generation_change_packet_identity(self):
        packet, _, _, _ = packet_fixture()
        changed_stream = replace(packet.stream_binding, runtime_generation=2)
        changed = replace(
            packet,
            stream_binding=changed_stream,
            arguments=replace(packet.arguments, stream=changed_stream.stream_handle),
        )
        self.assertNotEqual(changed.fingerprint, packet.fingerprint)

        buffers = list(packet.host_buffers)
        buffers[0] = replace(
            buffers[0],
            lease=replace(buffers[0].lease, allocation_generation=2),
        )
        changed = replace(packet, host_buffers=tuple(buffers))
        self.assertNotEqual(changed.fingerprint, packet.fingerprint)

    def test_stream_id_cannot_drift_with_coordinated_lease_and_handle_changes(self):
        packet, _, _, _ = packet_fixture()
        changed_stream = replace(
            packet.stream_binding,
            stream_id="host-synchronous-other",
            stream_handle=packet.stream_binding.stream_handle + 0x100,
        )
        changed_lease = replace(
            packet.launch_lease,
            stream_id=changed_stream.stream_id,
        )
        changed_arguments = replace(
            packet.arguments,
            stream=changed_stream.stream_handle,
        )
        with self.assertRaisesRegex(SchemaError, "stream context fingerprint"):
            replace(
                packet,
                stream_binding=changed_stream,
                launch_lease=changed_lease,
                arguments=changed_arguments,
            )
