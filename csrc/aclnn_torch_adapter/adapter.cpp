// Torch -> ACLNN adapter. The conversion/command structure follows the
// vLLM-Ascend adapter; this implementation uses torch_npu's public format API
// instead of copying its private NPUStorageImpl layout. Eager execution only.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <acl/acl.h>
#include <algorithm>
#include <dlfcn.h>
#include <ffi.h>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/framework/OpCommand.h"

struct aclTensor;
struct aclOpExecutor;

namespace {
using CreateTensor = aclTensor* (*)(const int64_t*, uint64_t, aclDataType,
    const int64_t*, int64_t, aclFormat, const int64_t*, uint64_t, void*);
using DestroyTensor = int (*)(const aclTensor*);
using Launch = int (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);

struct TensorApi {
    void* library = dlopen("libnnopbase.so", RTLD_NOW | RTLD_LOCAL);
    CreateTensor create = nullptr;
    DestroyTensor destroy = nullptr;
    TensorApi() {
        TORCH_CHECK(library, "ACLNN tensor runtime is unavailable: libnnopbase.so");
        create = reinterpret_cast<CreateTensor>(dlsym(library, "aclCreateTensor"));
        destroy = reinterpret_cast<DestroyTensor>(dlsym(library, "aclDestroyTensor"));
        TORCH_CHECK(create && destroy, "ACLNN tensor descriptor symbols are unavailable");
    }
};
TensorApi& tensor_api() { static TensorApi api; return api; }

struct Binding {
    std::string name;
    void* workspace;
    Launch launch;
    size_t inputs;
    size_t outputs;
};
std::mutex bindings_mutex;
std::unordered_map<std::string, Binding> bindings;
thread_local std::string last_error;

aclDataType acl_dtype(at::ScalarType dtype) {
    switch (dtype) {
        case at::kFloat: return ACL_FLOAT;
        case at::kHalf: return ACL_FLOAT16;
        case at::kBFloat16: return ACL_BF16;
        case at::kDouble: return ACL_DOUBLE;
        case at::kChar: return ACL_INT8;
        case at::kByte: return ACL_UINT8;
        case at::kShort: return ACL_INT16;
        case at::kInt: return ACL_INT32;
        case at::kLong: return ACL_INT64;
        case at::kBool: return ACL_BOOL;
        default: TORCH_CHECK(false, "Unsupported ACLNN tensor dtype: ", dtype);
    }
}

struct CallState {
    std::vector<at::Tensor> tensors;
    std::vector<aclTensor*> handles;
    at::Tensor workspace;
    ~CallState() { release(); }
    int release() noexcept {
        int status = 0;
        for (auto it = handles.rbegin(); it != handles.rend(); ++it) {
            const int current = tensor_api().destroy(*it);
            if (current && !status) status = current;
        }
        handles.clear();
        return status;
    }
};

void execute(const std::string& token, at::TensorList inputs, at::TensorList outputs) {
    Binding binding;
    {
        std::lock_guard<std::mutex> lock(bindings_mutex);
        const auto found = bindings.find(token);
        TORCH_CHECK(found != bindings.end(), "Unknown verified ACLNN binding");
        binding = found->second;
    }
    TORCH_CHECK(inputs.size() == binding.inputs && outputs.size() == binding.outputs,
                "ACLNN binding argument count mismatch");
    TORCH_CHECK(!inputs.empty() || !outputs.empty(), "ACLNN requires tensor arguments");
    const auto device = (inputs.empty() ? outputs.front() : inputs.front()).device();
    TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1, "ACLNN requires NPU tensors");
    c10_npu::NPUGuard guard(device);
    const auto stream = c10_npu::getCurrentNPUStream(device.index());
    const auto raw_stream = stream.stream(false);
    auto state = std::make_shared<CallState>();
    state->tensors.insert(state->tensors.end(), inputs.begin(), inputs.end());
    state->tensors.insert(state->tensors.end(), outputs.begin(), outputs.end());
    for (const auto& tensor : state->tensors) {
        TORCH_CHECK(tensor.device() == device, "ACLNN tensors must share one device");
        TORCH_CHECK(tensor.is_contiguous() && at_npu::native::get_npu_format(tensor) == ACL_FORMAT_ND,
                    "ACLNN adapter requires contiguous ND tensors");
        c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
    }
    for (size_t out = binding.inputs; out < state->tensors.size(); ++out) {
        const auto lo = reinterpret_cast<uintptr_t>(state->tensors[out].data_ptr());
        const auto hi = lo + state->tensors[out].nbytes();
        for (size_t i = 0; i < state->tensors.size(); ++i) {
            if (i == out) continue;
            const auto other = reinterpret_cast<uintptr_t>(state->tensors[i].data_ptr());
            TORCH_CHECK(std::max(lo, other) >= std::min(hi, other + state->tensors[i].nbytes()),
                        "ACLNN output tensors must not overlap other arguments");
        }
    }
    // The state owns tensors, descriptors and workspace until device completion.
    // Keep the existing synchronous eager lifetime contract; no capture claim.
    auto call = [state, binding, stream, raw_stream]() -> int {
        auto& api = tensor_api();
        try {
            for (const auto& tensor : state->tensors) {
                const int64_t storage_size = tensor.storage().nbytes() / tensor.element_size();
                auto* handle = api.create(tensor.sizes().data(), tensor.dim(), acl_dtype(tensor.scalar_type()),
                    tensor.strides().data(), tensor.storage_offset(), ACL_FORMAT_ND,
                    &storage_size, 1, const_cast<void*>(tensor.storage().data()));
                TORCH_CHECK(handle, "aclCreateTensor failed for ", binding.name);
                state->handles.push_back(handle);
            }
            uint64_t workspace_bytes = 0;
            aclOpExecutor* executor = nullptr;
            auto* workspace_pointer = &workspace_bytes;
            auto** executor_pointer = &executor;
            std::vector<ffi_type*> types(state->handles.size() + 2, &ffi_type_pointer);
            std::vector<void*> values;
            for (auto& handle : state->handles) values.push_back(&handle);
            values.push_back(&workspace_pointer);
            values.push_back(&executor_pointer);
            ffi_cif signature;
            TORCH_CHECK(ffi_prep_cif(&signature, FFI_DEFAULT_ABI, types.size(), &ffi_type_uint32,
                                    types.data()) == FFI_OK, "Invalid ACLNN workspace signature");
            ffi_arg status = 0;
            ffi_call(&signature, FFI_FN(binding.workspace), &status, values.data());
            TORCH_CHECK(status == 0 && executor, binding.name, "GetWorkspaceSize returned ", status);
            TORCH_CHECK(workspace_bytes <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                        "ACLNN workspace exceeds tensor allocation range");
            if (workspace_bytes) {
                state->workspace = at::empty({static_cast<int64_t>(workspace_bytes)},
                    state->tensors.front().options().dtype(at::kByte));
                c10_npu::NPUCachingAllocator::recordStream(state->workspace.storage().data_ptr(), stream);
            }
            const auto launch_status = binding.launch(
                workspace_bytes ? state->workspace.data_ptr() : nullptr, workspace_bytes, executor, raw_stream);
            TORCH_CHECK(launch_status == 0, binding.name, " returned ", launch_status);
            const auto sync_status = aclrtSynchronizeStream(raw_stream);
            TORCH_CHECK(sync_status == ACL_SUCCESS, "ACLNN eager stream synchronization returned ", sync_status);
            const auto destroy_status = state->release();
            TORCH_CHECK(destroy_status == 0, "aclDestroyTensor returned ", destroy_status);
        } catch (...) {
            aclrtSynchronizeStream(raw_stream);
            state->release();
            throw;
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi(binding.name, call, true);
}
} // namespace

// Internal registration is called only after the Python verified package loader.
// Public Torch schemas contain no raw function pointers or arbitrary library paths.
extern "C" __attribute__((visibility("default"))) int fi_aclnn_register(
    const char* token, const char* name, void* workspace, void* launch,
    uint64_t inputs, uint64_t outputs) noexcept {
    try {
        TORCH_CHECK(token && name && workspace && launch && (inputs || outputs),
                    "Invalid verified ACLNN registration");
        std::lock_guard<std::mutex> lock(bindings_mutex);
        auto result = bindings.emplace(token, Binding{name, workspace, reinterpret_cast<Launch>(launch),
                                                       static_cast<size_t>(inputs), static_cast<size_t>(outputs)});
        if (!result.second) {
            const auto& prior = result.first->second;
            TORCH_CHECK(prior.name == name && prior.workspace == workspace &&
                        prior.launch == reinterpret_cast<Launch>(launch) &&
                        prior.inputs == inputs && prior.outputs == outputs,
                        "Cannot replace an immutable ACLNN binding");
        }
        last_error.clear();
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}
extern "C" __attribute__((visibility("default"))) const char* fi_aclnn_last_error() noexcept {
    return last_error.c_str();
}

TORCH_LIBRARY(flashinfer_aclnn, m) {
    m.def("_execute(str token, Tensor[] inputs, Tensor(a!)[] outputs) -> ()");
}
TORCH_LIBRARY_IMPL(flashinfer_aclnn, PrivateUse1, m) {
    m.impl("_execute", TORCH_FN(execute));
}
