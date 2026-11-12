// Torch -> ACLNN adapter. The conversion/command structure follows the
// vLLM-Ascend adapter; this implementation uses torch_npu's public format API
// instead of copying its private NPUStorageImpl layout. Eager execution only.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <acl/acl.h>
#include <algorithm>
#include <atomic>
#include <dlfcn.h>
#include <ffi.h>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
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
    std::vector<ffi_type*> argument_types;
    ffi_cif workspace_signature;
    std::atomic<uint64_t> last_workspace_bytes{0};
    std::atomic<bool> workspace_observed{false};
    std::vector<int64_t> formats, conversions, optional, attributes;

    Binding(std::string op_name, void* query, Launch run, size_t input_count, size_t output_count)
        : name(std::move(op_name)), workspace(query), launch(run), inputs(input_count),
          outputs(output_count), argument_types(input_count + output_count + 2, &ffi_type_pointer) {
        TORCH_CHECK(ffi_prep_cif(&workspace_signature, FFI_DEFAULT_ABI, argument_types.size(),
                                &ffi_type_uint32, argument_types.data()) == FFI_OK,
                    "Invalid ACLNN workspace signature");
    }
};
std::mutex bindings_mutex;
// A binding never moves: libffi retains its argument_types allocation.
// Published signatures are immutable and shared by concurrent invocations.
std::unordered_map<std::string, std::shared_ptr<Binding>> bindings;
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
            if (!*it) continue;
            const int current = tensor_api().destroy(*it);
            if (current && !status) status = current;
        }
        handles.clear();
        return status;
    }
};

void execute(const std::string& token, at::TensorList inputs, at::TensorList outputs) {
    std::shared_ptr<Binding> binding;
    {
        std::lock_guard<std::mutex> lock(bindings_mutex);
        const auto found = bindings.find(token);
        TORCH_CHECK(found != bindings.end(), "Unknown verified ACLNN binding");
        binding = found->second;
    }
    TORCH_CHECK(inputs.size() == binding->inputs && outputs.size() == binding->outputs,
                "ACLNN binding argument count mismatch");
    TORCH_CHECK(binding->formats.empty(), "Extended ACLNN bindings require _execute_extended");
    TORCH_CHECK(!inputs.empty() || !outputs.empty(), "ACLNN requires tensor arguments");
    const auto device = (inputs.empty() ? outputs.front() : inputs.front()).device();
    TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1, "ACLNN requires NPU tensors");
    c10_npu::NPUGuard guard(device);
    const auto stream = c10_npu::getCurrentNPUStream(device.index());
    // Prepare the stream outside its callback using TorchNPU's exported API.
    // Draining prior host submissions is consistent with this synchronous
    // eager boundary and also handles task-queue level 2 safely.
    const auto raw_stream = stream.stream(true);
    auto state = std::make_shared<CallState>();
    state->tensors.insert(state->tensors.end(), inputs.begin(), inputs.end());
    state->tensors.insert(state->tensors.end(), outputs.begin(), outputs.end());
    for (const auto& tensor : state->tensors) {
        TORCH_CHECK(tensor.device() == device, "ACLNN tensors must share one device");
        TORCH_CHECK(tensor.is_contiguous() && at_npu::native::get_npu_format(tensor) == ACL_FORMAT_ND,
                    "ACLNN adapter requires contiguous ND tensors");
        c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
    }
    for (size_t out = binding->inputs; out < state->tensors.size(); ++out) {
        const auto lo = reinterpret_cast<uintptr_t>(state->tensors[out].data_ptr());
        const auto hi = lo + state->tensors[out].nbytes();
        for (size_t i = 0; i < state->tensors.size(); ++i) {
            if (i == out) continue;
            const auto other = reinterpret_cast<uintptr_t>(state->tensors[i].data_ptr());
            TORCH_CHECK(std::max(lo, other) >= std::min(hi, other + state->tensors[i].nbytes()),
                        "ACLNN output tensors must not overlap other arguments");
        }
    }
    // TorchNPU owns command scheduling. The caller retains this state until
    // RunOpApi(sync=true) has drained the host queue and completed the stream.
    // Do not synchronize the stream from inside its host queue callback.
    auto call = [state, binding, stream, raw_stream]() -> int {
        auto& api = tensor_api();
        for (const auto& tensor : state->tensors) {
            const int64_t storage_size = tensor.storage().nbytes() / tensor.element_size();
            auto* handle = api.create(tensor.sizes().data(), tensor.dim(), acl_dtype(tensor.scalar_type()),
                tensor.strides().data(), tensor.storage_offset(), ACL_FORMAT_ND,
                &storage_size, 1, const_cast<void*>(tensor.storage().data()));
            TORCH_CHECK(handle, "aclCreateTensor failed for ", binding->name);
            state->handles.push_back(handle);
        }
        uint64_t workspace_bytes = 0;
        aclOpExecutor* executor = nullptr;
        auto* workspace_pointer = &workspace_bytes;
        auto** executor_pointer = &executor;
        std::vector<void*> values;
        for (auto& handle : state->handles) values.push_back(&handle);
        values.push_back(&workspace_pointer);
        values.push_back(&executor_pointer);
        ffi_arg status = 0;
        ffi_call(&binding->workspace_signature, FFI_FN(binding->workspace), &status, values.data());
        TORCH_CHECK(status == 0 && executor, binding->name, "GetWorkspaceSize returned ", status);
        binding->last_workspace_bytes.store(workspace_bytes);
        binding->workspace_observed.store(true);
        TORCH_CHECK(workspace_bytes <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                    "ACLNN workspace exceeds tensor allocation range");
        if (workspace_bytes) {
            state->workspace = at::empty({static_cast<int64_t>(workspace_bytes)},
                state->tensors.front().options().dtype(at::kByte));
            c10_npu::NPUCachingAllocator::recordStream(state->workspace.storage().data_ptr(), stream);
        }
        const auto launch_status = binding->launch(
            workspace_bytes ? state->workspace.data_ptr() : nullptr, workspace_bytes, executor, raw_stream);
        TORCH_CHECK(launch_status == 0, binding->name, " returned ", launch_status);
        return 0;
    };
    try {
        at_npu::native::OpCommand::RunOpApi(binding->name, call, true);
        const auto destroy_status = state->release();
        TORCH_CHECK(destroy_status == 0, "aclDestroyTensor returned ", destroy_status);
    } catch (...) {
        // A failed launch may already have submitted work. Keep the original
        // exception while completing best-effort cleanup of retained resources.
        aclrtSynchronizeStream(raw_stream);
        state->release();
        throw;
    }
}
// The original _execute boundary above retains its mandatory-Tensor/ND ABI.
// This separately registered boundary requires explicit immutable conversion,
// format, optional-input and attribute declarations from the verified recipe.
void execute_attributes(const std::string& token,
                      const c10::List<c10::optional<at::Tensor>>& inputs,
                      at::TensorList outputs, const std::vector<c10::IValue>& attributes) {
    std::shared_ptr<Binding> binding;
    {
        std::lock_guard<std::mutex> lock(bindings_mutex);
        const auto found = bindings.find(token);
        TORCH_CHECK(found != bindings.end(), "Unknown extended ACLNN binding");
        binding = found->second;
    }
    TORCH_CHECK(!binding->formats.empty() && inputs.size() == binding->inputs &&
                outputs.size() == binding->outputs && attributes.size() == binding->attributes.size(),
                "Extended ACLNN argument count mismatch");
    std::vector<c10::optional<at::Tensor>> arguments;
    for (const auto& input : inputs) arguments.push_back(input);
    for (const auto& output : outputs) arguments.emplace_back(output);
    auto state = std::make_shared<CallState>();
    for (size_t i = 0; i < arguments.size(); ++i) {
        if (arguments[i].has_value()) state->tensors.push_back(*arguments[i]);
        else TORCH_CHECK(binding->optional[i], "Required ACLNN Tensor is absent");
    }
    TORCH_CHECK(!state->tensors.empty(), "ACLNN requires a Tensor device");
    const auto device = state->tensors.front().device();
    TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1, "ACLNN requires NPU Tensors");
    c10_npu::NPUGuard guard(device);
    const auto stream = c10_npu::getCurrentNPUStream(device.index());
    const auto raw_stream = stream.stream(true);
    for (size_t i = 0; i < arguments.size(); ++i) {
        if (!arguments[i]) continue;
        const auto& tensor = *arguments[i];
        const auto format = at_npu::native::get_npu_format(tensor);
        const int64_t mask = format == ACL_FORMAT_ND ? 1 : format == ACL_FORMAT_FRACTAL_NZ ? 2 : 0;
        TORCH_CHECK(tensor.device() == device && tensor.is_contiguous() &&
                    (binding->formats[i] & mask), "ACLNN Tensor violates its declared device/format");
        c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
    }
    for (size_t out = binding->inputs; out < arguments.size(); ++out) {
        const auto& tensor = *arguments[out];
        const auto lo = reinterpret_cast<uintptr_t>(tensor.data_ptr());
        const auto hi = lo + tensor.nbytes();
        for (size_t i = 0; i < arguments.size(); ++i) {
            if (i == out || !arguments[i]) continue;
            const auto& other = *arguments[i];
            auto begin = reinterpret_cast<uintptr_t>(other.data_ptr());
            auto bytes = other.nbytes();
            if (at_npu::native::get_npu_format(other) == ACL_FORMAT_FRACTAL_NZ) {
                begin = reinterpret_cast<uintptr_t>(other.storage().data());
                bytes = other.storage().nbytes();
            }
            TORCH_CHECK(std::max(lo, begin) >= std::min(hi, begin + bytes),
                        "ACLNN output Tensors must not overlap other arguments");
        }
    }
    // IValues own string storage until the asynchronous queue callback returns.
    std::vector<c10::IValue> scalars(attributes.begin(), attributes.end());
    auto call = [state, binding, arguments, scalars, stream, raw_stream]() -> int {
        auto& api = tensor_api();
        state->handles.reserve(arguments.size());
        for (size_t i = 0; i < arguments.size(); ++i) {
            if (!arguments[i]) { state->handles.push_back(nullptr); continue; }
            const auto& tensor = *arguments[i];
            std::vector<int64_t> sizes(tensor.sizes().begin(), tensor.sizes().end());
            std::vector<int64_t> strides(tensor.strides().begin(), tensor.strides().end());
            auto offset = tensor.storage_offset();
            const auto format = static_cast<aclFormat>(at_npu::native::get_npu_format(tensor));
            std::vector<int64_t> storage = format == ACL_FORMAT_FRACTAL_NZ
                ? at_npu::native::get_npu_storage_sizes(tensor)
                : std::vector<int64_t>{static_cast<int64_t>(tensor.storage().nbytes() / tensor.element_size())};
            auto dtype = acl_dtype(tensor.scalar_type());
            if (binding->conversions[i] == 1 && tensor.scalar_type() == at::kInt) {
                TORCH_CHECK(!sizes.empty() && !storage.empty(), "Packed int4 requires dimensional metadata");
                dtype = ACL_INT4;
                sizes.back() *= 8;
                for (size_t j = 0; j + 1 < strides.size(); ++j) strides[j] *= 8;
                storage.back() *= 8;
                offset *= 8;
                if (format == ACL_FORMAT_FRACTAL_NZ) {
                    TORCH_CHECK(sizes.size() >= 2 && sizes.back() % 64 == 0 &&
                                sizes[sizes.size() - 2] % 16 == 0,
                                "Native INT4 NZ requires N aligned to 64 and K aligned to 16");
                    // The carrier is int32, but the payload uses native INT4
                    // NZ tiles. Its physical C0 is 64 INT4 elements, rather
                    // than the carrier's int32 C0 multiplied by eight.
                    storage.assign(sizes.begin(), sizes.end() - 2);
                    storage.insert(storage.end(), {sizes.back() / 64, sizes[sizes.size() - 2] / 16, 16, 64});
                }
            } else if (binding->conversions[i] == 2 && tensor.scalar_type() == at::kLong) {
                dtype = ACL_UINT64;
            }
            auto* handle = api.create(sizes.data(), sizes.size(), dtype, strides.data(), offset,
                format, storage.data(), storage.size(), const_cast<void*>(tensor.storage().data()));
            TORCH_CHECK(handle, "aclCreateTensor failed for ", binding->name);
            state->handles.push_back(handle);
        }
        struct Attribute { uint8_t boolean; int64_t integer; double floating; const char* string; };
        std::vector<Attribute> values(scalars.size());
        std::vector<void*> ffi_values;
        for (size_t i = 0; i < binding->inputs; ++i) ffi_values.push_back(&state->handles[i]);
        for (size_t i = 0; i < scalars.size(); ++i) {
            if (binding->attributes[i] == 0) {
                TORCH_CHECK(scalars[i].isBool(), "ACLNN expected bool attribute");
                values[i].boolean = scalars[i].toBool(); ffi_values.push_back(&values[i].boolean);
            } else if (binding->attributes[i] == 1) {
                TORCH_CHECK(scalars[i].isInt(), "ACLNN expected int attribute");
                values[i].integer = scalars[i].toInt(); ffi_values.push_back(&values[i].integer);
            } else if (binding->attributes[i] == 2) {
                TORCH_CHECK(scalars[i].isDouble() || scalars[i].isInt(), "ACLNN expected float attribute");
                values[i].floating = scalars[i].isInt() ? scalars[i].toInt() : scalars[i].toDouble();
                ffi_values.push_back(&values[i].floating);
            } else {
                TORCH_CHECK(scalars[i].isString(), "ACLNN expected string attribute");
                const auto& value = scalars[i].toStringRef();
                TORCH_CHECK(value.find('\0') == std::string::npos, "ACLNN string attribute contains NUL");
                values[i].string = value.c_str(); ffi_values.push_back(&values[i].string);
            }
        }
        for (size_t i = binding->inputs; i < arguments.size(); ++i) ffi_values.push_back(&state->handles[i]);
        uint64_t workspace_bytes = 0;
        aclOpExecutor* executor = nullptr;
        auto* size_pointer = &workspace_bytes; auto** executor_pointer = &executor;
        ffi_values.push_back(&size_pointer); ffi_values.push_back(&executor_pointer);
        ffi_arg status = 0;
        ffi_call(&binding->workspace_signature, FFI_FN(binding->workspace), &status, ffi_values.data());
        TORCH_CHECK(status == 0 && executor, binding->name, "GetWorkspaceSize returned ", status);
        binding->last_workspace_bytes.store(workspace_bytes);
        binding->workspace_observed.store(true);
        TORCH_CHECK(workspace_bytes <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                    "ACLNN workspace exceeds Tensor allocation range");
        if (workspace_bytes) {
            state->workspace = at::empty({static_cast<int64_t>(workspace_bytes)},
                state->tensors.front().options().dtype(at::kByte));
            c10_npu::NPUCachingAllocator::recordStream(state->workspace.storage().data_ptr(), stream);
        }
        const auto launched = binding->launch(workspace_bytes ? state->workspace.data_ptr() : nullptr,
                                             workspace_bytes, executor, raw_stream);
        TORCH_CHECK(launched == 0, binding->name, " launch returned ", launched);
        return 0;
    };
    try { at_npu::native::OpCommand::RunOpApi(binding->name, call, true); }
    catch (...) { state->release(); throw; }
    TORCH_CHECK(state->release() == 0, "aclDestroyTensor failed");
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
        const auto found = bindings.find(token);
        if (found != bindings.end()) {
            const auto& prior = *found->second;
            TORCH_CHECK(prior.name == name && prior.workspace == workspace &&
                        prior.launch == reinterpret_cast<Launch>(launch) &&
                        prior.inputs == inputs && prior.outputs == outputs,
                        "Cannot replace an immutable ACLNN binding");
        } else {
            TORCH_CHECK(inputs <= std::numeric_limits<unsigned int>::max() - 2 &&
                        outputs <= std::numeric_limits<unsigned int>::max() - 2 - inputs,
                        "ACLNN binding has too many arguments");
            bindings.emplace(token, std::make_shared<Binding>(name, workspace,
                reinterpret_cast<Launch>(launch), static_cast<size_t>(inputs), static_cast<size_t>(outputs)));
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

extern "C" __attribute__((visibility("default"))) int fi_aclnn_register_extended(
    const char* token, const char* name, void* workspace, void* launch,
    uint64_t inputs, uint64_t outputs, const int64_t* formats,
    const int64_t* conversions, const int64_t* optional,
    const int64_t* attributes, uint64_t attribute_count) noexcept {
    try {
        TORCH_CHECK(token && name && workspace && launch && formats && conversions && optional &&
                    inputs && outputs && inputs + outputs < 1024 && attribute_count < 1024 &&
                    (!attribute_count || attributes), "Invalid extended ACLNN registration");
        auto binding = std::make_shared<Binding>(name, workspace, reinterpret_cast<Launch>(launch), inputs, outputs);
        binding->formats.assign(formats, formats + inputs + outputs);
        binding->conversions.assign(conversions, conversions + inputs + outputs);
        binding->optional.assign(optional, optional + inputs + outputs);
        if (attribute_count) binding->attributes.assign(attributes, attributes + attribute_count);
        binding->argument_types.clear();
        for (size_t i = 0; i < inputs + outputs; ++i) {
            TORCH_CHECK(formats[i] >= 1 && formats[i] <= 3 && conversions[i] >= 0 && conversions[i] <= 2 &&
                        (optional[i] == 0 || optional[i] == 1) && (i < inputs || !optional[i]),
                        "Invalid extended ACLNN Tensor declaration");
        }
        binding->argument_types.assign(inputs, &ffi_type_pointer);
        for (auto type : binding->attributes) {
            TORCH_CHECK(type >= 0 && type <= 3, "Invalid ACLNN attribute type");
            binding->argument_types.push_back(type == 0 ? &ffi_type_uint8 : type == 1 ? &ffi_type_sint64 :
                                             type == 2 ? &ffi_type_double : &ffi_type_pointer);
        }
        binding->argument_types.insert(binding->argument_types.end(), outputs + 2, &ffi_type_pointer);
        TORCH_CHECK(ffi_prep_cif(&binding->workspace_signature, FFI_DEFAULT_ABI,
                                binding->argument_types.size(), &ffi_type_uint32,
                                binding->argument_types.data()) == FFI_OK, "Invalid extended ACLNN signature");
        std::lock_guard<std::mutex> lock(bindings_mutex);
        auto previous = bindings.find(token);
        if (previous != bindings.end()) {
            const auto& prior = *previous->second;
            TORCH_CHECK(prior.name == binding->name && prior.workspace == workspace && prior.launch == binding->launch &&
                        prior.inputs == inputs && prior.outputs == outputs && prior.formats == binding->formats &&
                        prior.conversions == binding->conversions && prior.optional == binding->optional &&
                        prior.attributes == binding->attributes, "Cannot replace an immutable extended ACLNN binding");
        } else bindings.emplace(token, std::move(binding));
        last_error.clear(); return 0;
    } catch (const std::exception& error) { last_error = error.what(); return -1; }
}

// Diagnostics observe the actual native query without rerunning host tiling.
extern "C" __attribute__((visibility("default"))) int fi_aclnn_last_workspace_size(
    const char* token, uint64_t* size) noexcept {
    try {
        TORCH_CHECK(token && size, "Invalid ACLNN workspace observation");
        std::lock_guard<std::mutex> lock(bindings_mutex);
        const auto found = bindings.find(token);
        TORCH_CHECK(found != bindings.end() && found->second->workspace_observed.load(),
                    "ACLNN binding has no observed workspace query");
        *size = found->second->last_workspace_bytes.load();
        last_error.clear();
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}

void execute_extended(const std::string& token,
                      const c10::List<c10::optional<at::Tensor>>& inputs,
                      at::TensorList outputs, at::ArrayRef<at::Scalar> attributes) {
    std::vector<c10::IValue> values;
    for (const auto& attribute : attributes) values.emplace_back(attribute);
    execute_attributes(token, inputs, outputs, values);
}

void execute_with_strings(const std::string& token,
                         const c10::List<c10::optional<at::Tensor>>& inputs,
                         at::TensorList outputs, const c10::List<c10::IValue>& attributes) {
    std::vector<c10::IValue> values;
    for (const auto& attribute : attributes) values.push_back(attribute);
    execute_attributes(token, inputs, outputs, values);
}

TORCH_LIBRARY(flashinfer_aclnn, m) {
    m.def("_execute(str token, Tensor[] inputs, Tensor(a!)[] outputs) -> ()");
    m.def("_execute_extended(str token, Tensor?[] inputs, Tensor(a!)[] outputs, Scalar[] attributes) -> ()");
    m.def("_execute_with_strings(str token, Tensor?[] inputs, Tensor(a!)[] outputs, Any[] attributes) -> ()");
}
TORCH_LIBRARY_IMPL(flashinfer_aclnn, PrivateUse1, m) {
    m.impl("_execute", TORCH_FN(execute));
    m.impl("_execute_extended", TORCH_FN(execute_extended));
    m.impl("_execute_with_strings", TORCH_FN(execute_with_strings));
}
