# Standard CANN packaging API. No operator implementations or installer copied
# from a downstream product, and no install target is executed by our driver.
set(package_name ${FI_PACKAGE_VENDOR})
function(flashinfer_kernel_compile_options target)
  # CANN kernel libraries are logical handles until deferred target creation.
  set_property(GLOBAL APPEND PROPERTY FI_OPERATOR_KERNEL_OPTIONS_${target} ${ARGN})
endfunction()
npu_op_package(${package_name} TYPE RUN CONFIG INSTALL_PATH ${CMAKE_BINARY_DIR})
add_subdirectory(host)
add_subdirectory(kernel)
