# Standard CANN packaging API. No operator implementations or installer copied
# from a downstream product, and no install target is executed by our driver.
set(package_name ${FI_PACKAGE_VENDOR})
npu_op_package(${package_name} TYPE RUN CONFIG INSTALL_PATH ${CMAKE_BINARY_DIR})
add_subdirectory(cmake/op_package/host)
add_subdirectory(cmake/op_package/kernel)
