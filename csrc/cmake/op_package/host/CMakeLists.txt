# Definitions used by host registration are also bound into BuildRecipe.
add_compile_definitions(${FI_DEFINITIONS})
set(host_options)
foreach(definition IN LISTS FI_DEFINITIONS)
  list(APPEND host_options "-D${definition}")
endforeach()
foreach(directory IN LISTS FI_INCLUDES)
  list(APPEND host_options "-I${directory}")
endforeach()
npu_op_code_gen(SRC ${FI_HOST_SOURCES} PACKAGE ${package_name}
  OUT_DIR ${ASCEND_AUTOGEN_PATH} JOIN_OP_DEF True
  COMPILE_OPTIONS ${host_options} LINK_OPTIONS ${FI_LINK_OPTIONS})
file(GLOB api_sources ${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp)
file(GLOB proto_sources ${ASCEND_AUTOGEN_PATH}/group_op_proto/*.cc ${ASCEND_AUTOGEN_PATH}/op_proto.cc)
file(GLOB fallback_sources ${ASCEND_AUTOGEN_PATH}/fallback_*.cpp)
set_source_files_properties(${api_sources} ${proto_sources} ${fallback_sources} PROPERTIES GENERATED TRUE)
if(FI_ACLNN_GENERATOR)
  set(typed_adapter "${CMAKE_BINARY_DIR}/fi_aclnn_adapter.cpp")
  execute_process(COMMAND "${ASCEND_PYTHON_EXECUTABLE}" -B "${FI_ACLNN_GENERATOR}"
    --recipe "${FI_ACLNN_RECIPE_FILE}" --headers "${ASCEND_AUTOGEN_PATH}"
    --ops-info "${ASCEND_AUTOGEN_PATH}/aic-${ASCEND_COMPUTE_UNIT}-ops-info.ini"
    --output "${typed_adapter}" RESULT_VARIABLE adapter_status
    OUTPUT_VARIABLE adapter_stdout ERROR_VARIABLE adapter_stderr)
  if(NOT adapter_status EQUAL 0)
    message(FATAL_ERROR "ACLNN NativeBinding/OpDef mismatch: ${adapter_stdout}${adapter_stderr}")
  endif()
  list(APPEND api_sources "${typed_adapter}")
endif()
npu_op_library(cust_opapi ACLNN ${api_sources})
npu_op_library(cust_op_proto GRAPH ${FI_HOST_SOURCES} ${proto_sources})
npu_op_library(cust_optiling TILING ${FI_HOST_SOURCES} ${fallback_sources})
foreach(target cust_opapi cust_op_proto cust_optiling)
  target_include_directories(${target} PRIVATE ${FI_INCLUDES} ${ASCEND_AUTOGEN_PATH})
  target_compile_options(${target} PRIVATE -fvisibility=hidden)
  target_link_options(${target} PRIVATE ${FI_LINK_OPTIONS})
endforeach()
npu_op_package_add(${package_name} LIBRARY cust_opapi cust_op_proto cust_optiling)
