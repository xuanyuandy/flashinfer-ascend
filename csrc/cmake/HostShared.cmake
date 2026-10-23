# Both native and CATLASS arrive as sources/definitions/includes. There is no
# third-party discovery or implementation-specific build path here.
set_source_files_properties(${FI_SOURCES} PROPERTIES LANGUAGE ASC)
add_library(fi_ascend_kernel SHARED ${FI_SOURCES})
target_include_directories(fi_ascend_kernel PRIVATE ${FI_INCLUDES})
target_compile_definitions(fi_ascend_kernel PRIVATE ${FI_DEFINITIONS})
target_compile_options(fi_ascend_kernel PRIVATE ${FI_COMPILE_OPTIONS})
target_link_options(fi_ascend_kernel PRIVATE ${FI_LINK_OPTIONS})
get_filename_component(fi_output_dir "${FI_OUTPUT}" DIRECTORY)
get_filename_component(fi_output_name "${FI_OUTPUT}" NAME)
set_target_properties(fi_ascend_kernel PROPERTIES
  PREFIX "" SUFFIX "" OUTPUT_NAME "${fi_output_name}"
  LIBRARY_OUTPUT_DIRECTORY "${fi_output_dir}"
  POSITION_INDEPENDENT_CODE ON)
