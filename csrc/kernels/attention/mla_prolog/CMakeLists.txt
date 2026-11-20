# Author-owned options for the DS4.1 MLA prolog op package.
# These flags mirror the validated standalone CATLASS build.
function(flashinfer_configure_host target)
  target_compile_options(${target} PRIVATE -Wno-deprecated-declarations)
endfunction()

function(flashinfer_configure_kernel target)
  flashinfer_kernel_compile_options(${target}
    -Wno-macro-redefined
    -Wno-ignored-attributes
    -Wno-unused-value
    -mllvm -cce-aicore-record-overflow=true
    -mllvm -cce-aicore-dcci-insert-for-scalar=false)
endfunction()
