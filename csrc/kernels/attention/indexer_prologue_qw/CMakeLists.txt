# Author-owned compile policy for the shared dense/prequantized QW kernel.
function(flashinfer_configure_host target)
  target_compile_options(${target} PRIVATE -Wno-deprecated-declarations)
endfunction()

function(flashinfer_configure_kernel target)
  flashinfer_kernel_compile_options(${target} -O2 -Wno-deprecated-declarations)
endfunction()
