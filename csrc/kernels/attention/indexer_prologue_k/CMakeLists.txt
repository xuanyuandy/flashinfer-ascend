# Author-owned compile policy for the single mixed-launch K full fusion.
function(flashinfer_configure_host target)
  target_compile_options(${target} PRIVATE -Wno-deprecated-declarations)
endfunction()

function(flashinfer_configure_kernel target)
  flashinfer_kernel_compile_options(${target}
    --cce-auto-sync=off -Wno-deprecated-declarations)
endfunction()
