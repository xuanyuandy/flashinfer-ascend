# Author-owned options; this file and local .cmake includes are hashed by the framework.
function(flashinfer_configure_host target)
  target_compile_options(${target} PRIVATE -Wno-deprecated-declarations)
endfunction()

function(flashinfer_configure_kernel target)
  flashinfer_kernel_compile_options(${target} -O2 --cce-auto-sync=on -Wno-deprecated-declarations)
endfunction()
