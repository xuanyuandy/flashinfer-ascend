function(flashinfer_configure_host target)
endfunction()

function(flashinfer_configure_kernel target)
  # The imported kernel redefines SET_FLAG/WAIT_FLAG locally.
  flashinfer_kernel_compile_options(${target} -Wno-macro-redefined)
endfunction()
