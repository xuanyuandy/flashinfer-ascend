function(flashinfer_configure_kernel target)
  flashinfer_kernel_compile_options(${target}
    --cce-auto-sync=off
    -Wno-deprecated-declarations
    -DENABLE_CV_COMM_VIA_SSBUF=true)
endfunction()
