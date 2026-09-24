# Madgraph 4 GPU

This repository contains code developed in the context of porting the [MadGraph7](https://cp3.irmp.ucl.ac.be/projects/madgraph/) event generator software onto GPU platforms and vector instructions on CPUs. MadGraph7 is able to generate code for various physics processes in different programming languages (Fortran, C, C++). The code generated in this repository in "epochX" of the MadGraph7 generator allows to also produce source code for those physics processes to run on GPU and CPU platforms. 

## NLO real-amplitude offloading

The built-in MadMatrix exporter can be enabled for the real-emission part of an
FKS NLO output:

```text
generate p p > t t~ [QCD]
output /path/to/output -f --me_exporter=mg7 --vector_size=5
```

The option leaves the Born, virtual, subtraction, integration, and event-writing
workflow in Fortran. It adds one MadMatrix shared library per distinct real matrix
element, an exception-safe dynamic bridge, a versioned row/flavour/squared-order
manifest, and retained Fortran real routines. Generated MINT jobs use the requested
`vector_size`; inactive lanes are compacted before one batch call per physical FKS
row and restored to their original lanes afterwards.

The runtime defaults to the retained Fortran implementation, so generated output
does not require a MadMatrix process library at runtime unless offloading is
explicitly selected. Set `MG7_NLO_REAL_BACKEND` to one of `scalar`, `simd_128`,
`simd_256`, `simd_512`, `avx512y`, `cuda`, or `hip`. Values `fortran`, `off`, and
`none` select Fortran explicitly. For example:

```sh
make -C SubProcesses/P0_QQx_ttx madevent_mintMC \
  NLO_REAL_BACKEND=avx512y
MG7_NLO_REAL_BACKEND=avx512y SubProcesses/P0_QQx_ttx/madevent_mintMC
```

`MG7_NLO_REAL_PARAM_CARD` and `MG7_NLO_REAL_LIBRARY_DIR` override the default
locations. Otherwise they are resolved relative to `libnlo_real_bridge.so`, which
keeps a moved generated tree usable. `MG7_NLO_REAL_TRACE=1` prints the selected
backend and each physical-row batch size. The ordinary subprocess `make clean`
target removes the bridge, stamps, objects, and process libraries for every built
backend.

CUDA and HIP builds include only real matrix elements that have one squared-order
component. Unsupported multi-order real elements are identified by matrix-element
ID and FKS rows at startup and use their retained Fortran routine; they are never
sent through an incomplete GPU split-order path. CPU backends support all generated
split-order components. GPU evaluation is synchronous at the bridge boundary: the
call returns only after output has been copied back to host memory. Production
OpenMP parallel amplitude calls remain disabled.
