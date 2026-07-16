# Symbolica patch series

This development-only series targets Symbolica revision
`e4167e767147ab8f3b4f039057c396c8fa961f6a`. Apply the numbered patches in
order.

The series deliberately uses Symbolica's ordinary
`JITCompiledEvaluator<Complex<f64>>`. SymJIT enables SIMD in its default
configuration, prepares a SIMD application during compilation, and dispatches
batched `evaluate_matrix()` calls to that application. No explicit
`Complex<wide::f64x2>` evaluator, manual lane packing, custom profiling API, or
extended evaluator serialization is needed.

## Patches

1. `0001-fix-complex-export-aarch64.patch` fixes scalar complex constants in
   generated C++ and correctness issues in the scalar complex AArch64 assembly
   exporter. It intentionally excludes changes to real-valued and explicit
   `Complex<f64x4>` exporters, which pyAmpliCol does not use.
2. `0002-release-gil-for-multiple-evaluator-build.patch` releases the Python GIL
   while `evaluator_multiple()` performs its CPU-bound construction. This lets
   independent evaluator chunks and processes build concurrently from Python
   threads.
3. `0003-forward-symjit-optimization-level.patch` forwards the caller's requested
   SymJIT optimization level instead of silently compiling every evaluator at
   O2. It includes a regression test for levels 0 through 3.
4. `0004-export-symjit-application.patch` exposes the already serialized,
   ordinary `complex-f64` SymJIT application after JIT materialization. The
   export rejects external functions, allowing Rusticol to load the trusted
   application with SymJIT's own `Application::load()` path and evaluate f64
   artifacts without importing Symbolica.

These patches are candidates for upstreaming. They are not included in a
release wheel or sdist.
