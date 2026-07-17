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

## Handoff and regression evidence

- Patch 0001 affects ordinary complex C++ export on every target and scalar
  complex inline assembly on AArch64. The original exporter dropped the
  imaginary part while converting constants, emitted malformed wrapped
  complex constants, described vector-register clobbers as scalar registers,
  and let several real-only operations read or retain an undefined imaginary
  lane. Those defects caused compilation failures or incorrect complex f64
  values in ASM/C++ process artifacts. The guarded macOS arm64 candidate job
  generates JIT, ASM, and C++ artifacts for `d d~ > z`, requires their totals
  to agree at `rtol=1e-12`, and then exercises the installed native runtime.
- Patch 0002 is a scheduling fix rather than a numerical change. Without it,
  `evaluator_multiple()` retains the GIL for the complete CPU-bound evaluator
  build, serializing otherwise independent Python generation workers. The
  candidate source gate covers multi-evaluator process generation; the patch
  does not alter evaluator expressions or serialized payloads.
- Patch 0003 carries its own Rust regression over optimization levels 0, 1, 2,
  and 3. pyAmpliCol additionally records the requested level in each evaluator
  manifest, defaults production generation to O3, and checks that exported
  evaluator metadata preserves that setting.
- Patch 0004 carries a Rust regression for self-contained application export.
  pyAmpliCol's installed-deployment gate blocks every `symbolica` import,
  removes `SYMBOLICA_LICENSE`, loads the portable direct-SymJIT self-test
  artifact through Rusticol, and requires the f64 physics check to pass.

The authoritative patch digests and exact Symbolica/Symbolica-community/SymJIT
revisions are recorded in `dependencies/contributor-lock.toml`. Release mode
must use an upstream published implementation and never consumes this patch
directory.
