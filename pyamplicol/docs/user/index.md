<!-- SPDX-License-Identifier: 0BSD -->

# User Guide

pyAmpliCol uses one typed schema for TOML cards, direct CLI options, and Python
services. The primary workflow compiles the packaged external JSON Standard
Model, expands `p p > Z j j` into concrete processes, and evaluates one
schema-v3 artifact from Python, Rust, C++17, or Fortran 2008.

Start with:

1. [Installation](installation.md) for binary-wheel, source, retained-wheel,
   and contributor workflows.
2. [Configuration](configuration.md) for the primary run card, direct flags,
   overrides, color modes, and evaluator choices.
3. [Models And Processes](models.md) for JSON/UFO inputs, multiprocess
   expansion, the built-in compatibility model, and supported UFO features.
4. [Runtime](runtime.md) for total/resolved evaluation, selectors, genuine UFO
   parameter updates, benchmarking, and artifact trust.
5. [Native SDK](native-sdk.md) for generated Rust/C++/Fortran drivers and the
   installed static SDK.
6. [Symbolica Licensing](symbolica.md) for restricted generation and the
   Symbolica-independent direct-JIT f64 runtime path.
7. [Release Status](release-status.md) for the remaining `0.1.0` publication
   gates.

Every packaged card and source example is indexed in
[examples/README.md](../../examples/README.md). `pyamplicol examples copy`
creates an editable workspace and materializes the wheel-owned JSON/UFO models
without relying on a source-tree layout.
