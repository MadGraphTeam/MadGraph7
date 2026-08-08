<!-- SPDX-License-Identifier: 0BSD -->

# User Guide

pyAmpliCol uses one typed schema for TOML cards, direct CLI options, and Python
services. The primary workflow compiles the packaged external JSON Standard
Model, expands `p p > Z j j` into concrete processes, and evaluates one
schema-v3 artifact from Python, C11, C++17, Fortran 2008, or Rust 2021.

Start with:

1. [Get started: a gentle walkthrough](gentle-walkthrough.md) for a
   MadGraph-familiar, copy-paste tour of generation, process outputs, inspection,
   evaluation, profiling, Python, and the native APIs.
2. [LC workloads and execution modes](lc-workloads-and-execution-modes.md) for
   choosing topology replay versus all-flow union, comparing recurrence,
   compiled, eager, and OTF, and planning an explicit OTF warm-up.
3. [Profiling campaign: from measurements to the PDF](profiling-campaign-walkthrough.md)
   for launching, stopping and resuming a validated scan, retaining its
   evidence, publishing the report, and reading the “Best vs AmpliCol” matrix.
4. [Installation](installation.md) for binary-wheel, source, retained-wheel,
   and contributor workflows.
5. [Configuration](configuration.md) for the primary run card, direct flags,
   overrides, color modes, and evaluator choices.
6. [Models And Processes](models.md) for JSON/UFO inputs, multiprocess
   expansion, the built-in compatibility model, and supported UFO features.
7. [Runtime](runtime.md) for total/resolved evaluation, selectors, genuine UFO
   parameter updates, benchmarking, and artifact trust.
8. [Native SDK](native-sdk.md) for generated C11, C++17, Fortran 2008, and Rust
   2021 drivers and the installed static SDK.
9. [Symbolica Licensing](symbolica.md) for restricted generation and the
   Symbolica-independent direct-JIT f64 runtime path.
10. [Release Status](release-status.md) for the tagged and validated `0.1.1`
   artifacts, TestPyPI availability, and the pending PyPI upload.

Every packaged card and source example is indexed in
[examples/README.md](../../examples/README.md). `pyamplicol examples copy`
creates an editable workspace and materializes the wheel-owned JSON/UFO models
without relying on a source-tree layout.

`pyamplicol profiling-campaign copy DEST --force` similarly creates a fresh
installed-resource profiling workspace. Its visible `campaign_artifacts/`
directory holds all campaign state and moves with `DEST`; old repository-level
`.artifacts` state is ignored. `--force` resets only that local state and the
managed PDF, summary IDs, measurement lineage, and known LaTeX byproducts while
preserving unrelated files. The copy can run headlessly without optional
Ratatui bindings; only original-AmpliCol
measurements require the copied launcher's
`run --original-amplicol PATH_TO_CLEAN_COMPLETE_CHECKOUT` option.
