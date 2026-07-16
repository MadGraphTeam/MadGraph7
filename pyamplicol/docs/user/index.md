<!-- SPDX-License-Identifier: 0BSD -->

# User Guide

pyAmpliCol has three steering surfaces backed by the same immutable schema:
TOML run cards, direct CLI options, and typed Python services. Generated
schema-v3 artifacts are evaluated by Rusticol through Python or the packaged
C++17/Fortran 2008 SDK.

Start with:

1. [Installation](installation.md) for PyPI, source, retained-wheel, and
   contributor workflows.
2. [Configuration](configuration.md) for run cards, direct flags, overrides,
   color modes, and evaluator choices.
3. [Models And Processes](models.md) for the built-in SM, external UFO/JSON,
   process sets, and multiparticles.
4. [Runtime](runtime.md) for total/resolved evaluation, selectors, model
   parameters, and benchmarking.
5. [Native SDK](native-sdk.md) for C++17 and Fortran 2008 consumers.
6. [Symbolica Licensing](symbolica.md) for restricted mode, license requests,
   and generation resource policy.
7. [Release Status](release-status.md) for the remaining integration and
   publication gates in the current milestone checkout.

Every packaged example is indexed in
[examples/README.md](../../examples/README.md). Built-in-model cards can be run
directly; the external-model examples include a helper that copies wheel-owned
UFO/JSON assets into an editable example workspace.
