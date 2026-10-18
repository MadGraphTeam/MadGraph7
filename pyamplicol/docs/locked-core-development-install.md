<!-- SPDX-License-Identifier: 0BSD -->
# Development install with an existing Community wheel

A validated full Symbolica Community wheel can be reused while building the
native pyAmpliCol package from its checked-in Cargo manifest and lock. This
explicit candidate lane does not configure Community sources or rebuild its
plugins. The Python distribution version and the Rust kernel version remain
separate: the current inputs use distribution 3.0.2 and kernel 3.0.1.

```sh
PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP=1 \
  python dependencies/install_dependencies.py --candidate \
  --symbolica-wheel /absolute/path/to/symbolica-3.0.2-PLATFORM.whl
```

Use the actual platform wheel filename. Ordinary pip installs that wheel and
normal Python dependencies into the checkout's `.venv`; the existing PEP 517
backend builds the native package and C SDK, generates its non-publishable
candidate version and build information, and installs the resulting wheel.
The explicit bootstrap flag omits stale prepared models and self-test payloads;
this development package can prepare a model from source and generate ordinary
tree artifacts. It is not a release or a regenerated packaged-model distribution.

For direct PEP 517 use, first install the chosen wheel and normal build/runtime
requirements into an isolated environment, then set
`PYAMPLICOL_BUILD_MODE=candidate` and
`PYAMPLICOL_CANDIDATE_SYMBOLICA_WHEEL` to its absolute path. The existing
`pyamplicol.host-wheel=true` configuration supports a local Linux wheel.
`PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP=1` has the same meaning. Cargo's ordinary
cache may be reused; `--locked` controls dependency resolution.

The wheel's Python distribution pin is checked separately from Cargo's native
kernel resolution. Its bytes and location do not invalidate an unchanged native
core. This lane refuses the historical generated candidate lock/config/state
files; use an isolated checkout when those managed-source inputs are present.
Release builds continue to reject this candidate-only wheel option and retain
all existing publication guards.
