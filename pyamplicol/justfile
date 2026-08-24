set shell := ["bash", "-euo", "pipefail", "-c"]

python := env_var_or_default("PYTHON", "python3")
dev_python := env_var_or_default("PYAMPLICOL_DEV_PYTHON", ".venv/bin/python")
build_mode := env_var_or_default("PYAMPLICOL_BUILD_MODE", "release")
eager_builtin_pack := env_var_or_default("PYAMPLICOL_EAGER_BUILTIN_PACK", ".artifacts/eager-models/builtin-sm-jit-o3.pyamplicol-model")
eager_ufo_pack := env_var_or_default("PYAMPLICOL_EAGER_UFO_SM_PACK", ".artifacts/eager-models/ufo-sm-jit-o3.pyamplicol-model")
eager_ufo_source := env_var_or_default("PYAMPLICOL_EAGER_UFO_SM_SOURCE", "src/pyamplicol/assets/models/json/sm/sm.json")
dev_cache := ".artifacts/dev-install"
fft_performance_run_id := env_var_or_default("PYAMPLICOL_FFT_PERFORMANCE_RUN_ID", "manual")

default:
    @just --list

_source-checkout:
    @if [[ ! -e .git || ! -f dependencies/contributor-lock.toml || ! -f dependencies/install_dependencies.py ]]; then \
        printf '%s\n' "error: this command requires a full pyAmpliCol Git source checkout; contributor commands are unavailable from a release source distribution" >&2; \
        exit 1; \
    fi

build:
    PYAMPLICOL_BUILD_MODE={{build_mode}} {{python}} -m build --wheel

typing:
    PYTHONPATH="$PWD/src" {{python}} -m ruff check tools/typing tests/typing
    PYTHONPATH="$PWD/src" {{python}} tools/typing/check_public_typing.py
    PYTHONPATH="$PWD/src" {{python}} -m pytest tests/typing/test_typing_gate.py -q

dependency-gate:
    @if [ "{{build_mode}}" = candidate ]; then \
        {{python}} tools/release/check_dependencies.py --candidate; \
    else \
        {{python}} tools/release/check_dependencies.py; \
    fi

python-unit:
    PYTHONPATH="$PWD/src" {{python}} -m pytest tests/unit -q

python-release:
    PYTHONPATH="$PWD/src" {{python}} -m pytest tests/release -q

python-integration:
    PYTHONPATH="$PWD/src" PYAMPLICOL_REQUIRE_NATIVE_TESTS=1 {{python}} -m pytest tests/integration -q

python-physics:
    PYTHONPATH="$PWD/src" PYAMPLICOL_REQUIRE_NATIVE_TESTS=1 {{python}} -m pytest tests/integration/test_schema_v3_generation_runtime.py tests/unit/test_reference_fixture_v2.py tests/unit/test_tracked_reference_fixture_v2.py tests/unit/test_color_contraction_safety.py -q

# Explicit, fail-fast numerical authority gate. This is intentionally excluded
# from the default suite because it generates fifteen catalog ProcessSets plus
# four separate full-colour extra artifacts.
numerical-acceptance: _source-checkout
    PYTHONPATH="$PWD/src" PYAMPLICOL_RUN_NUMERICAL_ACCEPTANCE=1 PYAMPLICOL_REQUIRE_NATIVE_TESTS=1 {{dev_python}} tools/ci/memory_watchdog.py --limit-gib 30 -- {{dev_python}} -m pytest tests/integration/test_numerical_acceptance.py -q -x

# Dedicated FFT process-table parity and frozen-MadGraph replay.
# This expensive gate is absent from every default CI and release aggregate.
fft-numerical-acceptance: _source-checkout
    mkdir -p .artifacts/fft-acceptance-env/tmp .artifacts/fft-acceptance-env/cargo-home .artifacts/fft-acceptance-env/cargo-target .artifacts/fft-acceptance-env/pip-cache .artifacts/fft-acceptance-env/xdg-cache .artifacts/fft-acceptance-env/python-cache .artifacts/fft-numerical-acceptance
    TMPDIR="$PWD/.artifacts/fft-acceptance-env/tmp" CARGO_HOME="$PWD/.artifacts/fft-acceptance-env/cargo-home" CARGO_TARGET_DIR="$PWD/.artifacts/fft-acceptance-env/cargo-target" CARGO_NET_OFFLINE=true PIP_CACHE_DIR="$PWD/.artifacts/fft-acceptance-env/pip-cache" PIP_NO_INDEX=1 XDG_CACHE_HOME="$PWD/.artifacts/fft-acceptance-env/xdg-cache" PYTHONPYCACHEPREFIX="$PWD/.artifacts/fft-acceptance-env/python-cache" PYTHONPATH="$PWD/src" PYAMPLICOL_REQUIRE_NATIVE_TESTS=1 {{dev_python}} tools/ci/memory_watchdog.py --limit-gib 30 -- {{dev_python}} tools/developer/fft_numerical_acceptance.py --output-root "$PWD/.artifacts/fft-numerical-acceptance"

# Native same-host pure-gluon timing, RSS, and cold-to-ready comparison. This
# hardware-sensitive gate is deliberately excluded from default CI/release runs.
fft-performance-acceptance: _source-checkout
    mkdir -p .artifacts/fft-performance/env/tmp .artifacts/fft-performance/env/cargo-home .artifacts/fft-performance/env/cargo-target .artifacts/fft-performance/env/pip-cache .artifacts/fft-performance/env/xdg-cache .artifacts/fft-performance/env/python-cache
    TMPDIR="$PWD/.artifacts/fft-performance/env/tmp" CARGO_HOME="$PWD/.artifacts/fft-performance/env/cargo-home" CARGO_TARGET_DIR="$PWD/.artifacts/fft-performance/env/cargo-target" CARGO_NET_OFFLINE=true PIP_CACHE_DIR="$PWD/.artifacts/fft-performance/env/pip-cache" PIP_NO_INDEX=1 XDG_CACHE_HOME="$PWD/.artifacts/fft-performance/env/xdg-cache" PYTHONPYCACHEPREFIX="$PWD/.artifacts/fft-performance/env/python-cache" PYTHONPATH="$PWD/src" PYAMPLICOL_REQUIRE_NATIVE_TESTS=1 {{dev_python}} tools/ci/memory_watchdog.py --limit-gib 30 -- {{dev_python}} tools/developer/fft_gluon_performance_acceptance.py --include-optional --run-id "{{fft_performance_run_id}}"

# Build a fresh wheel through the real backend and stage only ignored native
# runtime/SDK resources beside the current Python source for source-tree tests.
source-runtime:
    PYAMPLICOL_BUILD_MODE={{build_mode}} {{python}} tools/developer/prepare_source_runtime.py

# Developer-only independent Fortran oracle. `just dev-install` includes this
# checkout and the pinned Reference FFT checkout by default.
legacy-physics: _source-checkout
    {{python}} tools/developer/legacy_amplicol.py --jobs 5

legacy-physics-verify: _source-checkout
    {{python}} tools/developer/legacy_amplicol.py --fixture tests/fixtures/reference/physics-v2.json --jobs 5 --check-output tests/fixtures/reference/legacy-fortran-v2.json

# Release-facing numerical comparison against the pinned independent Fortran
# implementation. The CI job additionally applies a 30 GiB process limit.
independent-physics-oracle: _source-checkout
    {{python}} tools/developer/legacy_amplicol.py --prepare-checkout --fixture tests/fixtures/reference/physics-v2.json --jobs 2 >/dev/null

installed-smoke:
    PYTHONPATH="$PWD/src" {{python}} -m pyamplicol.selftest
    PYTHONPATH="$PWD/src" {{python}} -m pyamplicol self-test --json
    PYTHONPATH="$PWD/src" {{python}} -m pyamplicol examples list --json
    PYTHONPATH="$PWD/src" PYAMPLICOL_EXAMPLE_CACHE="$PWD/.artifacts/source-gate-example" {{python}} -m pyamplicol examples run builtin_sm_lc --set generation.mode=replace --json

rust-check:
    {{python}} tools/release/run_cargo.py --mode {{build_mode}} -- fmt --all --check
    {{python}} tools/release/run_cargo.py --mode {{build_mode}} -- clippy --workspace --all-targets --locked

rust-test:
    {{python}} tools/release/run_cargo.py --mode {{build_mode}} -- test --workspace --locked
    {{python}} tools/release/run_cargo.py --mode {{build_mode}} -- test --locked -p rusticol-core --no-default-features --features f64-compiled native_role_exports_load_and_remain_callable_while_owners_move

# Complete source gate used before any release artifact is retained.
source-gate:
    just dependency-gate
    just source-runtime
    just typing
    just python-unit
    just python-release
    just python-integration
    just python-physics
    just rust-check
    just rust-test
    just installed-smoke

check:
    just typing
    just dependency-gate
    just python-unit
    just rust-check

test:
    PYTHONPATH="$PWD/src" {{python}} -m pytest
    just rust-test

sdist:
    PYAMPLICOL_BUILD_MODE={{build_mode}} {{python}} -m build --sdist

wheel:
    PYAMPLICOL_BUILD_MODE={{build_mode}} {{python}} -m build --wheel

wheel-from-sdist:
    {{python}} tools/release/build_from_sdist.py

install-wheel PYTHON_ARG="":
    @selected="{{PYTHON_ARG}}"; \
    if [[ "$selected" == PYTHON=* ]]; then \
        selected="$(printf '%s' "$selected" | cut -d= -f2-)"; \
    fi; \
    if [[ -z "$selected" ]]; then selected="{{python}}"; fi; \
    {{python}} tools/release/install_wheel.py --python "$selected"

dev-install *INSTALLER_ARGS: _source-checkout
    mkdir -p {{dev_cache}}/tmp {{dev_cache}}/cargo-home {{dev_cache}}/cargo-target {{dev_cache}}/pip-cache {{dev_cache}}/xdg-cache {{dev_cache}}/python-cache
    TMPDIR="$PWD/{{dev_cache}}/tmp" CARGO_HOME="$PWD/{{dev_cache}}/cargo-home" CARGO_TARGET_DIR="$PWD/{{dev_cache}}/cargo-target" PIP_CACHE_DIR="$PWD/{{dev_cache}}/pip-cache" XDG_CACHE_HOME="$PWD/{{dev_cache}}/xdg-cache" PYTHONPYCACHEPREFIX="$PWD/{{dev_cache}}/python-cache" PYAMPLICOL_CANDIDATE_CACHE_ROOT="$PWD/{{dev_cache}}" {{python}} dependencies/install_dependencies.py {{INSTALLER_ARGS}}
    TMPDIR="$PWD/{{dev_cache}}/tmp" CARGO_HOME="$PWD/{{dev_cache}}/cargo-home" CARGO_TARGET_DIR="$PWD/{{dev_cache}}/cargo-target" PIP_CACHE_DIR="$PWD/{{dev_cache}}/pip-cache" XDG_CACHE_HOME="$PWD/{{dev_cache}}/xdg-cache" PYTHONPYCACHEPREFIX="$PWD/{{dev_cache}}/python-cache" PYAMPLICOL_BUILD_MODE=candidate {{python}} tools/developer/prepare_source_runtime.py --candidate --wheel-directory .artifacts/candidate

# Report/campaign prerequisite. pyAmpliCol is not released yet, so this keeps
# the explicit build entrypoint tied to the pinned dev-install environment.
dev-build: _source-checkout
    just dev-install
    {{dev_python}} -c 'import pyamplicol; import pyamplicol.api'
    {{dev_python}} src/pyamplicol/_profiling_campaign/result_tables.py validate

dev-test: _source-checkout
    PYTHON={{dev_python}} PYAMPLICOL_BUILD_MODE=candidate just source-gate
    PYTHON={{dev_python}} PYAMPLICOL_BUILD_MODE=candidate just test-deployment-candidate

# Bounded built-in-SM eager gate under the 30 GiB memory guard.
eager-smoke: _source-checkout
    {{dev_python}} tools/ci/memory_watchdog.py --limit-gib 30 -- {{dev_python}} tools/developer/eager_benchmark_matrix.py --suite smoke --models built-in --builtin-pack {{eager_builtin_pack}} --output-root .artifacts/eager-benchmark/smoke

# Full built-in/UFO-SM eager milestone gate under the 30 GiB memory guard.
eager-milestone: _source-checkout
    {{dev_python}} tools/ci/memory_watchdog.py --limit-gib 30 -- {{dev_python}} tools/developer/eager_benchmark_matrix.py --suite milestone --models built-in,ufo-sm --builtin-pack {{eager_builtin_pack}} --ufo-source {{eager_ufo_source}} --ufo-pack {{eager_ufo_pack}} --output-root .artifacts/eager-benchmark/milestone

test-deployment-candidate: _source-checkout
    PYAMPLICOL_BUILD_MODE=candidate {{python}} tools/release/test_deployment.py --candidate

test-deployment:
    PYAMPLICOL_BUILD_MODE=release {{python}} tools/release/test_deployment.py

release-artifacts: _source-checkout
    PYAMPLICOL_BUILD_MODE=release just source-gate
    just independent-physics-oracle
    PYAMPLICOL_BUILD_MODE=release {{python}} tools/release/build_release_artifacts.py

publish-dry-run: _source-checkout
    PYAMPLICOL_BUILD_MODE=release just source-gate
    just independent-physics-oracle
    PYAMPLICOL_BUILD_MODE=release {{python}} tools/release/publish_dry_run.py
