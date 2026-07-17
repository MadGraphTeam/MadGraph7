set shell := ["bash", "-euo", "pipefail", "-c"]

python := env_var_or_default("PYTHON", "python3")
dev_python := env_var_or_default("PYAMPLICOL_DEV_PYTHON", ".venv/bin/python")
build_mode := env_var_or_default("PYAMPLICOL_BUILD_MODE", "release")

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
        {{python}} tools/release/check_dependencies.py --candidate --offline; \
    else \
        {{python}} tools/release/check_dependencies.py --offline; \
    fi

python-unit:
    PYTHONPATH="$PWD/src" {{python}} -m pytest tests/unit -q

python-release:
    PYTHONPATH="$PWD/src" {{python}} -m pytest tests/release -q

python-integration:
    PYTHONPATH="$PWD/src" PYAMPLICOL_REQUIRE_NATIVE_TESTS=1 {{python}} -m pytest tests/integration -q

python-physics:
    PYTHONPATH="$PWD/src" PYAMPLICOL_REQUIRE_NATIVE_TESTS=1 {{python}} -m pytest tests/integration/test_schema_v3_generation_runtime.py tests/unit/test_reference_fixture_v2.py tests/unit/test_tracked_reference_fixture_v2.py tests/unit/test_color_contraction_safety.py -q

# Build a fresh wheel through the real backend and stage only ignored native
# runtime/SDK resources beside the current Python source for source-tree tests.
source-runtime:
    PYAMPLICOL_BUILD_MODE={{build_mode}} {{python}} tools/developer/prepare_source_runtime.py

# Developer-only independent Fortran oracle. `just dev-install` includes the
# pinned checkout unless it was called with --without-legacy-amplicol.
legacy-physics: _source-checkout
    {{python}} tools/developer/legacy_amplicol.py --jobs 5

legacy-physics-verify: _source-checkout
    {{python}} tools/developer/legacy_amplicol.py --fixture tests/fixtures/reference/physics-v2.json --jobs 5 --check-output tests/fixtures/reference/legacy-fortran-v2.json

# Release-facing replay of the pinned independent physics evidence. The CI job
# additionally applies a 30 GiB process limit; local milestone runs invoke this
# recipe through the repository-external memory watchdog.
independent-physics-oracle: _source-checkout
    {{python}} tools/developer/legacy_amplicol.py --prepare-checkout --fixture tests/fixtures/reference/physics-v2.json --jobs 2 --check-output tests/fixtures/reference/legacy-fortran-v2.json

installed-smoke:
    PYTHONPATH="$PWD/src" {{python}} -m pyamplicol.selftest
    PYTHONPATH="$PWD/src" {{python}} -m pyamplicol self-test --format json
    PYTHONPATH="$PWD/src" {{python}} -m pyamplicol examples list --format json
    PYTHONPATH="$PWD/src" PYAMPLICOL_EXAMPLE_CACHE="$PWD/.artifacts/source-gate-example" {{python}} -m pyamplicol examples run builtin_sm_lc --set generation.mode=replace --format json

rust-check:
    {{python}} tools/release/run_cargo.py --mode {{build_mode}} -- fmt --all --check
    {{python}} tools/release/run_cargo.py --mode {{build_mode}} -- clippy --workspace --all-targets --locked -- -D warnings

rust-test:
    {{python}} tools/release/run_cargo.py --mode {{build_mode}} -- test --workspace --locked

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

dev-install: _source-checkout
    {{python}} dependencies/install_dependencies.py

# Report/campaign prerequisite. pyAmpliCol is not released yet, so this keeps
# the explicit build entrypoint tied to the patched dev-install environment.
dev-build: _source-checkout
    just dev-install
    {{dev_python}} -c 'import pyamplicol; import pyamplicol.api'
    {{dev_python}} docs/result_tables.py validate

dev-test: _source-checkout
    PYTHON={{dev_python}} PYAMPLICOL_BUILD_MODE=candidate just source-gate
    PYTHON={{dev_python}} PYAMPLICOL_BUILD_MODE=candidate just test-deployment-candidate

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
