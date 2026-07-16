set shell := ["bash", "-euo", "pipefail", "-c"]

python := env_var_or_default("PYTHON", "python3")
build_mode := env_var_or_default("PYAMPLICOL_BUILD_MODE", "release")

default:
    @just --list

build:
    PYAMPLICOL_BUILD_MODE={{build_mode}} {{python}} -m build --wheel

typing:
    {{python}} -m ruff check tools/typing tests/typing
    {{python}} tools/typing/check_public_typing.py
    {{python}} -m pytest tests/typing/test_typing_gate.py -q

dependency-gate:
    @if [ "{{build_mode}}" = candidate ]; then \
        {{python}} tools/release/check_dependencies.py --candidate --offline; \
    else \
        {{python}} tools/release/check_dependencies.py --offline; \
    fi

legal-gate:
    {{python}} tools/release/check_legal_inventory.py --mode {{build_mode}}

python-unit:
    {{python}} -m pytest tests/unit -q

python-release:
    {{python}} -m pytest tests/release -q

python-integration:
    PYAMPLICOL_REQUIRE_NATIVE_TESTS=1 {{python}} -m pytest tests/integration -q

python-physics:
    {{python}} -m pytest tests/integration/test_schema_v3_generation_runtime.py tests/unit/test_reference_fixtures.py tests/unit/test_color_contraction_safety.py -q

installed-smoke:
    {{python}} -m pyamplicol.selftest
    {{python}} -m pyamplicol self-test --format json
    {{python}} -m pyamplicol examples list --format json
    PYAMPLICOL_EXAMPLE_CACHE="$PWD/.artifacts/source-gate-example" {{python}} -m pyamplicol examples run builtin_sm_lc --set generation.mode=replace --format json

rust-check:
    {{python}} tools/release/run_cargo.py --mode {{build_mode}} -- fmt --all --check
    {{python}} tools/release/run_cargo.py --mode {{build_mode}} -- clippy --workspace --all-targets --locked -- -D warnings

rust-test:
    {{python}} tools/release/run_cargo.py --mode {{build_mode}} -- test --workspace --locked

# Complete source gate used before any release artifact is retained.
source-gate:
    just legal-gate
    just dependency-gate
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
    {{python}} -m pytest
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

dev-install:
    {{python}} dependencies/install_dependencies.py

dev-test:
    PYAMPLICOL_BUILD_MODE=candidate just source-gate
    PYAMPLICOL_BUILD_MODE=candidate just test-deployment-candidate

test-deployment-candidate:
    PYAMPLICOL_BUILD_MODE=candidate {{python}} tools/release/test_deployment.py --candidate

test-deployment:
    PYAMPLICOL_BUILD_MODE=release {{python}} tools/release/test_deployment.py

release-artifacts:
    PYAMPLICOL_BUILD_MODE=release just source-gate
    PYAMPLICOL_BUILD_MODE=release {{python}} tools/release/check_legal_inventory.py --mode release
    PYAMPLICOL_BUILD_MODE=release {{python}} tools/release/build_release_artifacts.py

publish-dry-run:
    PYAMPLICOL_BUILD_MODE=release just source-gate
    PYAMPLICOL_BUILD_MODE=release {{python}} tools/release/check_legal_inventory.py --mode release
    PYAMPLICOL_BUILD_MODE=release {{python}} tools/release/publish_dry_run.py
