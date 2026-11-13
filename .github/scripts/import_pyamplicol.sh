#!/usr/bin/env bash
# Import (or re-sync) the history of github.com/MadGraphTeam/pyamplicol into
# the pyamplicol/ subdirectory. The filter is deterministic, so re-running it
# on a newer upstream main reproduces the SHAs of commits imported earlier and
# a merge only brings in the new ones.
#
# Usage: .github/scripts/import_pyamplicol.sh [upstream-ref]
# Leaves the filtered history in the local branch pyamplicol-import; merge it
# with `git merge pyamplicol-import` (first import: --allow-unrelated-histories).
# Requires git-filter-repo (set FILTER_REPO to override the command).
set -euo pipefail

UPSTREAM_URL=${UPSTREAM_URL:-https://github.com/MadGraphTeam/pyamplicol.git}
REF=${1:-main}
FILTER_REPO=${FILTER_REPO:-git-filter-repo}
MG_ROOT=$(git rev-parse --show-toplevel)

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

git clone --quiet --no-local --single-branch --branch "$REF" "$UPSTREAM_URL" "$WORK/pyamplicol"
cd "$WORK/pyamplicol"
UPSTREAM_SHA=$(git rev-parse HEAD)

# Prepared-model packs are rebuilt in CI and shipped as one release archive;
# reports and papers are linked from the archived upstream repository.
# The one-loop reference-generation and benchmark tooling (everything under
# tools/loop_reference and tools/loop_benchmark but their JSON data, which the
# tests read as fixtures) and the one-loop performance reports stay upstream.
# These paths appear only after the first import, so the commits imported
# earlier keep their SHAs.
#
# Restored by hand after the import (MadGraph7 commit "restore the core
# one-loop engine tests"): the filter above drops these helpers, which the
# one-loop engine tests under tests/unit import. They are byte-identical
# copies from upstream; on a re-sync, refresh them from the imported upstream
# commit (git show <upstream-sha>:<path>) and check whether the tests need
# further helpers. The filter itself is left as is, so the imported SHAs stay
# reproducible.
#   tools/loop_reference/exact_points.py
#   tools/loop_reference/feynkit_loop_graphs.py
#   tools/loop_reference/legacy_capture.py
#   tools/loop_reference/levi_civita_aloha.py
#   tools/loop_reference/model_compatibility.py
#   tools/loop_reference/njet/njet_fixture.py
#   tools/loop_reference/ufo_source_oracle.py
$FILTER_REPO --force \
    --path-glob '*.pyamplicol-model' --invert-paths \
    --path-glob '*.pdf' --invert-paths \
    --path-regex '^tools/loop_(reference|benchmark)/.*(?<!\.json)$' --invert-paths \
    --path docs/performance_reports/loop/ --invert-paths \
    --to-subdirectory-filter pyamplicol

git -C "$MG_ROOT" fetch --quiet --no-tags "$WORK/pyamplicol" "+HEAD:refs/heads/pyamplicol-import"
echo "Imported $UPSTREAM_URL $REF at $UPSTREAM_SHA into branch pyamplicol-import"
