<!-- SPDX-License-Identifier: 0BSD -->
# `macbook_M3` table-filling runbook

## Required result and fixed policy

Produce `docs/performance_reports/macbook_M3/pyAmpliCol.pdf` from a fresh
architecture profile at one exact measured-source commit. Cover the complete
declared table ranges, from low to high final-state multiplicity; there is no
hard \(n=4\) campaign ceiling.

The bound `macbook-m3-v1` policy is not negotiable:

- one worker, using one core;
- a five-second timing target and at least five completed samples;
- a decimal 30 GB process-tree RSS ceiling for every cell, model, colour
  treatment, layout, generation step, validation step, and profiling step;
- an authenticated `>30GB` table entry when that ceiling is crossed; and
- no higher multiplicity in the same process lane after a lower multiplicity
  reaches the RAM terminal. The higher entries receive an authenticated
  frontier marker; they are not silently left as `N/A`.

There is no generation-time cutoff on the Mac. A timeout is a defect, not a
publishable resource result and not a reason to derive a frontier.

A process lane means the same process family, execution mode, model source,
colour accuracy, LC workload, backend/optimization, and Z variant. A terminal
in one lane must not suppress unrelated lanes.

Every successful timing must also pass the numerical agreement graph.
Recurrence is compared directly with AmpliCol at `rtol=1e-8`, `atol=1e-15`;
pyAmpliCol cross-mode, model-route, and layout checks use `rtol=1e-12`,
`atol=1e-15`. Scalar/exact lanes also compare precision 16 with precision 32.

The committed profile is a reset scaffold. Never reuse another source epoch's
measurements, evaluator artifacts, prepared models, or coordination state.
`--artifact-policy reuse` means reuse only within this campaign.

## Support lane for both machines

Before measuring, reserve a dedicated local subagent named
`x86_epyc_support_lane` in a separate worktree and artifact root. Give its task
address to the cluster filler and obtain an acknowledgement. It receives
cell-scoped reproductions, accumulates tested fixes, and coordinates batched
landings with the main local filler. It must never edit either active campaign
checkout or relabel existing evidence.

The Mac and cluster campaigns use independent measurement branches, worktrees,
profiles, artifact roots, coordination roots, virtual environments, and
candidate-wheel directories. Use these fixed branch roles:

- `codex/macbook-M3-full-report`: frozen-source Mac measurement checkout;
- `codex/x86-EPYC-full-report`: frozen-source cluster measurement checkout;
- `codex/x86-EPYC-support`: local fixes and reproductions only; and
- `codex/x86-EPYC-report-checkpoints`: report-only cluster snapshots for
  review.

Never merge or pull either cluster branch into the active Mac measurement
worktree. Executable fixes land on `main` in coordinated batches and begin a
new final measurement epoch only when necessary.

The Mac filler is the initial owner of the serialized main-push token. It
releases that token to the x86 filler immediately after its aggregate landing,
or may explicitly transfer it earlier if the x86 publication finishes first.
There is never an implicit or mutually awaited owner.

### Hourly cluster PDF review

The support lane is also the cluster publication reviewer. Once the cluster
creates its checkpoint branch, give the support lane a separate, read-only
review worktree:

```bash
set -euo pipefail
git fetch origin codex/x86-EPYC-report-checkpoints
REVIEW_TREE="../pyamplicol-x86-EPYC-report-review"
test ! -e "$REVIEW_TREE"
git worktree add -b codex/x86-EPYC-report-review \
  "$REVIEW_TREE" origin/codex/x86-EPYC-report-checkpoints
git -C "$REVIEW_TREE" branch --set-upstream-to=\
origin/codex/x86-EPYC-report-checkpoints
```

At least once per hour while the cluster campaign is active, the support lane
must obtain the newest lightweight checkpoint with the following recipe. Set a
recurring hourly wake-up in the support task; do not rely on the Mac
measurement turn or the cluster `populate` command returning.

```bash
set -euo pipefail
git -C "$REVIEW_TREE" pull --ff-only
PDF="$REVIEW_TREE/docs/performance_reports/x86_EPYC/pyAmpliCol.pdf"
REVIEW_SHA="$(git -C "$REVIEW_TREE" rev-parse --short=12 HEAD)"
REVIEW_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
QA_DIR=".artifacts/performance-report-qa/x86_EPYC/hourly-$REVIEW_STAMP-$REVIEW_SHA"
test ! -e "$QA_DIR"
mkdir -p "$QA_DIR"
pdfinfo "$PDF"
pdftoppm -png -r 144 "$PDF" "$QA_DIR/page"
```

Inspect every page, not only changed pages. Check process coverage, status
labels, units, ratios, A/B/C winners, summaries, legends, continuation pages,
spacing, clipping, blank regions, and whether values are physically and
numerically plausible. Compare raw status counts with what the PDF shows.
Send a timestamped review result and concrete page/cell feedback directly to
the cluster filler. If a Mac measurement worker is active, perform the pull and
freshness check on time but defer CPU-heavy rasterization to the first cell
boundary; report that delay explicitly and do not launch the next Mac cell
until the review completes. Do not commit or push from the review worktree.

## 1. Establish the exact measured source

From a clean coordinator checkout, create distinct measurement and support
worktrees. Do not switch branches in an existing campaign checkout:

```bash
set -euo pipefail
git fetch origin
MEASURE_TREE="../pyamplicol-macbook-M3-measure"
SUPPORT_TREE="../pyamplicol-x86-EPYC-support"
test ! -e "$MEASURE_TREE"
test ! -e "$SUPPORT_TREE"
git worktree add -b codex/macbook-M3-full-report \
  "$MEASURE_TREE" origin/main
git worktree add -b codex/x86-EPYC-support \
  "$SUPPORT_TREE" origin/main
git -C "$MEASURE_TREE" push -u origin codex/macbook-M3-full-report
git -C "$SUPPORT_TREE" push -u origin codex/x86-EPYC-support
cd "$MEASURE_TREE"
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
test -z "$(git status --short)"
test "$(uname -s)" = "Darwin"
test "$(uname -m)" = "arm64"
system_profiler SPHardwareDataType | grep -E 'Chip: Apple M3'
test -f docs/performance_reports/macbook_M3/TABLE_FILLING.md
test ! -e .artifacts/performance-report/macbook_M3
test ! -e .artifacts/performance-report-coordination/macbook_M3
test ! -e .venv
python3 tools/ci/memory_watchdog.py --limit-gib 27.9 -- \
  python3 dependencies/install_dependencies.py --reset --no-build
python3 docs/performance_reports/macbook_M3/result_tables.py validate
python3 - <<'PY'
import json
from pathlib import Path

profile = Path("docs/performance_reports/macbook_M3")
manifest = json.loads((profile / "report-workspace.json").read_text())
assert manifest["campaign_policy"]["name"] == "macbook-m3-v1"
entries = []
for path in (profile / "results").glob("*.json"):
    entries.extend(json.loads(path.read_text()).get("entries", ()))
assert len(entries) == 1646
assert all(
    entry["measurement"]["status"] == "not_available" for entry in entries
)
print("verified reset 1646-cell profile")
PY
MEASURED_SOURCE_REVISION="$(git rev-parse HEAD)"
test "$(git rev-parse HEAD)" = "$MEASURED_SOURCE_REVISION"
```

Keep `MEASURED_SOURCE_REVISION` unchanged until final publication. Generated
profile files may be dirty while measuring, but do not commit them at either
mandatory inspection pause. Assign `x86_epyc_support_lane` to `SUPPORT_TREE`;
the support lane must use its own `.venv` and `.artifacts`.

## 2. Build and authenticate that source

The build guard uses 27.9 GiB, which is below 30 decimal GB:

```bash
set -euo pipefail
test "$(git rev-parse HEAD)" = "$MEASURED_SOURCE_REVISION"
test -z "$(git status --short)"
CANDIDATE_DIR=".artifacts/candidate-$MEASURED_SOURCE_REVISION"
test ! -e "$CANDIDATE_DIR"
mkdir -p "$CANDIDATE_DIR"
env -u PYTHONPATH -u PYTHONHOME PYAMPLICOL_BUILD_MODE=candidate \
  .venv/bin/python tools/ci/memory_watchdog.py --limit-gib 27.9 -- \
  .venv/bin/python -m build --wheel --outdir "$CANDIDATE_DIR"
WHEEL_PATH="$(find "$CANDIDATE_DIR" -maxdepth 1 -type f -name '*.whl' -print)"
test "$(printf '%s\n' "$WHEEL_PATH" | sed '/^$/d' | wc -l)" -eq 1
.venv/bin/python -m pip install --force-reinstall --no-deps "$WHEEL_PATH"
env -u PYTHONPATH -u PYTHONHOME PYAMPLICOL_BUILD_MODE=candidate \
  .venv/bin/python tools/ci/memory_watchdog.py --limit-gib 27.9 -- \
  .venv/bin/python tools/developer/prepare_source_runtime.py \
    --candidate --wheel-directory "$CANDIDATE_DIR"
.venv/bin/python docs/performance_reports/macbook_M3/result_tables.py \
  refresh-profile-environment \
  --expected-source-revision "$MEASURED_SOURCE_REVISION"
```

The refresh must authenticate the source/tree, wheel, native build inputs,
installed distribution, native module, target, and candidate fingerprint.

## 3. Keep a live PDF beside the running worker

The Mac filler launches cells individually. The broad filters in Sections 4--6
are planning filters: run them first with `--dry-run`, copy the first
lowest-rank `cell_id`, and then dry-run that ID alone. Execute it only when the
single-ID plan contains exactly one cell. If it expands to dependencies, choose
the first dependency from that plan and repeat until the one-cell assertion
passes. Do not issue a broad non-dry batch.

Every one-cell command uses `--refresh-pdf end`. After it exits, confirm that
no `_prepare` or `_worker` process remains, run the following boundary refresh,
inspect the PDF, and only then launch the next cell:

```bash
set -euo pipefail
if pgrep -f 'result_tables.py .*(_prepare|_worker)' >/dev/null; then
  echo "measurement process still active; refusing publication refresh" >&2
  exit 1
fi
.venv/bin/python \
  docs/performance_reports/macbook_M3/result_tables.py recover --compile
```

`recover` merges only completed immutable attempts under the report lock, so it
is safe at that explicit boundary. Keep the current PDF open while a cell runs;
refresh it immediately afterward. If a concurrent live render is needed during
a long cell, export the last completed snapshot and compile it on another host,
never on the Mac measurement host.

## 4. Phase A: AmpliCol and recurrence only

Do not launch compiled or eager cells yet. For each `N` from 1 through 9, use
the following planner and one-cell command manually, in increasing order:

```bash
set -euo pipefail
.venv/bin/python docs/performance_reports/macbook_M3/result_tables.py populate \
  --n-final N --mode amplicol --mode recurrence \
  --missing-only --artifact-policy reuse \
  --workers 1 --cell-cores 1 --target-runtime 5 \
  --max-ram-gb 30 --refresh-pdf never --dry-run

CELL_ID='copy-one-cell-id-from-the-plan'
test "$(
  .venv/bin/python \
    docs/performance_reports/macbook_M3/result_tables.py populate \
    --cell-id "$CELL_ID" --missing-only --artifact-policy reuse \
    --workers 1 --cell-cores 1 --target-runtime 5 \
    --max-ram-gb 30 --refresh-pdf never --dry-run |
  python3 -c 'import json,sys; print(len(json.load(sys.stdin)["cells"]))'
)" -eq 1
.venv/bin/python docs/performance_reports/macbook_M3/result_tables.py populate \
  --cell-id "$CELL_ID" --missing-only --artifact-policy reuse \
  --workers 1 --cell-cores 1 --target-runtime 5 \
  --max-ram-gb 30 --refresh-pdf end
.venv/bin/python docs/performance_reports/macbook_M3/result_tables.py audit
```

Never wrap the multiplicities in an unattended loop. One worker means exactly
one process cell at a time. The bound frontier automatically avoids higher
multiplicities in a lane whose preceding multiplicity reached a resource
terminal.

After each `N`, repeat the same `populate` command with `--dry-run`. Its
`scheduled` count must be zero before moving to `N+1`. Inspect all
AmpliCol/recurrence pages through that multiplicity and require every
applicable slot to be numerical, `>30GB`, or an authenticated derived frontier
status. Ordinary timeout, skip, error, unsupported result, validation failure,
or unexplained `N/A` is a defect.

### Mandatory pause A

After all AmpliCol and recurrence ranges are closed, stop every worker and
finish one live refresh. Send the user:

- a clickable link to the current `macbook_M3/pyAmpliCol.pdf`;
- the raw status counts by multiplicity and mode;
- the zero-scheduled dry-run result for all AmpliCol/recurrence cells; and
- any resource-frontier or support-lane findings.

Wait for the user's explicit approval. Do not start any remaining Z variant,
compiled process matrix, eager process matrix, or scalar cell before it.

## 5. Phase B: finish both Z tables

After approval A, fill only the remaining compiled/eager Z variants. For each
`N=1,...,9`, in increasing order:

```bash
set -euo pipefail
.venv/bin/python docs/performance_reports/macbook_M3/result_tables.py populate \
  --dataset z_builtin_sm --dataset z_external_sm \
  --n-final N --mode compiled --mode eager \
  --missing-only --artifact-policy reuse \
  --workers 1 --cell-cores 1 --target-runtime 5 \
  --max-ram-gb 30 --refresh-pdf never --dry-run

CELL_ID='copy-one-cell-id-from-the-plan'
test "$(
  .venv/bin/python \
    docs/performance_reports/macbook_M3/result_tables.py populate \
    --cell-id "$CELL_ID" --missing-only --artifact-policy reuse \
    --workers 1 --cell-cores 1 --target-runtime 5 \
    --max-ram-gb 30 --refresh-pdf never --dry-run |
  python3 -c 'import json,sys; print(len(json.load(sys.stdin)["cells"]))'
)" -eq 1
.venv/bin/python docs/performance_reports/macbook_M3/result_tables.py populate \
  --cell-id "$CELL_ID" --missing-only --artifact-policy reuse \
  --workers 1 --cell-cores 1 --target-runtime 5 \
  --max-ram-gb 30 --refresh-pdf end
.venv/bin/python docs/performance_reports/macbook_M3/result_tables.py audit
```

Use the same zero-scheduled dry-run and full visual checkpoint before advancing
to the next multiplicity.

### Mandatory pause B

When both Z tables are closed for their complete declared ranges, stop all
workers, refresh once more, send the PDF link and status/dry-run evidence to
the user, and wait for explicit approval. Do not start the remaining process
matrices or scalar ladders before it.

## 6. Phase C: compiled/eager matrices and remaining tables

After approval B, run all remaining compiled and eager cells multiplicity by
multiplicity:

```bash
set -euo pipefail
.venv/bin/python docs/performance_reports/macbook_M3/result_tables.py populate \
  --n-final N --mode compiled --mode eager \
  --missing-only --artifact-policy reuse \
  --workers 1 --cell-cores 1 --target-runtime 5 \
  --max-ram-gb 30 --refresh-pdf never --dry-run

CELL_ID='copy-one-cell-id-from-the-plan'
test "$(
  .venv/bin/python \
    docs/performance_reports/macbook_M3/result_tables.py populate \
    --cell-id "$CELL_ID" --missing-only --artifact-policy reuse \
    --workers 1 --cell-cores 1 --target-runtime 5 \
    --max-ram-gb 30 --refresh-pdf never --dry-run |
  python3 -c 'import json,sys; print(len(json.load(sys.stdin)["cells"]))'
)" -eq 1
.venv/bin/python docs/performance_reports/macbook_M3/result_tables.py populate \
  --cell-id "$CELL_ID" --missing-only --artifact-policy reuse \
  --workers 1 --cell-cores 1 --target-runtime 5 \
  --max-ram-gb 30 --refresh-pdf end
.venv/bin/python docs/performance_reports/macbook_M3/result_tables.py audit
```

Already completed Z cells are reused and not rerun. After each `N`, require the
same command with `--dry-run` to schedule zero cells. Then render and inspect
the entire document—not just the newest page. Do not proceed to `N+1` until
every applicable slot at or below `N` is numerical or carries an authenticated
resource/frontier status and no unexplained `N/A` remains.

## 7. Numerical and visual review at every checkpoint

For successful cells check five-second timing evidence, at least five samples,
positive uncertainty and wall time, exact source/runtime/artifact identities,
resolved-sum agreement, and every required direct comparison. Recompute sample
ratios from raw JSON. In the best-mode matrices, `(A)`, `(B)`, and `(C)` must
identify recurrence, compiled JIT O3, and eager-DAG JIT O2, and the winner must
be the smallest validated wall time for that exact workload.

Render pages for inspection:

```bash
set -euo pipefail
PDF="docs/performance_reports/macbook_M3/pyAmpliCol.pdf"
QA_DIR=".artifacts/performance-report-qa/macbook_M3/checkpoint"
mkdir -p "$QA_DIR"
pdfinfo "$PDF"
pdftoppm -png -r 144 "$PDF" "$QA_DIR/page"
```

Inspect every page individually at final publication. Check headings, rows,
units, ratios, terminal labels, A/B/C codes, summaries, legends, continuation
pages, colour, spacing, and page numbers. Reject clipping, overlap, blank
blocks, unresolved references, and any overfull box.

An unexpected error or mismatch is not a resource marker. Quarantine the cell,
continue unrelated lanes at the frozen SHA, and send a complete reproduction
to the support lane. Batch related executable fixes; if a fix must land, begin
one new exact-source final epoch instead of relabelling old evidence.

## 8. Final audit and lightweight publication

The full declared catalog contains 1646 cells and 1571 direct-agreement catalog
edges. Successful endpoints must replay; resource/frontier endpoints remain
honestly unavailable and must not be counted as numerical evidence.

```bash
set -euo pipefail
git add \
  docs/performance_reports/macbook_M3/report_environment.json \
  docs/performance_reports/macbook_M3/report_environment.tex \
  docs/performance_reports/macbook_M3/results/*.json \
  docs/performance_reports/macbook_M3/result_*_table.tex \
  docs/performance_reports/macbook_M3/result_validation_summary.tex \
  docs/performance_reports/macbook_M3/pyAmpliCol.pdf
git diff --cached --check
git commit -m "Publish macbook_M3 performance report"
PUBLICATION_REVISION="$(git rev-parse HEAD)"
env -u PYTHONPATH -u PYTHONHOME \
  .venv/bin/python tools/ci/memory_watchdog.py --limit-gib 27.9 -- \
  .venv/bin/python docs/performance_reports/macbook_M3/result_tables.py \
    final-audit \
    --expected-source-revision "$MEASURED_SOURCE_REVISION" \
    --publication-revision "$PUBLICATION_REVISION" \
    --max-n-final 9 --expected-cell-count 1646
git push origin HEAD:codex/macbook-M3-full-report

# Confirm the Mac filler still holds the initial main-push token; reacquire it
# only if it was explicitly transferred earlier. Aggregate the already-audited
# profile in a clean landing worktree; do not redefine PUBLICATION_REVISION as
# the two-profile merge commit.
git fetch origin main
git merge-base --is-ancestor "$MEASURED_SOURCE_REVISION" origin/main
LANDING_TREE="../pyamplicol-macbook-M3-main-landing"
test ! -e "$LANDING_TREE"
git worktree add -b codex/macbook-M3-main-landing \
  "$LANDING_TREE" origin/main
LANDING_BASE_REVISION="$(git -C "$LANDING_TREE" rev-parse HEAD)"
git -C "$LANDING_TREE" merge --no-ff --no-edit "$PUBLICATION_REVISION"
AGGREGATE_REVISION="$(git -C "$LANDING_TREE" rev-parse HEAD)"
AUDITED_PROFILE_ARGS=(
  --audited-profile "macbook_M3=$PUBLICATION_REVISION"
)
if test -n "${X86_PUBLICATION_REVISION:-}"; then
  AUDITED_PROFILE_ARGS+=(
    --audited-profile "x86_EPYC=$X86_PUBLICATION_REVISION"
  )
fi
(
  cd "$LANDING_TREE"
  python3 -m tools.performance_report.aggregate_audit \
    --base-revision "$LANDING_BASE_REVISION" \
    --revision "$AGGREGATE_REVISION" \
    "${AUDITED_PROFILE_ARGS[@]}"
  python3 docs/performance_reports/macbook_M3/result_tables.py audit
)
if test -f \
  "$LANDING_TREE/docs/performance_reports/x86_EPYC/report-workspace.json"; then
  (
    cd "$LANDING_TREE"
    python3 docs/performance_reports/x86_EPYC/result_tables.py audit
  )
fi
git -C "$LANDING_TREE" push origin HEAD:main
git -C "$LANDING_TREE" pull --ff-only origin main
MAIN_LANDING_REVISION="$(git -C "$LANDING_TREE" rev-parse HEAD)"
test "$MAIN_LANDING_REVISION" = "$(git -C "$LANDING_TREE" rev-parse origin/main)"
```

Never add `.artifacts/`, evaluator/process artifacts, candidate wheels,
prepared models, attempts, logs, locks, coordination state, page PNGs, or
LaTeX auxiliary files. Commit raw JSON, generated TeX, environment metadata,
and the reviewed PDF.

Coordinate the final main advance with `x86_EPYC`; only one task holds the
main-push token at a time. Announce both the independently audited
`PUBLICATION_REVISION` and aggregate `MAIN_LANDING_REVISION`. If the main push
is rejected because the other report landed first, fetch and merge
`origin/main` into the landing worktree, obtain the sibling's independently
audited publication SHA, repeat the aggregate and structural audits, and retry
without force. Never merge sibling-profile output into the frozen measurement
worktree.

## Standalone copy

```bash
set -euo pipefail
python3 docs/result_tables.py export-profile macbook_M3 /absolute/output/path
cd /absolute/output/path
python3 build_pdf.py
```

The export contains raw data, TeX, and PDF, but no evaluator artifacts or
machine-local campaign state.
