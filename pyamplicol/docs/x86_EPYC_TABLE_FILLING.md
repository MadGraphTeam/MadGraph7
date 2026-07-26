<!-- SPDX-License-Identifier: 0BSD -->
# `x86_EPYC` table-filling runbook

## Required result and fixed policy

Produce `docs/performance_reports/x86_EPYC/pyAmpliCol.pdf` from scratch on an
AMD EPYC x86-64 host. Cover the complete declared table ranges, from low to
high final-state multiplicity; there is no hard \(n=4\) campaign ceiling.

The bound `x86-epyc-v1` policy requires:

- 10 concurrent workers, each restricted to one core;
- a five-second timing target and at least five completed samples;
- a decimal 100 GB process-tree RSS ceiling for every cell, model, colour
  treatment, layout, generation step, validation step, and profiling step;
- a two-hour process-generation ceiling except for mandatory-completion lanes;
- exact authenticated table markers `>100GB` and `>2h`; and
- no higher multiplicity in the same process lane after a lower multiplicity
  reaches a RAM or generation-time terminal. Higher entries receive an
  authenticated frontier marker rather than remaining `N/A`.

The two-hour generation ceiling does not apply to:

1. any original-AmpliCol reference cell; or
2. compiled JIT O3 and recurrence JIT O2 LC selected-flow cells—the non-union
   layout with one runtime-selected colour flow and a helicity sum.

Those lanes run to completion unless they cross the universal 100 GB ceiling
or encounter a real defect. Model preparation, identity checks, numerical
validation, and five-second profiling are outside the generation timer.

A process lane means the same process family, execution mode, model source,
colour accuracy, LC workload, backend/optimization, and Z variant. A terminal
in one lane must not suppress unrelated lanes.

Every successful timing must pass the numerical agreement graph.
Recurrence/AmpliCol uses `rtol=1e-8`, `atol=1e-15`; pyAmpliCol cross-mode,
model-route, and layout checks use `rtol=1e-12`, `atol=1e-15`. Scalar/exact
lanes also compare precision 16 with precision 32.

## 1. Connect to local support before measuring

The local table-filler must provide the task address of its dedicated
`x86_epyc_support_lane` subagent. Send a test message and receive an
acknowledgement before the first campaign command. The cluster filler may
communicate directly with that support lane or the main local filler.

The support lane works in a separate worktree and artifact root. It receives
cell-scoped reproductions, accumulates tested fixes, and coordinates batched
landings. It never edits the cluster checkout, current records, or immutable
attempts.

The campaigns must remain isolated:

- this cluster measures on `codex/x86-EPYC-full-report`;
- the Mac measures on `codex/macbook-M3-full-report`;
- hourly review snapshots use `codex/x86-EPYC-report-checkpoints`.

Each has a separate worktree, profile, artifact root, coordination root,
virtual environment, and candidate-wheel directory. Never merge or pull the
Mac or checkpoint branch into this active measurement worktree.

## 2. Establish the exact measured source

```bash
git fetch origin
git switch main
git pull --ff-only origin main
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
test "$(uname -m)" = "x86_64"
grep -m1 -i 'AMD EPYC' /proc/cpuinfo
test -z "$(git status --short)"
test -f docs/performance_reports/x86_EPYC/TABLE_FILLING.md
test ! -e .artifacts/performance-report/x86_EPYC
test ! -e .artifacts/performance-report-coordination/x86_EPYC
python3 docs/performance_reports/x86_EPYC/result_tables.py validate
python3 - <<'PY'
import json
from pathlib import Path

profile = Path("docs/performance_reports/x86_EPYC")
manifest = json.loads((profile / "report-workspace.json").read_text())
assert manifest["campaign_policy"]["name"] == "x86-epyc-v1"
entries = []
for path in (profile / "results").glob("*.json"):
    entries.extend(json.loads(path.read_text()).get("entries", ()))
assert len(entries) == 1646
assert all(
    entry["measurement"]["status"] == "not_available" for entry in entries
)
print("verified reset 1646-cell profile")
PY
git switch -c codex/x86-EPYC-full-report
MEASURED_SOURCE_REVISION="$(git rev-parse HEAD)"
git push -u origin HEAD
test "$(git rev-parse HEAD)" = "$MEASURED_SOURCE_REVISION"
```

The active profile policy and the copy stored at
`MEASURED_SOURCE_REVISION` must agree exactly. A CLI flag cannot weaken it.
Do not reuse another epoch's profile, artifacts, prepared models, or
coordination state.

## 3. Build and authenticate that source

The build guard uses 93.1 GiB, which is below 100 decimal GB:

```bash
test "$(git rev-parse HEAD)" = "$MEASURED_SOURCE_REVISION"
test -z "$(git status --short)"
CANDIDATE_DIR=".artifacts/candidate-$MEASURED_SOURCE_REVISION"
test ! -e "$CANDIDATE_DIR"
mkdir -p "$CANDIDATE_DIR"
env -u PYTHONPATH -u PYTHONHOME PYAMPLICOL_BUILD_MODE=candidate \
  .venv/bin/python tools/ci/memory_watchdog.py --limit-gib 93.1 -- \
  .venv/bin/python -m build --wheel --outdir "$CANDIDATE_DIR"
WHEEL_PATH="$(find "$CANDIDATE_DIR" -maxdepth 1 -type f -name '*.whl' -print)"
test "$(printf '%s\n' "$WHEEL_PATH" | sed '/^$/d' | wc -l)" -eq 1
.venv/bin/python -m pip install --force-reinstall --no-deps "$WHEEL_PATH"
PYAMPLICOL_BUILD_MODE=candidate .venv/bin/python \
  tools/developer/prepare_source_runtime.py \
  --candidate --wheel-directory "$CANDIDATE_DIR"
.venv/bin/python docs/performance_reports/x86_EPYC/result_tables.py \
  refresh-profile-environment \
  --expected-source-revision "$MEASURED_SOURCE_REVISION"
```

Record and retain the source/tree, wheel, native build inputs, installed
distribution, native module, candidate fingerprint, target triple, and CPU
features. Every result and censor binds to that identity.

## 4. Keep a live PDF beside the workers

In a second terminal, run:

```bash
while true; do
  .venv/bin/python \
    docs/performance_reports/x86_EPYC/result_tables.py recover --compile
  date
  sleep 120
done
```

This safely merges only completed immutable attempts under the report lock,
even while other workers remain active. Open the refreshed
`docs/performance_reports/x86_EPYC/pyAmpliCol.pdf` repeatedly. Stop the loop
with Ctrl-C only after the active phase has received its final render.

Original AmpliCol workers use one distinct pinned legacy workspace per cell
under the profile artifact root. Ten workers must never mutate one shared
legacy checkout.

## 5. Publish an hourly lightweight review checkpoint

Create a separate publication worktree once, rooted at the frozen measured
source:

```bash
CHECKPOINT_TREE="../pyamplicol-x86-EPYC-report-checkpoints"
test ! -e "$CHECKPOINT_TREE"
git worktree add -b codex/x86-EPYC-report-checkpoints \
  "$CHECKPOINT_TREE" "$MEASURED_SOURCE_REVISION"
git -C "$CHECKPOINT_TREE" push -u origin \
  codex/x86-EPYC-report-checkpoints
```

The active measurement checkout remains at `MEASURED_SOURCE_REVISION`.
Checkpoint commits are made only in `CHECKPOINT_TREE` and are never pulled
back into the measurement checkout.

At least once per hour, and at each multiplicity or mandatory-pause boundary,
finish `recover --compile`, export a portable snapshot, and push only raw JSON,
generated TeX/environment metadata, and the PDF:

```bash
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
EXPORT_DIR=".artifacts/hourly-x86-EPYC-$STAMP"
test ! -e "$EXPORT_DIR"
.venv/bin/python docs/result_tables.py export-profile \
  x86_EPYC "$EXPORT_DIR"
PROFILE_OUT="$CHECKPOINT_TREE/docs/performance_reports/x86_EPYC"
cp "$EXPORT_DIR"/results/*.json "$PROFILE_OUT/results/"
cp "$EXPORT_DIR"/report_environment.json "$PROFILE_OUT/"
cp "$EXPORT_DIR"/report_environment.tex "$PROFILE_OUT/"
cp "$EXPORT_DIR"/result_*_table.tex "$PROFILE_OUT/"
cp "$EXPORT_DIR"/result_validation_summary.tex "$PROFILE_OUT/"
cp "$EXPORT_DIR"/pyAmpliCol.pdf "$PROFILE_OUT/"
git -C "$CHECKPOINT_TREE" add \
  docs/performance_reports/x86_EPYC/report_environment.json \
  docs/performance_reports/x86_EPYC/report_environment.tex \
  docs/performance_reports/x86_EPYC/results/*.json \
  docs/performance_reports/x86_EPYC/result_*_table.tex \
  docs/performance_reports/x86_EPYC/result_validation_summary.tex \
  docs/performance_reports/x86_EPYC/pyAmpliCol.pdf
if ! git -C "$CHECKPOINT_TREE" diff --cached --quiet; then
  git -C "$CHECKPOINT_TREE" commit -m \
    "Checkpoint x86_EPYC report $STAMP"
  git -C "$CHECKPOINT_TREE" push origin \
    codex/x86-EPYC-report-checkpoints
fi
```

Never copy or stage evaluator/process artifacts, wheels, models, logs, attempts,
locks, coordination state, auxiliary files, entry points, manifests, or prose.
Notify `x86_epyc_support_lane` after every push. It will pull this branch into
its separate review worktree, inspect every PDF page, and send timestamped
feedback. A checkpoint is review evidence, not a new measurement source and
must not invalidate or relabel existing attempts.

## 6. Phase A: AmpliCol and recurrence only

Do not launch compiled or eager cells yet. For each `N=1,...,9`, in increasing
order, issue this batch manually:

```bash
.venv/bin/python docs/performance_reports/x86_EPYC/result_tables.py populate \
  --n-final N --mode amplicol --mode recurrence \
  --missing-only --artifact-policy reuse \
  --workers 10 --cell-cores 1 --target-runtime 5 \
  --max-ram-gb 100 --allow-symbolica-parallel --refresh-pdf end
.venv/bin/python docs/performance_reports/x86_EPYC/result_tables.py audit
```

Do not use an unattended multiplicity loop. The profile policy applies the
two-hour generation limit and its exceptions; do not add a whole-worker
timeout or a CLI generation limit.

After each `N`, repeat the same `populate` command with `--dry-run`. Its
`scheduled` count must be zero before moving to `N+1`. Inspect every
AmpliCol/recurrence page through that multiplicity. Accept only numerical
`ok`, authenticated `>2h`, authenticated `>100GB`, or authenticated
dependency/frontier status. An ordinary timeout, skip, error, unsupported
result, mismatch, resource-probe gap, or unexplained `N/A` is a defect.

### Mandatory pause A

After every AmpliCol and recurrence lane is closed, stop the workers and finish
one live refresh, publish the Section 5 checkpoint, and wait for the support
lane's page-by-page review. Send the user:

- a clickable link to the current `x86_EPYC/pyAmpliCol.pdf`;
- the exact checkpoint-branch commit containing that PDF;
- raw status counts by multiplicity and mode;
- the zero-scheduled dry-run result over all AmpliCol/recurrence cells; and
- all resource-frontier and support-lane findings.

Wait for explicit user approval. Do not start any remaining Z variant,
compiled process matrix, eager process matrix, or scalar cell before it.

## 7. Phase B: finish both Z tables

After approval A, fill only the remaining compiled/eager Z variants. For each
`N=1,...,9`, in increasing order:

```bash
.venv/bin/python docs/performance_reports/x86_EPYC/result_tables.py populate \
  --dataset z_builtin_sm --dataset z_external_sm \
  --n-final N --mode compiled --mode eager \
  --missing-only --artifact-policy reuse \
  --workers 10 --cell-cores 1 --target-runtime 5 \
  --max-ram-gb 100 --allow-symbolica-parallel --refresh-pdf end
.venv/bin/python docs/performance_reports/x86_EPYC/result_tables.py audit
```

Require a zero-scheduled dry-run and a complete numerical/visual checkpoint
before increasing `N`.

### Mandatory pause B

When both Z tables are closed for their complete declared ranges, stop every
worker, refresh once more, publish the Section 5 checkpoint, obtain the support
lane's full-page review, send the PDF link, checkpoint SHA, and status/dry-run
evidence to the user, and wait for explicit approval. Do not start the
remaining process matrices or scalar ladders before it.

## 8. Phase C: compiled/eager matrices and remaining tables

After approval B, run the remaining compiled/eager cells multiplicity by
multiplicity:

```bash
.venv/bin/python docs/performance_reports/x86_EPYC/result_tables.py populate \
  --n-final N --mode compiled --mode eager \
  --missing-only --artifact-policy reuse \
  --workers 10 --cell-cores 1 --target-runtime 5 \
  --max-ram-gb 100 --allow-symbolica-parallel --refresh-pdf end
.venv/bin/python docs/performance_reports/x86_EPYC/result_tables.py audit
```

Already completed Z cells are reused. After each `N`, require the same command
with `--dry-run` to schedule zero cells. Render and inspect the entire
document—not only the newest page. Do not proceed to `N+1` until every
applicable slot at or below `N` is numerical or carries an authenticated
resource/frontier status and no unexplained `N/A` remains.

## 9. Review and escalation after every batch

For successful cells verify five-second sampling, positive uncertainty and
wall time, finite generation time, exact source/runtime/artifact identities,
resolved-sum agreement, and all direct comparisons. Recompute representative
raw-JSON ratios. In the best-mode matrices, `(A)`, `(B)`, and `(C)` identify
recurrence, compiled JIT O3, and eager-DAG JIT O2; the winner must be the
smallest validated wall time for that exact workload.

```bash
PDF="docs/performance_reports/x86_EPYC/pyAmpliCol.pdf"
QA_DIR=".artifacts/performance-report-qa/x86_EPYC/checkpoint"
mkdir -p "$QA_DIR"
pdfinfo "$PDF"
pdftoppm -png -r 144 "$PDF" "$QA_DIR/page"
```

Inspect every page individually before publication. Check process coverage,
`>2h`/`>100GB`/frontier spelling, A/B/C codes, units, ratios, summaries,
legends, continuation pages, colour, spacing, page numbers, and the absence of
clipping, overlap, unresolved references, and overfull boxes.

When a defect appears, quarantine only the failing cell and continue unrelated
lanes at the frozen SHA. Send the support lane:

- measured commit/tree and profile/runtime identities;
- full cell descriptor and exact command;
- attempt/manifest IDs and digests;
- status, worker-log digest and concise tail;
- resource and authenticated phase-state evidence;
- artifact/process IDs if generation completed;
- baseline/direct-peer IDs and current-manifest digests; and
- a one-cell reproduction using a separate artifact root.

Never convert a defect or mismatch into `>2h` or `>100GB`. Continue unaffected
discovery while fixes accumulate. The cluster branch does not push executable
fixes. The support lane and main local filler test and land one batched fix
series. Only an executable/report-schema/prose/runtime change creates a new
measurement epoch; raw profile JSON, generated environment/table TeX, and PDF
updates are report-only. If a new epoch is necessary, preserve old attempts for
diagnosis, reset publication caches, clean-build, and rerun the final coherent
epoch instead of relabelling evidence.

## 10. Final audit and lightweight publication

The full declared catalog contains 1646 cells and 1571 direct-agreement catalog
edges. The audit must separate numerically verified, `>2h`, `>100GB`,
dependency, and frontier counts; censored endpoints are never claimed as
numerical evidence.

```bash
git add \
  docs/performance_reports/x86_EPYC/report_environment.json \
  docs/performance_reports/x86_EPYC/report_environment.tex \
  docs/performance_reports/x86_EPYC/results/*.json \
  docs/performance_reports/x86_EPYC/result_*_table.tex \
  docs/performance_reports/x86_EPYC/result_validation_summary.tex \
  docs/performance_reports/x86_EPYC/pyAmpliCol.pdf
git diff --cached --check
git commit -m "Publish x86_EPYC performance report"
PUBLICATION_REVISION="$(git rev-parse HEAD)"
.venv/bin/python docs/performance_reports/x86_EPYC/result_tables.py \
  final-audit \
  --expected-source-revision "$MEASURED_SOURCE_REVISION" \
  --publication-revision "$PUBLICATION_REVISION" \
  --max-n-final 9 --expected-cell-count 1646
git push origin HEAD
```

Never commit `.artifacts/`, process/evaluator artifacts, prepared models,
candidate wheels, build trees, attempts, logs, locks, coordination state, page
PNGs, or LaTeX auxiliary files. Commit raw JSON, generated TeX, environment
metadata, and the reviewed PDF.

Coordinate the final main advance with `macbook_M3`; only one task pushes at a
time. Then:

```bash
git switch main
git pull --ff-only origin main
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
```

## Standalone copy

```bash
python3 docs/result_tables.py export-profile x86_EPYC /absolute/output/path
cd /absolute/output/path
python3 build_pdf.py
```

The export contains raw data, TeX, and PDF and omits all process artifacts and
machine-local campaign state.
