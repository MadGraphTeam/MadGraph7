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
  reaches a RAM or policy timeout terminal. The higher entries receive an
  authenticated frontier marker; they are not silently left as `N/A`.

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
- `codex/x86-EPYC-report-checkpoints`: report-only cluster snapshots for
  review.

Never merge or pull either cluster branch into the active Mac measurement
worktree. Executable fixes land on `main` in coordinated batches and begin a
new final measurement epoch only when necessary.

### Hourly cluster PDF review

The support lane is also the cluster publication reviewer. Once the cluster
creates its checkpoint branch, give the support lane a separate, read-only
review worktree:

```bash
git fetch origin codex/x86-EPYC-report-checkpoints
REVIEW_TREE="../pyamplicol-x86-EPYC-report-review"
test ! -e "$REVIEW_TREE"
git worktree add -b codex/x86-EPYC-report-review \
  "$REVIEW_TREE" origin/codex/x86-EPYC-report-checkpoints
git -C "$REVIEW_TREE" branch --set-upstream-to=\
origin/codex/x86-EPYC-report-checkpoints
```

At least once per hour while the cluster campaign is active, the support lane
must obtain the newest lightweight checkpoint with:

```bash
git -C "$REVIEW_TREE" pull --ff-only
PDF="$REVIEW_TREE/docs/performance_reports/x86_EPYC/pyAmpliCol.pdf"
REVIEW_SHA="$(git -C "$REVIEW_TREE" rev-parse --short=12 HEAD)"
QA_DIR=".artifacts/performance-report-qa/x86_EPYC/hourly-$REVIEW_SHA"
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
the cluster filler. Do not commit or push from the review worktree.

## 1. Establish the exact measured source

Use a new clean worktree at the final handoff on `main`:

```bash
git fetch origin
git switch main
git pull --ff-only origin main
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
test -z "$(git status --short)"
test -f docs/performance_reports/macbook_M3/TABLE_FILLING.md
test ! -e .artifacts/performance-report/macbook_M3
test ! -e .artifacts/performance-report-coordination/macbook_M3
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
git switch -c codex/macbook-M3-full-report
MEASURED_SOURCE_REVISION="$(git rev-parse HEAD)"
git push -u origin HEAD
test "$(git rev-parse HEAD)" = "$MEASURED_SOURCE_REVISION"
```

Keep `MEASURED_SOURCE_REVISION` unchanged until final publication. Generated
profile files may be dirty while measuring, but do not commit them at either
mandatory inspection pause.

## 2. Build and authenticate that source

The build guard uses 27.9 GiB, which is below 30 decimal GB:

```bash
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
PYAMPLICOL_BUILD_MODE=candidate .venv/bin/python \
  tools/developer/prepare_source_runtime.py \
  --candidate --wheel-directory "$CANDIDATE_DIR"
.venv/bin/python docs/performance_reports/macbook_M3/result_tables.py \
  refresh-profile-environment \
  --expected-source-revision "$MEASURED_SOURCE_REVISION"
```

The refresh must authenticate the source/tree, wheel, native build inputs,
installed distribution, native module, target, and candidate fingerprint.

## 3. Keep a live PDF beside the running worker

In a second terminal, run this throughout each active phase:

```bash
while true; do
  .venv/bin/python \
    docs/performance_reports/macbook_M3/result_tables.py recover --compile
  date
  sleep 120
done
```

`recover` merges only completed immutable attempts under the report lock, so it
is safe while `populate` is still running. Open the refreshed
`docs/performance_reports/macbook_M3/pyAmpliCol.pdf` repeatedly. Stop this loop
with Ctrl-C only after the active phase has received its final render.

## 4. Phase A: AmpliCol and recurrence only

Do not launch compiled or eager cells yet. For each `N` from 1 through 9, issue
the following command manually, in increasing order:

```bash
.venv/bin/python docs/performance_reports/macbook_M3/result_tables.py populate \
  --n-final N --mode amplicol --mode recurrence \
  --missing-only --artifact-policy reuse \
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
.venv/bin/python docs/performance_reports/macbook_M3/result_tables.py populate \
  --dataset z_builtin_sm --dataset z_external_sm \
  --n-final N --mode compiled --mode eager \
  --missing-only --artifact-policy reuse \
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
.venv/bin/python docs/performance_reports/macbook_M3/result_tables.py populate \
  --n-final N --mode compiled --mode eager \
  --missing-only --artifact-policy reuse \
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
.venv/bin/python docs/performance_reports/macbook_M3/result_tables.py \
  final-audit \
  --expected-source-revision "$MEASURED_SOURCE_REVISION" \
  --publication-revision "$PUBLICATION_REVISION" \
  --max-n-final 9 --expected-cell-count 1646
git push origin HEAD
```

Never add `.artifacts/`, evaluator/process artifacts, candidate wheels,
prepared models, attempts, logs, locks, coordination state, page PNGs, or
LaTeX auxiliary files. Commit raw JSON, generated TeX, environment metadata,
and the reviewed PDF.

Coordinate the final main advance with `x86_EPYC`; only one task pushes at a
time. Pull `main` with `--ff-only` afterward and verify local `HEAD`,
`origin/main`, and the announced publication SHA are identical.

## Standalone copy

```bash
python3 docs/result_tables.py export-profile macbook_M3 /absolute/output/path
cd /absolute/output/path
python3 build_pdf.py
```

The export contains raw data, TeX, and PDF, but no evaluator artifacts or
machine-local campaign state.
