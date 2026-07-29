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

## Frozen-campaign fast mode

This section supersedes the per-cell approval, recovery, dry-run, cache-render,
and PDF-refresh instructions later in this runbook once the campaign has an
authenticated finalized measurement lineage.

- Keep the measured checkout, installed package, prepared models, artifact
  root, and coordination root pinned to the measured-source revision. A newer
  report-controller revision is observational and must not relabel that
  measured source.
- Run later `populate` batches from a disposable controller checkout with
  `--fast-lineage`. Route both `src/pyamplicol` and
  `dependencies/checkouts` from that controller to the live measured checkout;
  authenticate those two resolved paths before launch.
- Use `--workers 1 --cell-cores 1 --refresh-pdf never` and continuously feed
  the next independent cell. Do not wait for routine user approval between
  cells, batches, multiplicities, or phases.
- Per cell, retain only the immutable result/attempt, finite target-runtime
  evidence, numerical-agreement result, active source/runtime identity, and
  atomic `current.json` publication. Do not run `recover`, `audit`, `validate`,
  a post-plan, cache rendering, PDF compilation, or visual QA after each cell.
- Before each one-cell `populate`, write a checked-in-controller boundary
  snapshot; after `populate`, accept the reported attempt only through the
  authoritative current/attempt delta:

  ```sh
  "$PYTHON" "$CONTROLLER/docs/performance_reports/macbook_M3/result_tables.py" \
    --repo-root "$MEASURED_ROOT" --report-profile macbook_M3 \
    --artifact-root "$ARTIFACT_ROOT" \
    --coordination-root "$COORDINATION_ROOT" \
    snapshot-cell-boundary --cell-id "$CELL_ID" >"$BEFORE_SNAPSHOT"

  "$PYTHON" "$CONTROLLER/docs/performance_reports/macbook_M3/result_tables.py" \
    --repo-root "$MEASURED_ROOT" --report-profile macbook_M3 \
    --artifact-root "$ARTIFACT_ROOT" \
    --coordination-root "$COORDINATION_ROOT" \
    accept-cell-boundary --cell-id "$CELL_ID" \
    --expected-attempt-id "$ATTEMPT_ID" \
    --before-snapshot "$BEFORE_SNAPSHOT"
  ```

  The acceptance command locks the cell and authenticates an exact one-attempt
  inventory delta, the atomic current identity, the immutable manifest, result,
  and worker-result hashes, and the measurement schema. It never reads a
  rendered profile/cache and never invokes the asynchronous publisher.
- Start one report-only publisher beside the campaign. It snapshots stable
  `current.json` identities without a campaign writer lock, renders and
  compiles in a disposable copy, then holds the report lock only while
  atomically installing the validated cache/table/PDF set:

  ```sh
  python3 "$CONTROLLER/docs/performance_reports/macbook_M3/result_tables.py" \
    --repo-root "$MEASURED_ROOT" \
    --report-profile macbook_M3 \
    --artifact-root "$ARTIFACT_ROOT" \
    --coordination-root "$COORDINATION_ROOT" \
    publish-snapshot --watch --interval-seconds 600 \
    --pdf-timeout-seconds 900 --expected-page-count 59
  ```

  The ten-minute start cadence plus the fifteen-minute compile ceiling keeps a
  successful current-cache PDF at most thirty minutes behind a newly published
  cell. The measurement scheduler never renders, audits, compiles, or waits on
  this publisher. `--refresh-pdf end` requests the same publication in a
  detached one-shot process and never waits for it.
- The publisher daemon uses the private `publication/daemon.guard` plus
  `publication/daemon.json` PID state, outside the named-lock namespace; it
  does not hold a long-lived named lock. Controller lock census must ignore
  that private guard, PID state, and all publisher staging/snapshot work. The
  publisher probes
  `named/report-writer.lock` without waiting and backs off when a controller
  writer is active; populate and its boundary bookkeeping never wait for the
  publisher. The publisher holds `report-writer` only for the short atomic
  cache/table/PDF install.
- Fast-controller wrappers must use the tracked classifier rather than a raw
  `lsof +D` assertion. In the disposable wrapper, replace its local
  `require_idle` body with:

  ```python
  from tools.performance_report.campaign_activity import require_campaign_idle

  def require_idle() -> None:
      require_campaign_idle(
          coordination_root=COORDINATION_ROOT,
          entrypoints=(ENTRYPOINT, base.ENTRYPOINT),
      )
  ```

  This ignores only regular publisher-private files below
  `publication/` and the `publish-snapshot`/`pyAmpliCol.tex` render process.
  Measurement workers, native installs, `report-writer`, per-cell/named locks,
  and atomic-install owners remain blocking activity. Never copy the former
  blanket `lsof +D` implementation into a new controller.
- If interruption occurs after `populate` has published `current.json` but
  before the controller records its fast boundary, authenticate the current
  pointer, immutable manifest, and result hashes, then write only the missing
  boundary record. Resume with `--missing-only`; never rerun or duplicate the
  already successful cell. Pass the existing
  `pre-populate-current.json` and the attempt ID from `populate.json` to
  `accept-cell-boundary`; its output is the authoritative `after_current`
  record. Atomically finish `fast-boundary.json`, then resume the same batch.
  Run `recover --compile` only at a chosen publication/recovery boundary, not
  to decide whether the cell succeeded.
- Periodic publisher validation is deliberately limited to cache schema,
  snapshot/file consistency, reproducible table rendering, successful TeX
  compilation, exactly 59 pages, and no overfull boxes. Run
  `validate-snapshot` to recheck the installed snapshot. Numerical replay,
  campaign audit, dry-run planning, and full-page visual review remain final or
  explicit checkpoint gates, not periodic measurement gates. `recover` is only
  for interruption recovery.
- A cell-local failure preserves its attempt and holds only that cell and its
  dependency descendants; independent work continues. Pass every held or
  quarantined ID back to later batches as repeated `--exclude-cell-id ID`
  options. The planner also omits selected cells whose unresolved dependency
  closure reaches an excluded ID, so a batch cannot silently reselect the held
  closure. Never reset, relabel, or restart the whole campaign for a scoped
  defect. A dependency-derived historical hold is not permanent: each new
  batch must reclassify it with the checked-in helper before filtering the
  master plan:

  ```python
  from tools.performance_report.artifacts import ArtifactStore
  from tools.performance_report.campaign_holds import (
      classify_prior_held_cells,
      prior_held_history_record,
  )
  from tools.performance_report.catalog import REPORT_CATALOG
  from tools.performance_report.measurement_lineage import (
      load_measurement_lineage,
  )

  hold_store = ArtifactStore(
      artifact_root=shared.ARTIFACT_ROOT,
      lock_root=shared.COORDINATION_ROOT,
  )
  hold_observations = {}
  pattern = "campaign-controller-G-fast-catalog-*-5765/batch-summary.json"
  for summary_path in sorted(shared.ARTIFACT_ROOT.glob(pattern)):
      summary = json.loads(summary_path.read_text(encoding="ascii"))
      held_cells = summary.get("held_cells")
      shared.require(
          isinstance(held_cells, dict),
          f"completed summary has invalid held cells: {summary_path}",
      )
      for cell_id, record in held_cells.items():
          shared.require(
              isinstance(cell_id, str)
              and cell_id
              and isinstance(record, dict)
              and record.get("cell_id") == cell_id
              and isinstance(record.get("reason"), str)
              and record["reason"],
              f"completed summary has invalid hold: {summary_path}",
          )
          hold_observations.setdefault(cell_id, []).append(
              {
                  "reason": record["reason"],
                  "summary_path": str(summary_path),
                  "summary_sha256": shared.sha256(summary_path),
              }
          )
  prior_held = {
      cell_id: prior_held_history_record(cell_id, observations)
      for cell_id, observations in hold_observations.items()
  }
  active_scoped_hold_ids = {
      cell.cell_id
      for cell in REPORT_CATALOG.measurement_cells()
      if signed_zero_continuity_hold(cell.cell_id)
  }
  measurement_lineage = load_measurement_lineage(
      shared.LIVE_ROOT,
      (
          shared.LIVE_ROOT
          / "docs/performance_reports/macbook_M3"
      ),
      expected_active_revision=shared.EXPECTED_SOURCE_HEAD,
      expected_active_tree=shared.EXPECTED_SOURCE_TREE,
  )
  shared.require(
      measurement_lineage is not None,
      "fast controller has no authenticated finalized lineage",
  )
  prior_dispositions = classify_prior_held_cells(
      hold_store,
      prior_held,
      active_scoped_hold_ids=active_scoped_hold_ids,
      authenticate_current=lambda _cell, current: (
          measurement_lineage.source_for_current(
              current,
              active_revision=shared.EXPECTED_SOURCE_HEAD,
              active_tree=shared.EXPECTED_SOURCE_TREE,
          )
          is not None
      ),
  )
  prior_disposition_records = {
      cell_id: disposition.as_dict()
      for cell_id, disposition in prior_dispositions.items()
  }
  still_held_ids = {
      cell_id
      for cell_id, disposition in prior_dispositions.items()
      if not disposition.eligible
  }
  readmitted_ids = set(prior_held) - still_held_ids
  prior_held = {
      cell_id: record
      for cell_id, record in prior_held.items()
      if cell_id in still_held_ids
  }
  ```

  After constructing the wrapper's existing `selected_record`, persist the
  exact classification and released set before writing `selected-plan.json`:

  ```python
  selected_record["prior_held_dispositions"] = prior_disposition_records
  selected_record["readmitted_prior_held_ids"] = sorted(readmitted_ids)
  ```

  Replace the wrapper's old first-record `setdefault()` aggregation with the
  complete history above; otherwise a later exact-plan or signed-zero reason
  can be hidden by an earlier dependency reason. Persist
  `prior_disposition_records` in the new batch plan. A cell is re-admittable
  only when every digest-pinned historical reason is exactly
  `authenticated non-ok dependency`; exact-plan, signed-zero, and unknown
  nonempty reasons remain held. Re-admission requires no target attempt or
  current and an authenticated `ok` current for the complete catalog baseline
  and direct agreement/equivalence prerequisite closure. Missing, error, skip,
  unsupported, source-stale, malformed, or actively held prerequisites remain
  blocking. An orphaned/failed target attempt also remains blocking even when
  the target has no current. Unknown cells, dependency cycles, malformed hold
  records, and malformed currents abort classification rather than silently
  releasing a hold. The helper refuses to classify without either exact
  40--64 digit hexadecimal single-source revision/tree identities or a current
  authenticator. The frozen-G
  controller must use the exact finalized lineage envelope already bound by
  its authenticated source route, as shown above. Other finalized
  mixed-lineage controllers must pass `authenticate_current` backed by their
  fully audited `MeasurementLineage.source_for_current()` result; never force
  retained ancestor records to impersonate the active descendant source.
- A completed `worker-result.json` is not current evidence if its controller
  disappeared before attempt sealing. The publisher intentionally ignores such
  an orphan. Recovery must validate the original result and attempt files,
  seal that same immutable attempt, and atomically publish its authenticated
  `current.json`; do not hand-edit a pointer or rerun a valid worker result
  merely to recreate controller ceremony.
- Controller-only descendants do not require PREPARE, a native rebuild,
  runtime restaging, or a source bridge. Executable-source descendants use the
  authenticated scoped bridge and rerun only its certified closure.
- The HZZ orientation repair has 40 recurrence targets. Its scheduler closure
  also includes any missing original-AmpliCol baselines; those prerequisites
  are expected and are not bridge-scope expansion.
- Original AmpliCol remains limited to three open quark lines. For a process
  beyond that historical scope, preserve its report row as `unsupported`, but
  do not make it a pyAmpliCol dependency. Recurrence validates against its
  resolved sum, a precision-32 evaluation, and the applicable built-in/UFO and
  LC-layout peers; compiled/eager modes continue to validate against
  recurrence. Render valid candidate timing as an absolute compact `n.c.`
  value rather than hiding it behind the unavailable legacy ratio.

The default CLI remains strict. `--fast-lineage` is an explicit campaign
operation after full startup/bridge authentication; standalone final audit and
publication remain exhaustive.

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
  python3 dependencies/install_dependencies.py --reset --dependencies-only
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
assert len(entries) == 1666
assert all(
    entry["measurement"]["status"] == "not_available" for entry in entries
)
print("verified reset 1666-cell profile")
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

### Bounded Class-C correction of an existing campaign

Do not reset a substantially completed profile when the reviewed HZZ
orientation correction is the only executable change. Stop every worker and
publisher, retain every immutable attempt, advance the clean checkout from the
measured ancestor `A` to the reviewed descendant `D`, and—while the installed
runtime and `report_environment.json` still authenticate `A`—prepare the
fail-closed bridge:

```bash
set -euo pipefail
A='full-ancestor-measurement-SHA'
D="$(git rev-parse HEAD)"
test "$D" != "$A"
.venv/bin/python tools/developer/prepare_class_c_bridge.py \
  --report-profile macbook_M3 \
  --ancestor-revision "$A" --descendant-revision "$D" \
  --impact hzz-orientation-v1
```

The helper materializes a disposable tracked `A` worktree, binds only the
retained regular native extension, staged runtime metadata, and authenticated
ignored dependency inputs, and runs the `D` controller with `pyamplicol` from
that ancestor runtime. It prints both runtime identities and deletes the
temporary worktree after PREPARE. This avoids importing the descendant Python
package against the intentionally retained ancestor native extension. The
controller authenticates the exact `A..D` path/status/mode/blob diff, frozen
workspace policy, complete attempt history, current pins, and active
recurrence-schedule reachability. It fails if a worker is in flight or any
unreviewed executable/dependency/runtime path changed. This bootstrap is
restricted to PREPARE; normal runtime commands retain their strict
source/native checks. Now perform the exact candidate build/install sequence
from this section for `D`, without deleting the profile artifact or
coordination roots, and finalize:

```bash
set -euo pipefail
.venv/bin/python docs/performance_reports/macbook_M3/result_tables.py \
  finalize-class-c-bridge \
  --ancestor-revision "$A" --descendant-revision "$D"
.venv/bin/python docs/performance_reports/macbook_M3/result_tables.py \
  audit-source-bridge --expected-active-source-revision "$D"
```

Resume only the scheduler-selected descendant closure: the 20 built-in
`dd_zzz_jets` recurrence cells at \(n\geq3\), followed by their 20 UFO-SM
direct-agreement peers. Use `--missing-only --artifact-policy regenerate` and
the one-cell discipline below. A dry run over
`--process-key dd_zzz_jets --mode recurrence --n-final 3 --n-final 4
--n-final 5 --n-final 6 --n-final 7 --n-final 8 --n-final 9`
must initially schedule exactly 40 bridge targets and must schedule zero after
both groups close. Never rewrite provenance, copy caches, or remove an
ancestor attempt. Final audit authorizes every retained `A` current by its
immutable pin and requires all 40 targets from `D`.

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

The full declared catalog contains 1666 cells and 1560 direct-agreement catalog
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
    --max-n-final 9 --expected-cell-count 1666
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
python3 docs/arxiv/result_tables.py export-profile macbook_M3 /absolute/output/path
cd /absolute/output/path
python3 build_pdf.py
```

The export contains raw data, TeX, and PDF, but no evaluator artifacts or
machine-local campaign state.
