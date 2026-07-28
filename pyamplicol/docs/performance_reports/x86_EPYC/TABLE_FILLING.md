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
- Use `--workers 10 --cell-cores 1 --refresh-pdf never`, bind campaign work to
  the ten assigned CPUs, and continuously feed independent cells. Do not wait
  for routine user approval between cells, batches, multiplicities, or phases.
- Establish the Symbolica concurrency ceiling with a non-report 2, then 5,
  then 10 process canary using the campaign license. Use the highest all-green
  level for Symbolica-bearing cells. This limit is independent of the
  ten-worker CPU policy; Symbolica-free original-AmpliCol lanes should use all
  ten workers whenever the dependency graph has ten ready cells.
- Per cell, retain only the immutable result/attempt, finite target-runtime
  evidence, numerical-agreement result, active source/runtime identity, and
  atomic `current.json` publication. Do not run `recover`, `audit`, `validate`,
  a post-plan, cache rendering, PDF compilation, or visual QA after each cell.
- Start one report-only publisher beside the campaign. It snapshots stable
  `current.json` identities without a campaign writer lock, renders and
  compiles in a disposable copy, then holds the report lock only while
  atomically installing the validated cache/table/PDF set:

  ```sh
  python3 "$CONTROLLER/docs/performance_reports/x86_EPYC/result_tables.py" \
    --repo-root "$MEASURED_ROOT" \
    --report-profile x86_EPYC \
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
- If interruption occurs after `populate` has published `current.json` but
  before the controller records its fast boundary, authenticate the current
  pointer, immutable manifest, and result hashes, then write only the missing
  boundary record. Resume with `--missing-only`; never rerun or duplicate the
  already successful cell.
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
  defect.
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

The Mac filler initially owns the serialized main-push token. This filler must
wait for an explicit transfer or release before creating its main landing. If
x86 finishes first, request an explicit early transfer rather than assuming
ownership.

## 2. Establish the exact measured source

From a clean cluster coordinator checkout, create a dedicated measurement
worktree. Do not switch branches in an existing checkout:

```bash
set -euo pipefail
git fetch origin
MEASURE_TREE="../pyamplicol-x86-EPYC-measure"
test ! -e "$MEASURE_TREE"
git worktree add -b codex/x86-EPYC-full-report \
  "$MEASURE_TREE" origin/main
git -C "$MEASURE_TREE" push -u origin codex/x86-EPYC-full-report
cd "$MEASURE_TREE"
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
test "$(uname -m)" = "x86_64"
grep -m1 -i 'AMD EPYC' /proc/cpuinfo
test "$(nproc)" -ge 11
test -z "$(git status --short)"
test -f docs/performance_reports/x86_EPYC/TABLE_FILLING.md
test ! -e .artifacts/performance-report/x86_EPYC
test ! -e .artifacts/performance-report-coordination/x86_EPYC
test ! -e .venv
python3 tools/ci/memory_watchdog.py --limit-gib 93.1 -- \
  python3 dependencies/install_dependencies.py --reset --dependencies-only
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
MEASURED_SOURCE_REVISION="$(git rev-parse HEAD)"
test "$(git rev-parse HEAD)" = "$MEASURED_SOURCE_REVISION"
```

The active profile policy and the copy stored at
`MEASURED_SOURCE_REVISION` must agree exactly. A CLI flag cannot weaken it.
Do not reuse another epoch's profile, artifacts, prepared models, or
coordination state.

## 3. Build and authenticate that source

The build guard uses 93.1 GiB, which is below 100 decimal GB:

```bash
set -euo pipefail
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
env -u PYTHONPATH -u PYTHONHOME PYAMPLICOL_BUILD_MODE=candidate \
  .venv/bin/python tools/ci/memory_watchdog.py --limit-gib 93.1 -- \
  .venv/bin/python tools/developer/prepare_source_runtime.py \
    --candidate --wheel-directory "$CANDIDATE_DIR"
.venv/bin/python docs/performance_reports/x86_EPYC/result_tables.py \
  refresh-profile-environment \
  --expected-source-revision "$MEASURED_SOURCE_REVISION"
```

Record and retain the source/tree, wheel, native build inputs, installed
distribution, native module, candidate fingerprint, target triple, and CPU
features. Every result and censor binds to that identity.

### Bounded Class-C correction of an existing campaign

Do not reset a substantially completed profile when the reviewed HZZ
orientation correction is the only executable change. Stop all ten workers and
the live publisher, retain every immutable attempt, advance the clean checkout
from measured ancestor `A` to reviewed descendant `D`, and—while the installed
runtime and `report_environment.json` still authenticate `A`—prepare the
fail-closed bridge:

```bash
set -euo pipefail
A='full-ancestor-measurement-SHA'
D="$(git rev-parse HEAD)"
test "$D" != "$A"
.venv/bin/python tools/developer/prepare_class_c_bridge.py \
  --report-profile x86_EPYC \
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
source/native checks. Perform the exact candidate build/install sequence from
this section for `D`, without deleting the profile artifact or coordination
roots, then finalize:

```bash
set -euo pipefail
.venv/bin/python docs/performance_reports/x86_EPYC/result_tables.py \
  finalize-class-c-bridge \
  --ancestor-revision "$A" --descendant-revision "$D"
.venv/bin/python docs/performance_reports/x86_EPYC/result_tables.py \
  audit-source-bridge --expected-active-source-revision "$D"
```

Resume only the scheduler-selected descendant closure: first the 20 built-in
`dd_zzz_jets` recurrence cells at \(n\geq3\), then their 20 UFO-SM
direct-agreement peers. Use
`--missing-only --artifact-policy regenerate --workers 10 --cell-cores 1`
under the normal CPU/RSS policy. A dry run over
`--process-key dd_zzz_jets --mode recurrence --n-final 3 --n-final 4
--n-final 5 --n-final 6 --n-final 7 --n-final 8 --n-final 9`
must initially schedule exactly 40 bridge targets and must schedule zero after
both groups close. Never rewrite provenance, copy caches, or remove an
ancestor attempt. Final audit authorizes each retained `A` current by its
immutable pin and requires all 40 targets from `D`.

### EPYC continuity for the recurrence-summary cap

Use this second Class-C impact only for the measured EPYC ancestor and the
reviewed summary-cap descendant. Stop all workers and the publisher first. The
four `GenerationError` currents, including their exact reported byte counts,
must still be current. Seal the one completed controller-orphaned AmpliCol
worker result before freezing the bridge. The audited predecessor preserves
accepted currents from both of its source epochs; all other terminal currents
are digest-covered but receive no new source authorization. In addition to the
summary-cap closure, the bridge admits only the signed-zero helicity repair's
16 AmpliCol reference cells, eight directly blocked recurrence cells, and their
exact 40-cell agreement closure for `dd_epemzh_jets` and `dd_ttzh_jets` at
\(n=4,\ldots,7\). The prepare step
fails closed on any changed current, failure message, predecessor pin, or path
outside the reviewed `A..D` allowlist.

```bash
set -euo pipefail
A='be11d8304fdc04893dc0e23e9619be848126e3bc'
FIX='b61e68e92eff2f2a77bfc7830c12cd99ceeaa71a'
test "$(git rev-parse HEAD)" = "$A"
git fetch origin main codex/recurrence-summary-cap-continuity
D="$(git rev-parse origin/codex/recurrence-summary-cap-continuity)"
test "$(git rev-parse origin/main)" = "$D"
CONTROLLER_PARENT="$(mktemp -d)"
CONTROLLER="$CONTROLLER_PARENT/summary-cap-controller"
git worktree add --detach "$CONTROLLER" "$D"
.venv/bin/python \
  "$CONTROLLER/docs/performance_reports/x86_EPYC/result_tables.py" \
  --repo-root "$PWD" --report-profile x86_EPYC \
  seal-existing-worker-result \
  --cell-id reference-amplicol-lc-n8-gg-gluons-selected-flow \
  --attempt-id 83e5c9c7-dbf6-4d61-b724-f4580df2cfa3 \
  --worker-result-sha256 \
  5f3a42f9e3d034efedd8b670e7acbf2b54a427449106dbabc29050f3d93afbe6 \
  --artifact-policy regenerate --expected-source-revision "$A"
# The JSON output must report
# "resource_monitoring":"unavailable-pinned-worker-result". The sealed result
# preserves the worker's exact external-supervisor/null-RSS resource record.
.venv/bin/python docs/performance_reports/x86_EPYC/result_tables.py \
  audit-source-bridge --expected-active-source-revision "$A"
git merge --ff-only "$D"
test "$(git rev-parse HEAD)" = "$D"
test "$D" != "$A"
git merge-base --is-ancestor "$A" "$FIX"
git merge-base --is-ancestor "$FIX" "$D"
.venv/bin/python tools/developer/prepare_class_c_bridge.py \
  --report-profile x86_EPYC \
  --ancestor-revision "$A" --descendant-revision "$D" \
  --impact recurrence-summary-cap-v1
```

The helper recreates the exact retained `A` package/runtime namespace
automatically. Do not symlink the descendant package over `src/pyamplicol`,
rerun `just dev-install` before PREPARE, or hand-build a mixed controller
worktree. The JSON preflight must name `A` as the package revision, `D` as the
tools revision, and must reproduce the native-extension and native-input
digests recorded by the retained source runtime.

Build, install, prepare, and authenticate the exact candidate for `D` using
the commands in Section 3 without deleting either profile artifact root. Then
finalize and audit the frozen bridge:

```bash
set -euo pipefail
.venv/bin/python docs/performance_reports/x86_EPYC/result_tables.py \
  finalize-class-c-bridge \
  --ancestor-revision "$A" --descendant-revision "$D"
.venv/bin/python docs/performance_reports/x86_EPYC/result_tables.py \
  audit-source-bridge --expected-active-source-revision "$D"
```

Run the four failed selected-flow cells first, their four built-in all-flow
dependents second, and the eight UFO-SM peers last. Keep the normal EPYC
resource policy:

```bash
set -euo pipefail
RUNNER=(taskset -c 0-9 .venv/bin/python \
  docs/performance_reports/x86_EPYC/result_tables.py populate)
COMMON=(--missing-only --artifact-policy regenerate --cell-cores 1 \
  --target-runtime 5 --max-ram-gb 100 --allow-symbolica-parallel \
  --refresh-pdf end)
"${RUNNER[@]}" "${COMMON[@]}" --workers 4 \
  --cell-id matrix-recurrence-builtin-sm-lc-n7-gg-gluons-selected-flow \
  --cell-id matrix-recurrence-builtin-sm-lc-n8-dd-tt-jets-selected-flow \
  --cell-id matrix-recurrence-builtin-sm-lc-n9-dd-z-jets-selected-flow \
  --cell-id matrix-recurrence-builtin-sm-lc-n9-ud-w-jets-selected-flow
"${RUNNER[@]}" "${COMMON[@]}" --workers 4 \
  --cell-id matrix-recurrence-builtin-sm-lc-n7-gg-gluons-all-flow \
  --cell-id matrix-recurrence-builtin-sm-lc-n8-dd-tt-jets-all-flow \
  --cell-id matrix-recurrence-builtin-sm-lc-n9-dd-z-jets-all-flow \
  --cell-id matrix-recurrence-builtin-sm-lc-n9-ud-w-jets-all-flow
"${RUNNER[@]}" "${COMMON[@]}" --workers 8 \
  --cell-id matrix-recurrence-ufo-sm-lc-n7-gg-gluons-selected-flow \
  --cell-id matrix-recurrence-ufo-sm-lc-n7-gg-gluons-all-flow \
  --cell-id matrix-recurrence-ufo-sm-lc-n8-dd-tt-jets-selected-flow \
  --cell-id matrix-recurrence-ufo-sm-lc-n8-dd-tt-jets-all-flow \
  --cell-id matrix-recurrence-ufo-sm-lc-n9-dd-z-jets-selected-flow \
  --cell-id matrix-recurrence-ufo-sm-lc-n9-dd-z-jets-all-flow \
  --cell-id matrix-recurrence-ufo-sm-lc-n9-ud-w-jets-selected-flow \
  --cell-id matrix-recurrence-ufo-sm-lc-n9-ud-w-jets-all-flow
```

Then run the signed-zero closure in dependency order: eight selected-flow and
eight all-flow AmpliCol references, eight built-in recurrence selected-flow
cells, eight built-in recurrence all-flow cells, 16 UFO-SM recurrence cells,
and eight all-flow cells in each compiled and eager mode. These selectors are
exact and must each schedule 8, 8, 8, 8, 16, 8, and 8 cells respectively:

```bash
set -euo pipefail
RUNNER=(taskset -c 0-9 .venv/bin/python \
  docs/performance_reports/x86_EPYC/result_tables.py populate)
COMMON=(--missing-only --artifact-policy regenerate --cell-cores 1 \
  --target-runtime 5 --max-ram-gb 100 --allow-symbolica-parallel \
  --refresh-pdf end)
"${RUNNER[@]}" "${COMMON[@]}" --workers 8 \
  --mode amplicol --accuracy lc \
  --workload selected-flow \
  --process-key dd_epemzh_jets --process-key dd_ttzh_jets \
  --n-final 4 --n-final 5 --n-final 6 --n-final 7
"${RUNNER[@]}" "${COMMON[@]}" --workers 8 \
  --mode amplicol --accuracy lc \
  --workload all-flow \
  --process-key dd_epemzh_jets --process-key dd_ttzh_jets \
  --n-final 4 --n-final 5 --n-final 6 --n-final 7
"${RUNNER[@]}" "${COMMON[@]}" --workers 8 \
  --mode recurrence --model builtin_sm --accuracy lc \
  --workload selected-flow \
  --process-key dd_epemzh_jets --process-key dd_ttzh_jets \
  --n-final 4 --n-final 5 --n-final 6 --n-final 7
"${RUNNER[@]}" "${COMMON[@]}" --workers 8 \
  --mode recurrence --model builtin_sm --accuracy lc \
  --workload all-flow \
  --process-key dd_epemzh_jets --process-key dd_ttzh_jets \
  --n-final 4 --n-final 5 --n-final 6 --n-final 7
"${RUNNER[@]}" "${COMMON[@]}" --workers 8 \
  --mode recurrence --model ufo_sm --accuracy lc --workload both \
  --process-key dd_epemzh_jets --process-key dd_ttzh_jets \
  --n-final 4 --n-final 5 --n-final 6 --n-final 7
"${RUNNER[@]}" "${COMMON[@]}" --workers 8 \
  --mode compiled --model builtin_sm --accuracy lc --workload all-flow \
  --process-key dd_epemzh_jets --process-key dd_ttzh_jets \
  --n-final 4 --n-final 5 --n-final 6 --n-final 7
"${RUNNER[@]}" "${COMMON[@]}" --workers 8 \
  --mode eager --model builtin_sm --accuracy lc --workload all-flow \
  --process-key dd_epemzh_jets --process-key dd_ttzh_jets \
  --n-final 4 --n-final 5 --n-final 6 --n-final 7
.venv/bin/python docs/performance_reports/x86_EPYC/result_tables.py audit
.venv/bin/python docs/performance_reports/x86_EPYC/result_tables.py \
  audit-source-bridge --expected-active-source-revision "$D"
```

Repeat all ten `populate` invocations with `--dry-run`; each must report zero
scheduled cells.

Do not reset, relabel, copy, or rewrite any current or attempt. The bridge
retains every accepted ancestor current by its byte digest, including the
newly sealed orphan, carries the audited HZZ predecessor, and requires only the
reviewed 80-cell union (four summary-cap failures plus 12 dependents, and 24
signed-zero direct targets plus 40 dependents) from `D`.

## 4. Keep a live PDF beside the workers

Pin all measurement commands in Sections 6--8 to logical CPUs 0--9. Reserve
logical CPU 10 for publication recovery and checkpoint export; it is not one
of the ten measurement cores. In a dedicated publisher terminal, run:

```bash
set -euo pipefail
taskset -pc 10 "$$"
renice 19 -p "$$"
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
set -euo pipefail
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

Create a dedicated checkpoint-publisher lane with a recurring hourly wake-up;
it must remain active independently of a long or mandatory-completion
`populate` call. At least once per hour, and at each multiplicity or
mandatory-pause boundary, finish `recover --compile` on reserved CPU 10,
export a portable snapshot, and push only raw JSON, generated TeX/environment
metadata, and the PDF:

```bash
set -euo pipefail
taskset -pc 10 "$$"
renice 19 -p "$$"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
EXPORT_DIR=".artifacts/hourly-x86-EPYC-$STAMP"
test ! -e "$EXPORT_DIR"
.venv/bin/python docs/arxiv/result_tables.py export-profile \
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
fi
git -C "$CHECKPOINT_TREE" push origin \
  codex/x86-EPYC-report-checkpoints
CHECKPOINT_SHA="$(git -C "$CHECKPOINT_TREE" rev-parse HEAD)"
```

Never copy or stage evaluator/process artifacts, wheels, models, logs, attempts,
locks, coordination state, auxiliary files, entry points, manifests, or prose.
Notify `x86_epyc_support_lane` after every hourly attempt, including when the
snapshot is unchanged; include `STAMP` and `CHECKPOINT_SHA`. It will still pull
the branch, render into a timestamped QA directory, inspect every PDF page, and
send timestamped feedback. A checkpoint is review evidence, not a new
measurement source and must not invalidate or relabel existing attempts.

## 6. Phase A: AmpliCol and recurrence only

Do not launch compiled or eager cells yet. For each `N=1,...,9`, in increasing
order, issue this batch manually:

```bash
set -euo pipefail
taskset -c 0-9 .venv/bin/python \
  docs/performance_reports/x86_EPYC/result_tables.py populate \
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
set -euo pipefail
taskset -c 0-9 .venv/bin/python \
  docs/performance_reports/x86_EPYC/result_tables.py populate \
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
set -euo pipefail
taskset -c 0-9 .venv/bin/python \
  docs/performance_reports/x86_EPYC/result_tables.py populate \
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
set -euo pipefail
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

The full declared catalog contains 1646 cells and 1556 direct-agreement catalog
edges. The audit must separate numerically verified, `>2h`, `>100GB`,
dependency, and frontier counts; censored endpoints are never claimed as
numerical evidence.

```bash
set -euo pipefail
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
env -u PYTHONPATH -u PYTHONHOME \
  .venv/bin/python tools/ci/memory_watchdog.py --limit-gib 93.1 -- \
  .venv/bin/python docs/performance_reports/x86_EPYC/result_tables.py \
    final-audit \
    --expected-source-revision "$MEASURED_SOURCE_REVISION" \
    --publication-revision "$PUBLICATION_REVISION" \
    --max-n-final 9 --expected-cell-count 1646
git push origin HEAD:codex/x86-EPYC-full-report

# Obtain the shared main-push token from the macbook_M3 filler. Aggregate the
# already-audited profile in a clean landing worktree; do not redefine the
# profile's PUBLICATION_REVISION as the two-profile merge commit.
git fetch origin main
git merge-base --is-ancestor "$MEASURED_SOURCE_REVISION" origin/main
LANDING_TREE="../pyamplicol-x86-EPYC-main-landing"
test ! -e "$LANDING_TREE"
git worktree add -b codex/x86-EPYC-main-landing \
  "$LANDING_TREE" origin/main
LANDING_BASE_REVISION="$(git -C "$LANDING_TREE" rev-parse HEAD)"
git -C "$LANDING_TREE" merge --no-ff --no-edit "$PUBLICATION_REVISION"
AGGREGATE_REVISION="$(git -C "$LANDING_TREE" rev-parse HEAD)"
AUDITED_PROFILE_ARGS=(
  --audited-profile "x86_EPYC=$PUBLICATION_REVISION"
)
if test -n "${MAC_PUBLICATION_REVISION:-}"; then
  AUDITED_PROFILE_ARGS+=(
    --audited-profile "macbook_M3=$MAC_PUBLICATION_REVISION"
  )
fi
(
  cd "$LANDING_TREE"
  python3 -m tools.performance_report.aggregate_audit \
    --base-revision "$LANDING_BASE_REVISION" \
    --revision "$AGGREGATE_REVISION" \
    "${AUDITED_PROFILE_ARGS[@]}"
  python3 docs/performance_reports/x86_EPYC/result_tables.py audit
)
if test -f \
  "$LANDING_TREE/docs/performance_reports/macbook_M3/report-workspace.json"; then
  (
    cd "$LANDING_TREE"
    python3 docs/performance_reports/macbook_M3/result_tables.py audit
  )
fi
git -C "$LANDING_TREE" push origin HEAD:main
git -C "$LANDING_TREE" pull --ff-only origin main
MAIN_LANDING_REVISION="$(git -C "$LANDING_TREE" rev-parse HEAD)"
test "$MAIN_LANDING_REVISION" = "$(git -C "$LANDING_TREE" rev-parse origin/main)"
```

Never commit `.artifacts/`, process/evaluator artifacts, prepared models,
candidate wheels, build trees, attempts, logs, locks, coordination state, page
PNGs, or LaTeX auxiliary files. Commit raw JSON, generated TeX, environment
metadata, and the reviewed PDF.

Coordinate the final main advance with `macbook_M3`; only one task holds the
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
python3 docs/arxiv/result_tables.py export-profile x86_EPYC /absolute/output/path
cd /absolute/output/path
python3 build_pdf.py
```

The export contains raw data, TeX, and PDF and omits all process artifacts and
machine-local campaign state.
