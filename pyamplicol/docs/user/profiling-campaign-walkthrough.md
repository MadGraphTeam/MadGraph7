<!-- SPDX-License-Identifier: 0BSD -->

# Profiling campaign: from measurements to the PDF

The profiling campaign answers a practical physics-computing question: for a
fixed process, multiplicity, colour treatment, and workload, which pyAmpliCol
execution mode is both numerically validated and fastest on *this* machine?
It also records how that result compares with original AmpliCol and, where
appropriate, MadGraph.

Think of a copied campaign as a self-contained electronic lab notebook. It
contains the catalogue of measurements to make, the report sources, and a
visible `campaign_artifacts/` directory in which every attempt, log, generated
process output, and current result is recorded. The controller can stop and
resume without turning a long scan into one fragile all-or-nothing job.

Here **current result** means the latest completed result that the controller
would use in the report; it does not mean “a worker that is currently
running.” A new attempt replaces that pointer only after its record is
complete.

The campaign is deliberately broader than the ordinary `pyamplicol profile`
command. `profile` times one process output that you already selected. The
campaign generates many outputs, checks them against independent numerical
references, profiles the accepted results under matched workloads, and fills
the tables in `pyAmpliCol.pdf`.

## Create an independent campaign workspace

Start from an installed pyAmpliCol environment and copy the packaged template
into a new or empty directory:

```console
pyamplicol profiling-campaign copy ./my-profiling-campaign
cd ./my-profiling-campaign
./steer_performance_campaign.py --help
```

The copied directory is the unit to archive or move. All measurement state
stays below it; the controller does not consult an old repository-level
`.artifacts` directory.

Two optional external programs extend the comparisons:

```console
pyamplicol profiling-campaign copy ./my-full-campaign \
  --local-amplicol /path/to/clean/original-AmpliCol-checkout \
  --local-madgraph /path/to/MG5_aMC
```

- Original AmpliCol supplies a legacy numerical diagnostic and the performance
  denominator in the “versus AmpliCol” tables. It is not a correctness
  authority for pyAmpliCol. The checkout must be clean and complete and expose
  the maintained profiling and colour-probe interface from PR #12.
- MadGraph standalone supplies the independent full-colour reference for the
  UFO Standard Model comparison views. The path must contain executable
  `bin/mg5_aMC` and the standard `models/sm` UFO model.

The saved paths are defaults for that copy. A later `run --original-amplicol
PATH` or `run --madgraph PATH` overrides one invocation.

The reset JSON and TeX tables shipped in a fresh copy are **templates**, not
benchmark predictions. Their cells begin unmeasured or `N/A`; they are filled
from results obtained on your machine. In particular, do not expect a reset
copy to contain the example numbers shown later on this page.

## First preview, then run a small real cell

Before using substantial CPU time, preview the selected work:

```console
./steer_performance_campaign.py run \
  --dry-run \
  --table matrix \
  --process-id 1 \
  --multiplicity 1 \
  --color-approximation lc \
  --generation-mode non-union-flow \
  --generation-engine recurrence \
  --model built_in \
  --no-dependencies-added
```

The preview separates the cells you requested from any prerequisite or
numerical-reference cells that the controller would add. It also prints the
public pyAmpliCol recipe corresponding to each runnable cell, explicitly
labelling any pre-generation template or report-protocol exception.

Remove `--dry-run` to perform the deliberately small installation smoke test:

```console
./steer_performance_campaign.py run \
  --workers 1 \
  --table matrix \
  --process-id 1 \
  --multiplicity 1 \
  --color-approximation lc \
  --generation-mode non-union-flow \
  --generation-engine recurrence \
  --model built_in \
  --no-dependencies-added \
  --no-dashboard
```

This measures only the recurrence cell for `d d~ > Z`, at final-state
multiplicity one. It is an installation check, not a representative estimate
of the time needed for a broad high-multiplicity campaign.

Release wheels omit the optional Ratatui dashboard bindings. Their campaigns
continue headlessly, as though `--no-dashboard` had been supplied. A
contributor installation prepared with those optional bindings can show the
live coloured dashboard.

## Select the physics question, not individual shell jobs

The campaign catalogue is a matrix. Useful selector dimensions include:

| Selector | What it chooses | Examples |
| --- | --- | --- |
| `--table` | A report surface | `matrix`, `matrix_best`, `z_table`, `reference`, `scalar`, or an exact dataset ID |
| `--process-id` | One of the numbered process families | `1`, or a catalogue key or quoted process |
| `--multiplicity` | Number of final-state particles, `n` | `3 4` |
| `--color-approximation` | Colour treatment | `lc`, `nlc`, `full` |
| `--generation-mode` | LC layout or contracted workload | `non-union-flow`, `union-flow`, `contracted` |
| `--generation-engine` | Program or execution mode | `amplicol`, `recurrence`, `compiled`, `eager`, `on-the-fly` |
| `--model` | Model preparation used by the cell | `built_in`, `sm_ufo` |
| `--cell-id` / `--cell-id-file` | Exact canonical cells | IDs printed by `inspect` or the summary files |

Multiple values within one selector are alternatives; different selector
dimensions are combined. For example:

```console
./steer_performance_campaign.py run \
  --dry-run \
  --table matrix \
  --multiplicity 3 4 \
  --generation-engine recurrence compiled \
  --model built_in sm_ufo
```

means “multiplicity 3 **or** 4, recurrence **or** compiled, and built-in
**or** UFO-SM where those catalogue cells exist.” An omitted dimension means
all of its applicable values. Use `--help` to see the exact current choices and
aliases before scripting a large selection.

By default, the controller also schedules the available numerical authority
needed to accept each requested candidate. `--no-dependencies-added` suppresses
optional independent-reference cells, but never removes unavoidable model,
process, or selector prerequisites. It is useful for a narrow smoke test; a
publication campaign normally keeps those independent checks.

### Resource controls

The safe defaults are one worker, one core per worker, a one-hour
generation/preparation limit, a one-hour whole-worker limit, and a decimal
30 GB RAM cap for each supervised worker process tree. The main controls are:

```text
--workers N
--cores-per-worker N
--generation-time-limit SECONDS
--worker-wall-limit SECONDS
--ram-limit BYTES
--target-measurement-duration SECONDS
--minimum-samples N
--warmups N
--batch-size N
```

Increase parallelism only when the machine has enough RAM for that many
independent worker trees. `--fail-fast` changes validation scheduling into
multiplicity waves and stops after the first terminal non-success, including a
required mismatch, error, timeout, or authenticated resource cap. A legacy
AmpliCol disagreement remains non-terminal because original AmpliCol is not a
correctness authority.

## What one campaign cell actually does

For an ordinary successful cell, the controller:

1. resolves the catalogue process, model, colour treatment, workload, and
   execution mode;
2. prepares reusable model data when needed;
3. generates the process output and records generation time;
4. evaluates a shared deterministic phase-space point;
5. checks the resolved sum and the required cross-mode or MadGraph agreement;
6. profiles the accepted workload in repeated timing chunks; and
7. seals the attempt and makes it the cell's current result in one complete
   replacement, so a half-written record cannot become current.

Independent processes can run in parallel. Dependencies are explicit: for
example, recurrence is the correctness authority for matching compiled/eager
views, while MadGraph standalone is the full-colour authority for the UFO-SM
views. For LC on-the-fly cells through `n <= 4`, the matching recurrence cell
checks the complete resolved component array; no compiled output is created
merely to perform that gate.

The common deterministic point matters. A timing ratio is published only when
the candidate and denominator are linked to the intended same-point comparison
and the required validation has passed. This prevents a visually plausible
ratio from joining unrelated measurements.

## Watch, stop, and resume without losing completed work

In the interactive dashboard, `Ctrl-C` stops the campaign safely. `Esc` first
closes an open command drawer; otherwise it performs the same orderly stop. In
headless mode, use `Ctrl-C`. The controller stops dispatch, terminates each
supervised process tree with its configured grace period, preserves completed
currents and compact interruption evidence, removes its live lease, writes the
summary IDs, and exits with the conventional interrupted status (code 130).

There is no supported “hold” or in-memory pause command. To pause a long scan,
stop it orderly and later run the same command again. Compatible successful
results from the same source revision (code version) are reused automatically;
a cell with no compatible successful current is attempted again. If an older
successful current survived the interrupted attempt, the ordinary rerun reuses
that result; add `--force-refresh` only when you intentionally require a fresh
attempt. Historical results from another source revision participate only when
you explicitly pass `--continue-across-revisions`.

With dashboard bindings installed, another terminal can take a read-only
snapshot of the newest live campaign:

```console
./steer_performance_campaign.py \
  dashboard-snapshot --live --width 160 --height 48
```

This reads compact coordination data; it cannot pause, resume, or stop the
campaign. Use `--instance ID_OR_PREFIX` when several campaign invocations are
active. Without the optional dashboard bindings, this subcommand explains how
to install them rather than attaching to a wheel-owned headless run.

### Replay failures or policy caps

After normal completion or an orderly interrupt, `campaign_summary_ids/`
contains one sorted text file per non-success status. Typical names include
`error.txt`, `blocked_dependency.txt`, `validation_failed.txt`,
`worker_timeout.txt`, `interrupted.txt`, and `unverified.txt`. Replay exact
cells with:

```console
./steer_performance_campaign.py run \
  --cell-id-file campaign_summary_ids/error.txt \
                 campaign_summary_ids/validation_failed.txt \
  --force-refresh
```

Replaying a blocked cell includes its required prerequisites. An unverified
timing result is not a successful current; replay `unverified.txt` without
`--force-refresh` once its authority is available.

Alternatively, keep ordinary table/process selectors and filter them by their
latest state:

```console
./steer_performance_campaign.py run \
  --table matrix \
  --generation-engine recurrence compiled \
  --rerun-failed --rerun-capped
```

`--rerun-failed` selects unsuccessful outcomes that are not authenticated
policy caps. `--rerun-capped` selects resource-policy caps. Together they take
the union. Successful, static, and never-attempted cells are not direct retry
targets. These retry flags cannot be combined with `--force-refresh`; a retry
selection with no matches exits successfully and leaves the existing summary
untouched.

## What is retained as evidence

The controller owns `campaign_artifacts/`. A cell normally retains a compact
current pointer and one or more uniquely named attempt directories. Depending
on the cell, an attempt can contain:

- a manifest describing the immutable attempt;
- `result.json` and `worker-result.json`;
- `worker.log`, streamed progress events, and a phase timeline;
- resource measurements and termination diagnostics; and
- an `artifact/` payload containing the generated process output or external
  reference workspace needed to reproduce the measurement.

Prepared models, model caches, live coordination data, and publication staging
also stay below the campaign directory. Treat their exact subdirectory names
as controller internals; use `inspect`, the summary ID files, and the current
result JSON rather than editing pointers by hand.

By default, heavy payloads from every attempt are retained. If disk space is
more important than reproducing obsolete attempts byte for byte, launch with:

```console
./steer_performance_campaign.py run \
  --cleanup-artifacts \
  --workers 1 --table matrix --process-id 1 --multiplicity 1 \
  --color-approximation lc --generation-mode non-union-flow \
  --generation-engine recurrence --model built_in \
  --no-dependencies-added --no-dashboard
```

This compacts obsolete, sealed, non-current attempts by removing their complete
heavy `artifact/` directories. Those directories may contain a generated
process output or an external-reference workspace, not only an evaluator.
Compact metadata and any result, log, progress events, and timeline evidence
present for the attempt remain. Current payloads and payloads borrowed by
equivalent current results remain protected. During fail-fast termination,
failed and cancelled attempts retain their full evidence.

Temporary publication staging directories and live coordination leases are not
historical evidence. Staging is removed when a refresh attempt ends, whether it
succeeds or fails; the live lease is removed when the campaign invocation ends
or is stopped orderly. Do not prune the campaign with broad shell cleanup
commands.

### Reset or branch a campaign safely

Use another `profiling-campaign copy` for an independent machine, compiler, or
experimental branch. To reset one existing destination intentionally:

```console
pyamplicol profiling-campaign copy ./my-profiling-campaign --force
```

Stop its active controller first. `--force` overwrites the managed template
files and resets only that destination's `campaign_artifacts/`, generated PDF,
summary-ID directory, measurement lineage, and known LaTeX byproducts. It
preserves unrelated files. It also preserves saved AmpliCol and MadGraph paths
unless new `--local-amplicol` or `--local-madgraph` values are supplied.

This narrow reset is the supported replacement for manually deleting a broad
temporary directory or guessing which attempt files are safe to remove.

## Turn current results into the report

When the desired cells have finished, publish one internally consistent report
snapshot:

```console
./steer_performance_campaign.py refresh-pdf
```

The controller captures a stable report snapshot containing authenticated
current results and compact terminal presentation outcomes for cells without a
current. It reads and validates that snapshot with a coloured progress display,
renders the result JSON and generated table TeX in a fresh staging directory,
runs LaTeX there, and atomically installs the new files and `pyAmpliCol.pdf`.
A failed render leaves the previous published report in place, and its staging
directory is removed.

Unresolved LaTeX references and compilation errors are fatal. Overfull boxes
are reported by interactive `refresh-pdf` so they can be reviewed. The final
maintainer audit is stricter: it authenticates the current results and physics
links, rebuilds the PDF, extracts its text, renders every page to an image, and
compares both text and pixels. Thus `refresh-pdf` is the normal campaign-owner
command; the release audit is the publication boundary rather than a second
measurement campaign.

Useful report-only controls are:

```console
./steer_performance_campaign.py refresh-pdf --list-sections
./steer_performance_campaign.py refresh-pdf \
  --remove-sections worked-zgg shared-current-dag
./steer_performance_campaign.py refresh-pdf --quiet
```

Removing sections affects only that staged PDF. It does not remove source,
measurements, result JSON, or table TeX; the next plain `refresh-pdf` restores
the complete report.

## Find the matrices and their evidence

Open `pyAmpliCol.pdf` and go to **Standard-Model Process Matrices**. The
“best measured mode versus AmpliCol” tables provide the compact overview. For
each process, multiplicity, and LC workload, the renderer independently chooses
the validated pyAmpliCol mode with the smallest measured wall time. It does not
declare one universally best mode.

The generated TeX files beside the PDF are useful when a cell is too dense on
screen. For example, the LC overview is
`result_matrix_best_builtin_sm_lc_table.tex`. The corresponding detailed raw
tables are rendered from JSON caches below `results/`, such as
`matrix_compiled_builtin_sm_lc.json` and `reference_amplicol_lc.json`.

For a human-readable view of current coverage and status, use:

```console
./steer_performance_campaign.py inspect \
  --color-approximation lc \
  --generation-engine recurrence compiled
```

Add `--format json` for stable, uncoloured machine-readable output. Exact cell
IDs in `inspect`, the current JSON, and `campaign_summary_ids/` connect a
compact table status back to its retained attempt evidence. The PDF does not
turn every timing into a clickable evidence link; the authenticated linkage is
stored in the record's validation and provenance fields, including the shared
point and reference relationship.

## Read one real “Best vs AmpliCol” cell

The following is a populated snapshot from one MacBook M3 campaign. It is a
worked notation example, **not** a performance promise for another machine,
compiler, source revision, or model preparation. Take process ID 1 at
`n = 3`, which is

```text
d d~ -> Z + (n-1)g  =  d d~ -> Z g g
```

The LC table contains two aligned workloads in the fixed order
**selected flow, helicity sum | all flows, one helicity**:

| Table row | Original AmpliCol | Best validated pyAmpliCol mode |
| --- | --- | --- |
| generation `[s]` | `2.19 | 0.000263` | `(c) x1.63 | 2.76 (c)` |
| runtime `[microseconds/point]` | `0.461 | 0.328` | `([x1.64] x1.64) | ([x0.945] x0.945)` |

Here is every visible entry:

- `2.19` seconds is original AmpliCol's selected-flow process-library
  generation time.
- `0.000263` seconds is original AmpliCol's direct all-flow setup time. This is
  a different setup boundary from pyAmpliCol process generation.
- `(c)` means the measured wall-time winner for that workload was **compiled
  JIT O3**. The code is muted because it is a label, not a ratio. Current
  tables use `(r)`, `(c)`, `(e)`, and, for eligible LC cells, `(o)` for
  recurrence JIT O2, compiled JIT O3, eager-DAG JIT O2, and on-the-fly JIT O2.
- `x1.63` is the selected-flow compiled generation time divided by the 2.19 s
  AmpliCol library-generation time. The stored values were about 3.566 s and
  2.185 s; the table prints three significant figures. It is orange because
  the ratio is at least one but below two.
- `2.76 (c)` is the absolute all-flow-union compiled generation time in
  seconds. It is intentionally not divided by `0.000263`: process generation
  and AmpliCol direct setup are not comparable operations. Absolute timings
  are printed neutrally rather than given ratio colours.
- `0.461 | 0.328` are original AmpliCol wall times in microseconds per point
  for the selected-flow and all-flow workloads, respectively.
- In `([x1.64] x1.64)`, the muted bracketed `x1.64` is the supplementary
  timing ratio. For this cell it divides compiled's authenticated
  evaluator-total clock (about 0.759 microseconds per point) by original
  AmpliCol's direct-execution clock (about 0.461 microseconds per point), the
  defined legacy fallback because AmpliCol exposes no evaluator-total clock.
  The outer `x1.64` independently divides candidate wall time by AmpliCol wall
  time. The candidate wall time was also about 0.759 microseconds per point,
  so compiled was roughly 1.64 times slower for this selected-flow workload;
  the outer ratio is orange.
- Likewise, `([x0.945] x0.945)` contains the evaluator-total/direct-execution
  supplementary ratio and the primary wall ratio for the all-flow workload.
  The candidate wall time was about 0.310 microseconds per point,
  approximately 0.945 times the 0.328 baseline: compiled was slightly faster
  there, so the outer ratio is green. Both bracketed supplementary ratios stay
  muted because they are diagnostic clocks.

The supplementary and wall ratios happen to round to the same value in this
cell; they are still independently defined clocks. Mode selection uses the
outer wall ratio, not the bracketed diagnostic ratio and not generation time.
The overview does not print the candidate's absolute runtime, but it can be
recovered from the matching detailed result or, approximately, by multiplying
the displayed ratio by the displayed baseline.

### How an OTF winner adds cold construction

There is no OTF entry in the worked cell above: the older snapshot selected
compiled from recurrence, compiled, and eager. It did not measure an OTF
candidate for this choice, so the cell does not establish compiled-versus-OTF
performance. In a current table, when OTF wins an eligible LC cell, its
generation entry includes the cost of constructing its first selected family:

```text
selected flow: ([xG] x(G+W))
all flows:      [G] G+W
```

- `G` is compact OTF process-output generation.
- `W` is the first complete **campaign profiling batch** evaluation after the
  output has been loaded.
- For selected flow, both `G` and `G+W` are divided by the matching AmpliCol
  library-generation time. The `G` ratio is muted; the normally coloured
  `G+W` ratio is the cold-start comparison.
- For all flows, `[G]` and `G+W` are absolute seconds because the AmpliCol setup
  boundary is not comparable.

In this table notation, `W` excludes artifact loading and the conventional
benchmark warm-up calls. It is also not the same measurement contract as the
public OTF `Runtime.warm_up(...)` API, which deliberately accepts exactly one
double-precision point for one selected family. The campaign times the first
complete batch needed by its profiling protocol.

## Colours, `N/A`, caps, and failures

For a candidate/baseline multiplier `x`:

- green means `x < 1`: the candidate was faster;
- orange means `1 <= x < 2`;
- red means `x >= 2`.

The colours describe timing ratios, not physics accuracy. Accuracy is a
separate acceptance gate. An orange resource-cap token such as `>5h` or
`>40GB` means a configured policy boundary was reached; a red failure or
validation token means the measurement was not accepted. Muted `N/A` denotes
an unavailable entry, while `not applicable` means the process family itself
is not defined at that multiplicity. `not run` means it exists but was outside
the measured scope. `static N/A` marks a catalogue limitation known without
launching a worker, such as an original-AmpliCol surface outside its supported
open-quark-line scope.

A compact token such as `N out.[5d35ae]` is a digested terminal status, not a
number or speed ratio. Use `inspect` or the relevant file under
`campaign_summary_ids/` to recover the exact cell ID and outcome. `not exposed`
has a different meaning: the wall measurement succeeded, but that mode did not
provide the dedicated authenticated internal timing boundary requested by the
detailed table. It is not a missing or failed wall measurement.

Unverified, capped, unsupported, and validation-failed candidates do not enter
“best” selection or performance summaries. A successful but unlinked candidate
can still be selected by its absolute wall time; it is displayed absolutely and
is excluded from timing ratios, mode-mix counts, and ratio summaries.

## Read the summary rows

Below each multiplicity block, the **mode mix** counts wall-time winners. In
the same M3 snapshot, the `n = 3` LC block showed:

```text
r:1 | c:7 | e:0
```

Among the eight selected-flow winners with a valid AmpliCol denominator, one
was recurrence, seven were compiled, and none was eager. The snapshot predates
OTF's inclusion in this overview; current eligible LC tables can append `o` in
the order `r | c | e | o`.

Every five-number summary is ordered as:

```text
minimum | maximum | median | arithmetic mean | weighted mean
```

The weighted mean is framed in the PDF. It is the ratio of summed candidate
time to summed baseline time, not the average of the individual ratios:

```text
weighted mean = sum(candidate times) / sum(baseline times)
```

For `n = 3`, the same snapshot displayed:

| Summary row | `min | max | median | mean | weighted` |
| --- | --- |
| generation, selected-flow layout | `x1.03 | x2.58 | x1.63 | x1.68 | x1.65` |
| runtime, selected flow and helicity sum | `x1.38 | x3.63 | x1.70 | x2.00 | x1.70` |
| runtime, all flows and one helicity | `x0.779 | x1.30 | x0.973 | x1.00 | x0.920` |

The generation summary uses only the comparable selected-flow generation
ratios; absolute all-flow generation is omitted. Runtime summaries use only
the primary wall ratios, not the muted supplementary ratios, and keep the two
LC workloads on separate lines. Structurally inapplicable, resource-limited,
unsupported, unverified, unlinked, and validation-failed cells are excluded.

The minimum answers “what was the best individual ratio?”, the maximum the
worst, the median the middle process, and the arithmetic mean treats every
included process equally. The framed weighted mean gives processes with larger
baseline times more influence, because it asks what would happen if the
included candidate times and baseline times were each added together.

## A campaign-owner checklist

1. Copy a fresh workspace for each machine or measurement environment.
2. Record external AmpliCol or MadGraph paths only when those comparisons are
   required.
3. Preview a narrow selector with `run --dry-run`.
4. Confirm worker count, per-worker cores, wall-time limit, and 30 GB RAM cap
   before broadening the scan.
5. Stop with `Ctrl-C` or dashboard `Esc`; resume by repeating the command.
6. Use `inspect` and `campaign_summary_ids/` to understand non-successes before
   forcing a refresh.
7. Keep full evidence by default, or request controller-managed
   `--cleanup-artifacts` for obsolete attempts.
8. Run `refresh-pdf` only after the desired current cells are in place.
9. Read ratios together with workload, mode code, validation status, and the
   machine/source identity; no timing number is universal.

For the exact controller contract and all current options, see the packaged
[campaign template README](../../src/pyamplicol/_profiling_campaign/README.md)
and [table-filling reference](../../src/pyamplicol/_profiling_campaign/TABLE_FILLING.md).
For the physical difference between the two LC workloads and the four
pyAmpliCol execution modes, continue with
[LC workloads and execution modes](lc-workloads-and-execution-modes.md). The
ordinary one-output timing workflow remains documented in
[Runtime](runtime.md#runtime-profiling).
