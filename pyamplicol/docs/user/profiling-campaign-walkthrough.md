---
title: "Profiling campaign: from measurements to the PDF"
nav_order: 2
parent: "Get started: a gentle walkthrough"
---
<!-- SPDX-License-Identifier: 0BSD -->

# Profiling campaign: from measurements to the PDF

The profiling campaign answers a practical physics-computing question: for a
fixed process, multiplicity, colour treatment, and workload, which pyAmpliCol
execution mode is both numerically validated and fastest on *this* machine?
It also records how that result compares with original AmpliCol and, where
appropriate, MadGraph.

Think of a copied campaign as a self-contained electronic lab notebook. It
contains the catalogue of measurements to make, the report sources, and a
visible `campaign_artifacts/` directory in which every attempt and current
result is recorded, together with bounded diagnostics and any process output
that must be retained. The controller can stop and resume without turning a
long scan into one fragile all-or-nothing job.

The dashboard and report use a few precise words:

| Word | Plain-language meaning |
| --- | --- |
| **cell** | One measurement question: one process, multiplicity, model, colour treatment, workload, and execution mode |
| **controller** | The steering script that chooses cells, starts measurements, and records their outcome |
| **worker** | One supervised operating-system process doing one cell at a time; a worker may itself be allowed several CPU cores |
| **dependency** | Another cell whose numerical result is needed to validate the requested cell, such as recurrence or MadGraph |
| **process output** | The generated directory that can later evaluate phase-space points, roughly like a MadGraph standalone-process directory; internal files may call it an `artifact` |
| **current result** | The latest complete record that the next report will use, not a worker that happens to be running now |
| **provenance** | The recorded origin of a number: source revision, settings, machine, phase-space point, and validation link |
| **authenticated record** | A record whose identity and integrity fields the controller has checked; this is a data-integrity term, not a user login |
| **precision lane** | One supported numerical route, such as `p16` (ordinary double precision) or `p200` (about 200 decimal digits); “lane” does not mean another physics approximation |

A new attempt replaces a cell's current-result pointer only after the complete
record has been written. An interrupted or half-written attempt therefore
cannot silently become the number shown in the PDF.

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

Release wheels omit the optional terminal-dashboard component (implemented
with Ratatui). Their campaigns continue **headlessly**, meaning without the
interactive dashboard, as though `--no-dashboard` had been supplied. A
contributor installation prepared with that optional component can show the
live coloured dashboard.

## Select the physics question, not individual shell jobs

The campaign catalogue is a matrix: each axis describes part of the physics or
measurement question, and their allowed combinations define the cells. Useful
selector dimensions include:

| Selector | What it chooses | Examples |
| --- | --- | --- |
| `--table` | A report surface | `matrix`, `matrix_best`, `z_table`, `reference`, `scalar`, or an exact dataset ID |
| `--process-id` | One of the numbered process families | `1`, or a catalogue key or quoted process |
| `--multiplicity` | Number of final-state particles, `n` | `3 4` |
| `--color-approximation` | Colour treatment | `lc`, `nlc`, `full` |
| `--generation-mode` | Physical workload/layout | `non-union-flow`, `union-flow`, `contracted` |
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

The campaign names map to the physical LC calls as follows:

| Campaign name | Repeated request | Generated layout in recurrence/eager/compiled |
| --- | --- | --- |
| `non-union-flow` | one selected flow, summed over all helicities | `topology-replay` |
| `union-flow` | all flows, one selected helicity | `all-flow-union` |
| `contracted` | the requested axes already summed/contracted | used for NLC and full-colour campaign cells |

OTF accepts the same two LC requests, but does not write either layout during
generation. Its first selected request constructs the needed family in memory.
This is why “engine” and “generation mode” are separate selectors: the former
chooses *when and how* currents are prepared, while the latter chooses *which
physics workload* is timed.

By default, the controller also schedules the numerical reference needed to
accept each requested candidate. The dashboard calls such a prerequisite a
dependency. `--no-dependencies-added` suppresses optional independent-reference
cells, but never removes unavoidable model, process, or selector prerequisites.
It is useful for a narrow smoke test; a publication campaign normally keeps
those independent checks.

### Resource controls

The safe defaults are one worker, one core per worker, a one-hour
generation/preparation limit, a one-hour whole-worker limit, a decimal 30 GB
RAM cap for each supervised worker process tree, a 64 MiB cap for each watched
attempt-output stream, and a 5 GiB free-disk reserve. The main controls are:

```text
--workers N
--cores-per-worker N
--amplicol-build-jobs N
--generation-time-limit SECONDS
--worker-wall-limit SECONDS
--ram-limit BYTES
--campaign-ram-limit BYTES
--attempt-output-limit BYTES
--minimum-free-disk BYTES
--target-measurement-duration SECONDS
--minimum-samples N
--warmups N
--batch-size N
```

`--cores-per-worker` controls pyAmpliCol generation and runtime construction.
Original AmpliCol compilation has the separate `--amplicol-build-jobs` control,
which defaults to 1 even when pyAmpliCol receives more cores. Keep that serial
default for the maintained legacy checkout: its generator Make target is not
parallel-safe.

`--ram-limit` is a cap for each worker tree—the worker plus every compiler or
external program it starts. The optional
`--campaign-ram-limit` is an aggregate ceiling: the controller conservatively
divides it by the requested worker count and uses the smaller of that share and
the per-worker cap. For example, ten workers with both limits set to decimal
30 GB receive at most 3 GB each, so they cannot collectively claim ten times
30 GB:

```console
./steer_performance_campaign.py run \
  --workers 10 --ram-limit 30000000000 \
  --campaign-ram-limit 30000000000 \
  --table matrix --multiplicity 1 2 3
```

The controller measures the worker and all of its descendants as one memory
tree. It guards resident RAM (RSS), and on macOS also checks the operating
system's physical-footprint measure and uses whichever is larger. This RAM
guard does **not** limit bytes written to disk: a runaway external-program log
can fill a volume while the process remains well below its RAM ceiling.
`--attempt-output-limit` therefore stops a worker when any watched attempt log
or output file crosses its per-file limit; the default is 64 MiB (67,108,864
bytes). `--minimum-free-disk` stops it when free space on the campaign-artifact
volume falls below the reserve; the default is 5 GiB (5,368,709,120 bytes).
Both cases are recorded as explicit errors rather than being mistaken for
physics or timing results.

Increase parallelism only when the machine has enough RAM for that many
independent worker trees. `--fail-fast` changes validation scheduling into
multiplicity waves and stops after the first final non-success, including a
required mismatch, error, timeout, or supervisor-confirmed resource cap. A
legacy AmpliCol disagreement remains a diagnostic rather than a stopping
condition because original AmpliCol is not a correctness authority.

## What one campaign cell actually does

For an ordinary successful cell, the controller:

1. resolves the catalogue process, model, colour treatment, workload, and
   execution mode;
2. prepares reusable model data when needed;
3. generates the process output and records generation time;
4. evaluates a shared deterministic phase-space point;
5. checks the resolved sum and the required cross-mode or MadGraph agreement;
6. profiles the accepted workload in repeated timing chunks; and
7. seals the attempt (marks the record complete and no longer editable) and
   makes it the cell's current result in one replacement.

Independent cells can run in parallel. Dependencies are explicit: for example,
the matching recurrence result checks compiled and eager results, while
MadGraph standalone supplies the independent full-colour reference for UFO-SM
results. For LC OTF cells through `n <= 4`, the matching recurrence cell checks
every resolved helicity/flow component, not only their final sum; no compiled
output is generated merely to perform that check.

The common deterministic point matters. A timing ratio is published only when
the candidate and denominator are explicitly linked to the same phase-space
point and the required validation has passed. This prevents a visually
plausible ratio from accidentally joining unrelated measurements.

## Watch, stop, and resume without losing completed work

In the interactive dashboard, `Ctrl-C` stops the campaign safely. `Esc` first
closes an open command drawer; otherwise it performs the same orderly stop. In
headless mode, use `Ctrl-C`. The controller stops dispatch, terminates each
supervised worker and any child programs with its configured grace period,
preserves completed current results and compact interruption evidence, marks
the campaign as no longer live, writes the summary IDs, and exits with the
conventional interrupted status (code 130).

There is no supported “hold” or in-memory pause command. To pause a long scan,
stop it orderly and later run the same command again. Compatible successful
results from the same source revision (code version) are reused automatically;
a cell with no compatible successful current is attempted again. If an older
successful current survived the interrupted attempt, the ordinary rerun reuses
that result; add `--force-refresh` only when you intentionally require a fresh
attempt. Historical results from another source revision participate only when
you explicitly pass `--continue-across-revisions`. Use that option only after
checking that the intervening edits cannot change the generated physics,
validation, or measured runtime path—for example, a report-layout-only change.
Each reused result still records its original revision, so the PDF's provenance
remains auditable rather than pretending that every cell came from one commit.
Resume and cross-revision reuse follow the sealed `current.json` and result
records; they do not depend on retaining a compiler or external-program build
workspace.

With dashboard bindings installed, another terminal can take a read-only
snapshot of the newest live campaign:

```console
./steer_performance_campaign.py \
  dashboard-snapshot --live --width 160 --height 48
```

This reads a small status record; it cannot pause, resume, or stop the campaign.
Use `--instance ID_OR_PREFIX` when several campaign invocations are active.
Without the optional dashboard bindings, this subcommand explains how to
install them rather than attaching to a wheel-owned headless run.

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

The controller owns `campaign_artifacts/`. Here “artifact” means retained
campaign evidence, not only a generated pyAmpliCol process output. A cell
normally retains a tiny pointer to its current result and one or more uniquely
named attempt directories. Depending on the cell, an attempt can contain:

- a manifest (an inventory identifying the completed attempt);
- `result.json` and `worker-result.json`;
- a bounded `worker.log`, streamed progress events, and a phase timeline;
- resource measurements and termination diagnostics; and
- when retained, an `artifact/` payload containing the generated process output
  or an external-reference workspace used for the measurement.

Prepared models, reusable model data (“model caches”), live status data, and a
temporary directory used while building the PDF (“publication staging”) also
stay below the campaign directory. Treat their exact subdirectory names as
controller internals; use `inspect`, the summary ID files, and the current
result JSON rather than editing pointers by hand.

The default is **compact retention**. The controller keeps the result,
provenance, commands, progress and phase records, resource diagnostics, and
bounded final log tails. It keeps a generated pyAmpliCol process output when a
current result still needs it. For original AmpliCol it keeps the small
selected-flow library needed for replay and self-contained structural evidence,
but removes the disposable checkout/build workspace; disposable MadGraph and
other build workspaces are likewise removed after their measurements are
sealed. Heavy payloads from obsolete, failed, or cancelled attempts are also
removed. The old `--cleanup-artifacts` spelling is still accepted, but now
names this default policy rather than enabling a more aggressive one.

When diagnosing an external generator, opt in to complete workspaces for that
run with:

```console
./steer_performance_campaign.py run \
  --retain-workspaces \
  --workers 1 --table matrix --process-id 1 --multiplicity 1 \
  --color-approximation lc --generation-mode non-union-flow \
  --generation-engine recurrence --model built_in \
  --no-dependencies-added --no-dashboard
```

`--retain-workspaces` can consume disk quickly and is intended for a bounded
debugging session. It does not disable the attempt-output or minimum-free-disk
guards. Without it, a failed or fail-fast-cancelled attempt still retains the
compact diagnostics needed to understand and replay the cell, but not an
unbounded partial workspace.

Temporary PDF-building directories and “live” markers are not historical
evidence. They are removed when their operation ends, whether it succeeds or
fails. Do not prune the campaign with broad shell cleanup commands.

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

The controller first takes a stable snapshot—a read-only list of the complete
current results that belong in this version of the report. It verifies their
recorded origin and validation links, shows coloured progress, renders the JSON
and table TeX in a fresh temporary directory, and runs LaTeX there. Only after
all of that succeeds does it replace the published files and `pyAmpliCol.pdf`
as one unit. A failed render therefore leaves the previous PDF in place.

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

For a human-readable view of current coverage and status, including OTF, use:

```console
./steer_performance_campaign.py inspect \
  --color-approximation lc \
  --generation-engine recurrence compiled eager on-the-fly
```

Add `--json` for stable, uncoloured output suitable for scripts. Exact
cell IDs in `inspect`, the current JSON, and `campaign_summary_ids/` connect a
compact table status back to its retained attempt evidence. The PDF does not
turn every timing into a clickable evidence link; the result record stores the
source/settings (“provenance”), shared point, and validation-reference link.

### Wall, evaluator, and recurrence clocks

The report keeps several clocks separate instead of renaming whichever number
is available:

- **Runtime** is the broad physics task of evaluating already generated output
  at phase-space points. The **evaluator** is the loaded numerical calculator
  that performs it.
- **Wall time** is elapsed real time observed around the complete timed call.
  It is available across implementations and is therefore the primary number
  used to choose the fastest validated mode.
- **Evaluator total** is a separately recorded accumulated clock inside the
  complete warmed evaluator. Comparing it with wall time helps reveal time
  spent in wrappers or in choosing and calling the numerical routine; it is
  not copied from wall time.
- **Recurrence core** is the still narrower time spent replaying recurrence
  schedules. It is useful for diagnosing recurrence itself, but it is not a
  complete evaluator clock and is never relabelled or used as another mode's
  denominator.

The muted bracketed `xS` in a Best-vs-AmpliCol runtime cell first compares
compatible internal clocks that each implementation explicitly assigns to
numerical execution, excluding its documented surrounding work. If those are
unavailable, the table may use its explicitly documented evaluator-total
versus legacy-direct fallback. Either way `xS` is diagnostic. The coloured
outer `xW` is always the wall-time ratio and remains the performance
conclusion.

`JIT` in row labels means “just-in-time”: numerical machine code is prepared
automatically for the current environment. `O2` and `O3` name compiler
optimization levels; they do not denote perturbative orders.

### The Z-gluon ladder also includes OTF

The built-in-SM and UFO-SM Z tables follow

```text
d d~ -> Z + (n-1)g
```

through increasing final-state multiplicity. For each `n`, their setup rows now
include compiled variants, eager, recurrence, **and on-the-fly JIT O2**, beside
the original-AmpliCol reference. The two column groups remain the same physical
workloads used throughout the report: selected flow with a helicity sum, and
all flows at one helicity.

An OTF generation cell in these detailed Z tables is printed as:

```text
[G] G+W
```

Both numbers are absolute seconds. `G` is generation of the compact process
recipe; `G+W` adds the first cold evaluation of the complete campaign batch for
that workload. The muted brackets separate generation alone without hiding the
construction cost moved to first use. This `W` is not the public
single-point `Runtime.warm_up(...)` call: the campaign deliberately measures
its complete profiling batch. The adjacent `wall` column is the ordinary warm
wall time per point. The `eval` column is shown only when the mode supplied a
separately authenticated warmed evaluator-total clock; `not exposed` does not
mean that wall timing failed.

## Read one real “Best vs AmpliCol” cell

The following is the process-ID 1, `n = 2` cell from the current authoritative
campaign rendered for this release. It is a worked notation example, **not** a
performance promise for another machine, compiler, source revision, or model
preparation:

```text
d d~ -> Z + (n-1)g  =  d d~ -> Z g
```

The LC table contains two aligned workloads in the fixed order
**selected flow, helicity sum | all flows, one helicity**:

| Table row | Original AmpliCol | Best validated pyAmpliCol mode |
| --- | --- | --- |
| generation `[s]` | `3.66 | 0.000217` | `(o) ([x1.52] x1.52) | [5.57] 5.57 (o)` |
| runtime `[microseconds/point]` | `0.209 | 0.172` | `([x2.46] x2.51) | ([x0.888] x1.04)` |

Here is every visible entry:

- `3.66` seconds is original AmpliCol's selected-flow process-library
  generation time.
- `0.000217` seconds is original AmpliCol's direct all-flow setup time. This is
  a different setup boundary from pyAmpliCol process generation.
- `(o)` says that **on-the-fly JIT O2** had the smallest validated wall time
  for that workload. The code is a label, not a ratio. The four possible codes
  are `(r)`, `(c)`, `(e)`, and `(o)` for recurrence JIT O2, compiled JIT O3,
  eager-DAG JIT O2, and on-the-fly JIT O2.
- In `([x1.52] x1.52)`, the muted bracketed multiplier uses compact OTF
  process-output generation `G` alone. The coloured outer multiplier uses
  `G+W`, where `W` is construction and first evaluation of the selected family
  on the complete campaign batch. Both are divided by the matching 3.66 s
  AmpliCol library-generation time. They round to the same value here because
  the stored `G` was about 5.565 s and `W` about 0.00362 s.
- `[5.57] 5.57 (o)` is the all-flow OTF generation pair in absolute seconds:
  muted `[G]` followed by `G+W`. It is intentionally not divided by
  `0.000217`, because AmpliCol direct setup and pyAmpliCol process generation
  are different operations.
- `0.209 | 0.172` are original AmpliCol wall times in microseconds per point
  for the selected-flow and all-flow workloads, respectively.
- In `([x2.46] x2.51)`, the muted bracketed value compares the OTF
  execution-attribution clock with AmpliCol's compatible direct-execution
  clock. The coloured outer value independently compares wall time: about
  0.526 microseconds per point divided by 0.209. The outer ratio is the primary
  figure used to choose a winner; generation time is not.
- Likewise, `([x0.888] x1.04)` contains the supplementary execution ratio and
  the primary wall ratio for all flows at one helicity. The OTF wall time was
  about 0.178 microseconds per point, versus about 0.172 for AmpliCol.

Muted supplementary ratios help diagnose where time is spent, but they never
replace the outer wall ratio. In OTF generation notation, `W` excludes process
output loading and the conventional benchmark warm-up calls. It is also not
the public `Runtime.warm_up(...)` contract, which accepts exactly one
double-precision point for one selected family; the campaign times the first
complete batch required by its profiling protocol.

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

A compact token such as `N out.[5d35ae]` is a shortened final status, not a
number or speed ratio. Use `inspect` or the relevant file under
`campaign_summary_ids/` to recover the exact cell ID and outcome. `not exposed`
has a different meaning: the wall measurement succeeded, but that mode did not
provide the separately recorded internal clock requested by the detailed
table. It is not a missing or failed wall measurement.

Unverified, capped, unsupported, and validation-failed candidates do not enter
“best” selection or performance summaries. A successful but unlinked candidate
can still be selected by its absolute wall time; it is displayed absolutely and
is excluded from timing ratios, mode-mix counts, and ratio summaries.

## Read the summary rows

Below each multiplicity block, the **mode mix** counts selected-flow wall-time
winners that also have the valid AmpliCol link needed for the generation
summary. The current authoritative `n = 2` LC block shows:

```text
r:0 | c:2 | e:0 | o:4
```

Among the six eligible process families, no winner was recurrence, two were
compiled, none was eager, and four were OTF. A zero is meaningful—it says that
the mode was measured but did not win this particular group. The order is
always `r | c | e | o`; it matches the four mode labels in the cells.

Every five-number summary is ordered as:

```text
minimum | maximum | median | arithmetic mean | weighted mean
```

The weighted mean is framed in the PDF. It is the ratio of summed candidate
time to summed baseline time, not the average of the individual ratios:

```text
weighted mean = sum(candidate times) / sum(baseline times)
```

For `n = 2`, the same rendered campaign snapshot displays:

| Summary row | `min | max | median | mean | weighted` |
| --- | --- |
| generation, selected-flow workload | `x0.801 | x1.52 | x1.34 | x1.24 | x1.23` |
| runtime, selected flow and helicity sum | `x2.00 | x4.12 | x2.46 | x2.89 | x2.65` |
| runtime, all flows and one helicity | `x0.754 | x1.43 | x1.00 | x1.05 | x1.05` |

The generation summary uses only the comparable selected-flow **primary**
generation ratios; for an OTF winner this is `G+W`, not the muted `G`-only
ratio. Absolute all-flow generation is omitted. Runtime summaries use only the
primary wall ratios, not the muted supplementary ratios, and keep the two LC
workloads on separate lines. Structurally inapplicable, resource-limited,
unsupported, unverified, unlinked, and validation-failed cells are excluded.
This Best-vs summary follows the selected winner's primary token; a fixed-engine
OTF table instead keeps its generation summary as the diagnostic `G`-only
quantity while displaying both `G` and `G+W` in each row.

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
4. Confirm worker count, per-worker cores, wall-time limit, per-worker RAM cap,
   any aggregate campaign RAM cap, attempt-output cap, and free-disk reserve
   before broadening the scan.
5. Stop with `Ctrl-C` or dashboard `Esc`; resume by repeating the command.
6. Use `inspect` and `campaign_summary_ids/` to understand non-successes before
   forcing a refresh.
7. Use compact retention by default. Add `--retain-workspaces` only for a
   deliberate, disk-bounded debugging run.
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
[Profiling and Benchmarking](profiling-and-benchmarking.md#what-is-timed).
