# Manual MacBook M3 Performance Campaign

## Goal

Build a fresh, manually steerable MacBook M3 performance campaign whose concurrent workers are safe, observable through a polished Ratatui dashboard, resource-bounded, selectively runnable, statistically inspectable, and able to regenerate its PDF without importing prior campaign evidence.

## Implementation Start Protocol

- Before editing, inspect the worktree, branch, remote, hooks, and relationship between local `main` and `origin/main`.
- Never request command escalation or approval. Vet commands for sandbox escape, hook execution, shell interpolation, and unintended writes; use sandbox-safe alternatives. If updating is impossible without escalation, stop and report it.
- Fetch `origin/main` with hooks disabled and fast-forward local `main`. Never force, reset, stash user work, or resolve divergence automatically. Stop on a dirty worktree, divergence, unexpected remote, or blocked network.
- The user subsequently authorized running `just dev-install`; after every requested fast-forward to a newer `origin/main`, rebuild the contributor environment before continuing. The campaign script itself must never install or rebuild anything.
- Immediately after updating `main`, write this revised plan verbatim to `docs/performance_reports/MACBOOK_M3_MANUAL_CAMPAIGN_PLAN.md`. Before any implementation edit, create an agent goal: “Implement and verify every requirement in `docs/performance_reports/MACBOOK_M3_MANUAL_CAMPAIGN_PLAN.md` exactly as written.”

## Campaign and Command-Line Interface

- Create `docs/performance_reports/macbook_M3_manual` from the canonical blank report template. Start with unavailable measurements and no copied PDF, result, digest, measurement, or artifact from `macbook_M3`.
- Add executable `steer_performance_campaign.py`. It re-executes directly with the repository’s `.venv/bin/python`, without a shell, and never performs Git, network, installation, or rebuild operations.
- Provide `run`, `inspect`, `refresh-pdf`, and deterministic `dashboard-snapshot` subcommands.
- Make `--help` exhaustive and practical:
  - explain every subcommand, selector, default, unit, resource cap, profiling option, reuse rule, and keyboard control;
  - list canonical values and accepted aliases;
  - explain wildcard, repetition, union/intersection, generation-layout, and force-refresh semantics;
  - include copy-paste examples for one cell, several multiplicities/models, a complete table, all entries, dry runs, inspection, forced refresh, multicore execution, and PDF rebuilding;
  - add detailed help to every subcommand, not only the top-level parser;
  - show defaults directly in option descriptions and use readable grouped sections with a short steering guide and common recipes.
- Use shared repeatable selectors where applicable: `--table`, `--process-id`, `--multiplicity`, `--color-approximation`, `--generation-mode`, `--generation-engine`, `--model`, `--variant`, and `--cell-id`.
- Accept multiple values after one option or repeated options. Omitted dimensions and quoted `*`/`all` mean all. Combine values by union within a dimension and intersection across dimensions.
- Normalize case, hyphens, and underscores and support friendly aliases such as `z_table`, `built_in`, `sm_ufo`, `nlc`, `union-flow`, and `non-union-flow`.
- Define layouts clearly:
  - `non-union-flow`: one selected flow with helicity sum.
  - `union-flow`: all-flow union with one helicity.
  - `contracted`: contracted workload required by NLC/full-color entries.
- Validate selectors against the catalog and show a concise colored allowed-values table with suggestions for invalid or empty selections. `run --dry-run` displays the direct selection and required dependency work.

## Lightweight Reuse and Provenance

- Keep authentication, validation, and artifact reuse deliberately minimal. Do not introduce MD5/SHA content comparisons, repeated directory scans, elaborate integrity ceremonies, or new HMAC/digest systems.
- At startup, check only what is required to run safely:
  - expected profile/schema version;
  - recorded source revision string;
  - readable current-result metadata;
  - required artifact files exist;
  - the result is complete and matches the selected cell.
- Trust an existing current result when those inexpensive metadata checks pass. Do not re-hash large files or recursively revalidate artifact history.
- Record provenance once when publishing a result: source revision, selected cell identity, engine/model/settings, effective caps, and result status. Use atomic current pointers and existing framework primitives only where necessary.
- Automatically update the profile’s source-revision marker under a small cross-process lock. Require no SHA entry or separate authentication command from the user.
- When the source revision changes, treat earlier currents as non-reusable without expensive verification. Retain them as history, but do not perform content-digest comparisons.
- Reuse valid same-source results by default. `--force-refresh` creates a new regeneration and remeasurement attempt without deleting history. Replace current only with a completed success or a resource-cap terminal result.

## Execution and Concurrency

- Drive pyAmpliCol generation and profiling explicitly through the public `pyamplicol generate` and `pyamplicol profile` command paths, invoked through their Python parser/handler APIs rather than a shell. Show copy-paste equivalent CLI commands for every selected runnable cell in dry runs, worker details, and result provenance so users can reproduce individual entries directly. Label report-only validation, legacy AmpliCol work, and any specialized timing path that cannot be expressed by a public subcommand.
- Reuse the existing catalog, dependency scheduler, atomic publication, and per-cell cross-process locks. Multiple instances may safely run overlapping or disjoint selections.
- Add lightweight heartbeat leases beneath the profile’s ignored coordination directory so dashboards can observe workers from concurrent instances. Per-cell locks remain authoritative; leases are informational.
- Default to one worker and one core per worker. Expose workers, cores per worker, target measurement duration, minimum samples, warmups, batch size, resource-sampling interval, termination grace, oversubscription control, and relevant engine parallelism.
- Default each worker to a decimal 30 GB process-tree RAM ceiling and a one-hour generation-only limit. Include preparation and legacy generation in that limit; profiling may continue afterward.
- Refuse real measurements unless the installed runtime has the exact clean committed source revision selected by the controller. Never weaken recurrence runtime identity checks, and never mix pre- and post-revision evidence in one claimed row.
- Keep outer wall time, accumulated evaluator-total time, and recurrence core/execution attribution as independent clocks in result metadata and visualizations. Always show wall and evaluator-total for completed pyAmpliCol cells, show recurrence core separately for recurrence cells, retain enough significant digits to distinguish nearby values, and never fabricate or derive one clock from another. Mark unavailable legacy or narrow attribution as not exposed.
- Prewarm original AmpliCol before generation timing, keep its runtime profiling outside the generation-only watchdog, and retain the upstream scale-independent nonzero selector behavior for tiny physical amplitudes.
- Treat ASM/C++ Z-table entries above final-state multiplicity six as static N/A with a stable reason and zero attempt directories.
- Allow final-state multiplicity-eight all-flow recurrence entries to run under the ordinary worker caps. The upstream bounded spooled/compressed numerical-current evidence path removes the former manual 1 GiB-envelope policy block, so these entries are not static N/A.
- Offer a separate optional total worker wall-time limit, disabled by default.
- Record the effective resource caps and profiling settings directly in result metadata.
- Publish generation/RAM cap hits as terminal results (`>1h` or `>30GB`) so they count as addressed. Preserve failed attempts and the preceding usable current.
- Extend generation and all profiling backends with a small typed progress interface. Process supervision reports phase, step, PID tree, elapsed/CPU time, and current/peak RAM. Workers emit compact atomic state snapshots and append-only progress events for the dashboard.
- Handle `Ctrl-C` cleanly: stop dispatching new cells, terminate every active worker process tree using the configured grace period, discard this invocation's cancelled/incomplete attempt directories and private legacy workspaces, preserve prior valid currents and completed attempts, remove this instance's leases, restore the terminal, print a concise interruption summary, and exit with status 130.

## Ratatui Dashboard and Colored Output

- Use the Python `ratatui` compatibility module, pinned through the developer-install dependency path. The campaign script must never install it. Use Ratatui’s headless frame and styled-cell APIs for deterministic capture tests. [Ratatui package documentation](https://pypi.org/project/ratatui/)
- Render a responsive dashboard with an overview table, selectable worker table, selected-worker detail panel, progress gauges, resource information, recent events, and concise log tail.
- Support arrow-key worker selection, scrolling, help, and reliable terminal restoration after completion, errors, or interrupts.
- Keep running/preparing/queued workers above attention and completed rows, pan the worker viewport with arrow-key selection, use `d` to show/hide completed and successfully recycled rows, and use `e` to show/hide failures, caps, and recycled non-success rows. Show the error count in the overview.
- For completed and recycled rows, replace repeated resource-sample chatter with a persisted typed phase table containing only directly measured durations (preparation, generation and its measured substeps, warm-up, calibration, profiling, attribution, and overall observed resources). Show original attempt/source/time/reuse provenance, state explicitly when no work ran in this invocation, and mark unavailable historical timings rather than deriving them.
- Coalesce repeated active resource samples by phase so preparation and post-generation supervision remain visible without flooding the recent-event list.
- Always show:
  - total entries selected;
  - entries already covered and recycled;
  - entries currently worked on;
  - entries completed by this invocation’s workers;
  - entries remaining.
- Separately identify static unavailable entries, resource-capped results, failures, and dependency-only work so totals remain understandable.
- Color human-facing output by default, including ordinary non-dashboard command output. Use cyan for keys, green for success/improvement, red for failure/regression, yellow for incomplete/capped states, and neutral styling for equality.
- Disable colors only through explicit `--no-color` or `NO_COLOR`. JSON output remains uncolored.
- Use clean table layouts throughout the dashboard, help-adjacent value listings, dry runs, errors, and inspection summaries.
- Provide deterministic dashboard captures at configurable width and height for reviewing layout without an interactive terminal.
- Keep deterministic synthetic capture as the `dashboard-snapshot` default, and provide an explicit `--live` capture mode that reads only non-stale informational lease JSON. Allow selecting the newest active invocation or one exact/uniquely prefixed instance ID, show active same-source peer workers without guessing overlapping totals, and never inspect measurements, currents, artifacts, source identity, or Git for a live capture.

## Inspection and PDF Refresh

- `inspect` reads lightweight current-result metadata and accepts the same selectors. It reports status/coverage plus generation and runtime comparisons with matching AmpliCol baselines.
- Define multiplier as `candidate / AmpliCol`, with lower values better. Report best, worst, median, arithmetic mean, and ratio-of-sums weighted mean: `sum(candidate) / sum(AmpliCol)`.
- Show the exact best/worst identities: table, process ID/key, concrete process, multiplicity, color level, layout/workload, engine/backend, model, variant, candidate value, baseline value, and multiplier.
- Exclude AmpliCol itself, missing or capped values, entries without a compatible baseline, and incompatible generation layouts. Report concise exclusion counts without performing additional artifact authentication.
- Provide `--format json` as a machine-readable uncolored alternative.
- `refresh-pdf` reads one stable snapshot of current-result metadata, regenerates every result JSON and TeX table, compiles in a fresh temporary build directory, and atomically installs a newly built `pyAmpliCol.pdf`.
- Treat LaTeX overfull-box diagnostics as non-fatal layout warnings during `refresh-pdf`; genuine compilation errors remain fatal. On success, print the absolute path of the installed PDF.
- Serialize only the report publication/install step. Incomplete worker attempts are naturally absent from current pointers and require no costly validation.
- Keep campaign workspaces relocatable: a copied report directory derives its profile identity, artifact root, coordination root, source marker, report input, and output PDF from the copied directory/workspace metadata rather than a hard-coded `macbook_M3_manual` name, so copies can run as independent campaigns.

## Verification

- Do not launch the broad performance campaign during implementation verification. Exercise only deliberately small, low-multiplicity selections spanning the Z, process-matrix, scalar-contact, and scalar-gravity report sections; use multicore smoke runs where safe, capture live Ratatui frames while they run, and verify recycled/completed/active/remaining accounting before rebuilding the PDF.
- Test all selector aliases, repetition, wildcards, intersections, invalid values, help text, examples, catalog counts, and dependency closure.
- Verify fresh initialization contains no result, artifact, digest, current, or PDF imported from the old profile.
- Exercise overlapping and disjoint concurrent instances, ensuring one current publication per cell and correct lightweight reuse.
- Test generation-only timeout, 30 GB process-tree enforcement, profiling continuation, optional total wall limit, cap-terminal publication, and failed-refresh preservation.
- Verify progress reporting across generation, recurrence, compiled, and eager profiling.
- Capture Ratatui frames at 80×24, 120×36, and 160×48; assert styling, gauges, counters, units, arrow navigation, interruption handling, and terminal cleanup.
- Capture an actual running multicore lease through `dashboard-snapshot --live`, verify exact atomic counters and worker details, peer-row visibility, instance selection, stale/malformed lease rejection, and that the capture path is read-only and lease-only.
- Exercise keyboard interrupts while workers are queued, generating, and profiling; verify process-tree termination, deletion of cancelled/incomplete attempts, lease cleanup, terminal restoration, preservation of prior currents, and exit status 130.
- Validate exact inspection statistics, ratio-of-sums weighting, exclusions, default colors, `--no-color`, `NO_COLOR`, and JSON.
- Test forced PDF rebuilding and atomic installation, including concurrent incomplete attempts and real LaTeX compilation when available.
- Assert that startup validation remains metadata-only and that no hashing of artifact contents or recursive history verification occurs.
- Assert that the campaign never invokes Git, `just`, package managers, network access, shells, or command escalation.
- Run the existing performance-report regression suite and verify executable permissions and local-venv re-execution.
