# Paper-analogue tables and stability figures

This is a consumer design and software contract, not a numerical result. No final timing or DP-only self-consistency dataset is available at this checkpoint. The historical top-level `benchmark.typ` and `benchmark.pdf` remain unchanged. Methods and attribution are in [METHODS.md](METHODS.md); no code from the optional external `madninja` archive is used.

Three distinct observations remain separate:

1. The acceptance campaign measures warmed, full-helicity MadLoop evaluation with normal stability checks and actual rescue. Its literal original Ninja denominator, coefficient agreement, seven-or-more paired repetitions and 1.05 confidence-bound gate are unchanged.
2. The paper-analogue timing component measures one common physical helicity, a color sum with the generated normalization, full poles, normal rescue, 100 fixed points and eight balanced repetitions for CutTools, Ninja, COLLIER and NinjaRS. This is a separate diagnostic table. A rejected backend cell retains its elapsed cost and exact rejection/DP/QP counts as `failed_physics`; calibration is never a repetition or a successful result.
3. The dedicated native-DP component disables rescue. It compares forward, reversed-loop-direction and exactly rotated observations through the actual producer's per-dispatch rejection instrumentation. Its relative spread is a self-consistency estimator, not an observed error against an analytic answer. Normal-controller accuracy estimates, measured inter-tool discrepancies and high-precision convergence belong to separate report sections.

## Concrete inputs

`tools/ninjars_paper_report_v3.py` calls the two concrete producer authorities; it does not trust supplied summary pass flags:

- `ninjars_paper_timing_v3.report_snapshot(result_path, expected=<runtime identity>, inputs=<mapping>)` reparses actual timing/calibration raw files and returns `ninjars-paper-timing-summary-v1`. Each case's `backend` cells retain medians, 95% intervals, dispersion, actual status counts, DP/QP/failed totals, eight measurement bindings and a separate calibration binding. `numerator_dp` and `remaining` stay pending until their separate diagnostic authority exists.
- `ninjars_dp_stability.report_snapshot(manifest_or_summary, expected={'cohort': <exact cohort>}, inputs=<add-capable mapping>)` returns `ninjars-dp-stability-summary-v1`, with `stream='forced_dp_self_consistency'`, `rescue_enabled=false`, declared thresholds and per-case/tool/point `combined`, `direction` and `lorentz` estimators. Missing tools remain missing; attempted missing/failed points stay classified in the fixed planned denominator.

The explicit cohort JSON has the producer's full object, including `numerical_revision`, `generation_revision`, `generator_identity_sha256`, `runtime_identity_sha256` and `archive_sha256`. Any additional pins/fields remain part of the exact DP authority object. The consumer retains both selected input files, actual reader source hashes and all files checked transitively by each reader. It emits `qualification_eligible=false`; these diagnostic components cannot self-certify installation, precision rescue, core correctness, performance, profiling or final publication.

Raw `observations_complete` is distinct from physical qualification. The high-level figure/CLI path requires the DP summary's `physical_qualified=true` for every nonsoftware render, so complete-looking raw coverage cannot substitute for the owner's actual runtime/process-parity authority. A software-fixture exception is explicitly labeled and is not a physics observation.

## Process table definitions

Static process inventory requires the actual output directory, `generation.json`, `compilation.json`, `compiled-sources.json` and, when available, its CLI log. The inventory verifies every read generated file against the retained compilation source manifest. It executes no matrix element, model import or pickle.

- Generation and compilation are their recorded full command wall times, not the log's narrower internal generation timer. Initialization is separate.
- Distinct loop-diagram IDs come from the DP `coef_construction_*.f` comments and are cross-checked against the CLI's contributing-loop count. Identified equivalent copies, including grouped flavors, are reported separately. UV and R2 diagrams do not enter this count.
- `NLOOPS` from `loop_matrix.f` is explicitly the HELAS loop-amplitude count. It is not relabeled as the number of distinct loop diagrams. `NLOOPGROUPS` supplies the reducible-group count.
- Maximum tensor rank is the exact integer inverse of `LOOPMAXCOEFS = binomial(rank+4,4)`, the four-dimensional symmetric-polynomial coefficient count. An unavailable or nonmatching count remains missing.
- `NBORNAMPS` and `NCOMB` are displayed only as their literal generated amplitude/helicity metadata.

The source definitions are MG7's `LoopAmplitude.identify_loop_diagrams` in `madgraph/loop/loop_diagram_generation.py`, optimized export metadata in `madgraph/loop/loop_exporters.py`, `get_max_loop_rank` in `madgraph/loop/loop_helas_objects.py` and coefficient counting in `madgraph/various/q_polynomial.py`. The retained generation source identity binds their measured revision. No table entry substitutes a guessed process count.

The table keeps total single-helicity costs and all backend status information together. A separate numerator-only private build, if measured, must supply its own source/runtime/input authority. `total minus DP coefficient construction` includes stability checks, rescue and other work and must be labeled **remaining full-evaluation work**, never pure reduction. Negative or unresolved differences remain visible. Equation B.1's helicity scaling for Born interference is not applied to loop-induced amplitudes.

## Plot semantics

The adapter generates original Matplotlib SVGs using the established navy/serif palette; vector text is retained. Separate figures can be grouped as QCD, EW, loop-induced and challenging stress processes. The declared threshold grid is augmented with actual positive observed jump locations, without smoothing.

At threshold `t`, the primary curve is `(failed or unbounded points + finite spreads strictly greater than t) / N`. All-zero triples are separately counted with spread zero. Nonfinite values, backend rejection, missing observations, process failures and nonzero-spread zero means contribute at every finite threshold. A pending or unavailable tool produces no observed curve, not a zero curve. The fixed `N`, helicity and sample kind remain in the machine-readable data.

The main axes use logarithmic threshold and probability scales. Zero observed probabilities are omitted from the logarithmic curve, with an open marker at the exact one-sided 95% zero-event bound `1 - 0.05**(1/N)`; this marker is not an observed event fraction. A separate `1/N` reference shows finite-sample resolution. Wilson intervals are supplied in data. For fixed, boosted or targeted samples, these binomial references do not establish population coverage.

The Ninja-ratio panel has a linear probability-ratio ordinate, allowing a genuine zero ratio. If Ninja has no tail events, `0/0` is undefined and positive/zero is unbounded; both remain null gaps with exact numerator/denominator counts, with no epsilon substitution. No independent-sample ratio uncertainty is invented for point-matched tool observations.

An initial 100-point pilot does not reproduce the paper's 100K/10K/1K samples. At 100 independent points, even zero events gives a 2.95% one-sided 95% upper reference. Larger sampling remains separately pending unless actually collected.

## Reproduction commands after real collection

Run inside the repository with its retained toolchain. `EXPECTED_COHORT.json` must be the exact measured producer cohort, and inventory selection maps case IDs to `process_inventory` arguments. No placeholder here is an existing result:

```sh
python -B tools/ninjars_paper_report_v3.py \
  --timing ACTUAL_TIMING_RESULT.json --stability ACTUAL_DP_MANIFEST.json \
  --expected EXPECTED_COHORT.json --inventory ACTUAL_INVENTORY_SELECTION.json \
  --figures NEW_OUTPUT_DIRECTORY --output NEW_REPORT_DATA.json
```

`--figures` and `--output` require new paths to preserve prior analyses. Direct `plot_stability(..., estimator='direction')` or `estimator='lorentz'` produces the corresponding separate contrast. Final integration into the elegant Typst report waits for validated real data; synthetic software plots are labeled and never promoted as physics evidence.
