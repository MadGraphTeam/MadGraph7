---
title: "Performance Reports"
nav_order: 2
parent: "Profiling and Benchmarking"
---
<!-- SPDX-License-Identifier: 0BSD -->
# Performance Reports

See [FullColor FFT Profiling](../user/fullcolor-fft-profiling.md) for the
source-checkout driver, cluster resource limits, incremental top-ups,
targeted overwrite, and render-at-any-time workflow that produces the
FullColor FFT reports, and [Profiling Campaigns](../user/profiling-campaigns.md)
for the general host reports. The one-loop benchmark report is
[loop/](loop/README.md).

The FFT plots use a deliberately method-specific *setup time*. pyAmpliCol
includes artifact generation, a fresh load, the first requested evaluation,
and OTF family warm-up where applicable. Reference FFT includes its build,
initialization, and first pass. AmpliCol includes process/color-object
generation for the fixed workload, or process/raw-library generation and build
plus the immutable snapshot for the summed workload. Warmed runtime remains a
separate measurement after those boundaries.

Rendered reports are measurement-host results, not release-CI results, and
are not kept in the repository. Raw JSON,
generated TeX, build workspaces, attempts, logs, locks, and coordination state
remain untracked. The blank/reset JSON and TeX resources installed by
pyAmpliCol are campaign templates, not published measurements.

Create or reset a self-contained campaign from any installed wheel with:

```console
pyamplicol profiling-campaign copy ./pyamplicol-profiling-campaign --force
./pyamplicol-profiling-campaign/steer_performance_campaign.py run \
  --workers 1 --table matrix --process-id 1 --multiplicity 1 \
  --color-approximation lc --generation-mode non-union-flow \
  --generation-engine recurrence --model built_in \
  --no-dependencies-added --no-dashboard
./pyamplicol-profiling-campaign/steer_performance_campaign.py refresh-pdf
```

The run above is the supported quick installation check: it measures only the
final-state-multiplicity-one `d d~ > Z` recurrence cell. Broader selections
produce the general host report format. Re-running
`refresh-pdf` directly reproduces the PDF from the campaign's current results;
no repository report data is required.

Runtime attempts, artifacts, locks, and coordination state live visibly in
`pyamplicol-profiling-campaign/campaign_artifacts/`. Moving or renaming the
whole campaign directory moves that state with it; campaigns with the same
basename in different parents remain independent. Historical repository-global
`.artifacts` state is intentionally ignored.

`--force` replaces the managed template files and resets only
`campaign_artifacts/`, the managed PDF and summary IDs, measurement lineage,
and known LaTeX byproducts. It preserves unrelated destination files and the
recorded local-AmpliCol checkout. Stop an active campaign before resetting it;
an active controller makes the command fail without removing state.

All pyAmpliCol backends use installed resources. Measuring the optional
original-AmpliCol reference additionally requires
`--original-amplicol PATH_TO_COMPLETE_CHECKOUT`; without that checkout only
that backend is reported unavailable. pyAmpliCol and the supported patched
comparison checkout have no LHAPDF dependency.
