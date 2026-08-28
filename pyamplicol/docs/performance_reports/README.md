---
title: "Performance Reports"
nav_order: 2
parent: "Profiling and Benchmarking"
---
<!-- SPDX-License-Identifier: 0BSD -->
# Performance Reports

See [FullColor FFT Profiling](../user/fullcolor-fft-profiling.md) for the
source-checkout driver, cluster resource limits, incremental top-ups,
targeted overwrite, and render-at-any-time workflow that produces the FFT
snapshots below.

The repository publishes four selected rendered snapshots from manual
measurement campaigns:

- [MacBook M3 report](https://github.com/mg5amcnlo/pyamplicol/blob/main/docs/performance_reports/macbook_M3_pyAmpliCol.pdf)
- [AMD EPYC report](https://github.com/mg5amcnlo/pyamplicol/blob/main/docs/performance_reports/EPYC_pyAmpliCol.pdf)
- [FullColor FFT selected-helicity snapshot](https://github.com/mg5amcnlo/pyamplicol/blob/main/docs/performance_reports/summary_plots_final.pdf)
- [FullColor FFT helicity-sum snapshot](https://github.com/mg5amcnlo/pyamplicol/blob/main/docs/performance_reports/summary_plots_final_helicity_sum.pdf)

The first two PDFs are general host reports. The FullColor FFT selected-
helicity snapshot times one shared known-nonzero helicity from helicity-general
artifacts; the helicity-sum snapshot times the complete physical helicity sum.
Both snapshots include genuine same-workload MadGraph standalone series. The
fixed series uses the generated helicity-general standalone and selects the
shared helicity through `MATRIX(P,NHEL,IC)`; it currently measures pure gluons
through `n=5` and `d d~ > d d~ + gluons` through `n=6`. The independent summed
series measures both families at `n=2..5` through MadGraph's generated
`SMATRIX(P,ANS)` entry point with `USERHEL=-1`, native IDEN normalization, and
warmed `GOODHEL` pruning. It does not reuse the fixed-helicity timings.
The FFT PDFs are rolling development snapshots, not claims that every
final-state multiplicity through `n=9` has completed.

Their frontiers are curve-specific. Both fixed-helicity and helicity-sum OTF
curves are requested only through final-state `n=6`; beyond that the publication
protocol retains recurrence, AmpliCol, and Reference FFT where applicable.
Pure-gluon OTF FFT reached the 3,600 s first-use runtime cap at `n=6` and retains
its measured `n=5` point. The helicity-sum snapshot extends
`d d~ > d d~ + gluons` through `n=6`, where every requested curve measured.
OTF direct and FFT setup took 1,725.8 s and 1,607.9 s; their warmed runtimes were
5.321 and 3.759 ms/point (1.416x faster with FFT), at 1.17 and 1.22 GiB peak
RSS. These isolated measurements used a 10-hour time limit and 30 GiB
process-tree guard and retain their own per-cell provenance.

The FFT plots use a deliberately method-specific *setup time*. pyAmpliCol
includes artifact generation, a fresh load, the first requested evaluation,
and OTF family warm-up where applicable. Reference FFT includes its build,
initialization, and first pass. AmpliCol includes process/color-object
generation for the fixed workload, or process/raw-library generation and build
plus the immutable snapshot for the summed workload. Warmed runtime remains a
separate measurement after those boundaries.

These PDFs are measurement-host results, not release-CI results. Raw JSON,
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
produce the same report format as the two general host PDFs. Re-running
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
