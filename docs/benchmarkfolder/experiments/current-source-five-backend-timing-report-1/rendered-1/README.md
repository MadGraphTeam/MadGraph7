# Four-process, five-backend shared-host timing appendix

This frozen appendix reports actual full-amplitude wall timing for twenty
process/backend lanes: QCD `u u~ > z`, QCD `g g > t t~`, EW
`e+ e- > mu+ mu-`, and QCD `u u~ > e+ e- mu+ mu-`. Each lane uses the
same full saved 100-point cohort for that process and ten position-balanced
timed cycles. All twenty lanes pass the retained `3e-6` numerical regression
criterion and the separate ordinary `1e-8` target. The coefficient gates replay
the final 100-point traversal of learning, both calibrations and each timed
batch. Histograms count every output call in the ten timed cycles. They do not
observe all internal attempts or hardware floating-point exception flags.

`timing-summary.csv` has exactly twenty reusable rows. Its medians, unscaled
MADs and within-cycle ratios are the controller analyzer's unchanged statistics,
with seconds converted to milliseconds for the first two columns. JSON retains
the original statistics, every batch, READY learning interval, counts and
provenance. Ratios are medians of ten backend/NinjaRS8 ratios, not ratios of
medians. No confidence interval or final performance acceptance is claimed.

The shared host is an AMD EPYC 9754 with GNU Fortran 15.2.0; workers use
affinity CPU12 and a one-thread environment, without CPU reservation. The
target batch duration is two seconds. Actual full-cohort batch durations are
retained; integral traversal counts can exceed that target. READY time includes
the first amplitude and full100 learning, excluding SETPARA/process startup.
The on-shell Z cohort varies boosts at sqrt(s)=91.188GeV, rather than 1TeV.
The other three cohorts use 1TeV. No points are subsetted for timing.

The new four-lepton process was generated through the MG CLI and built from
source. Its compiled source text is reparsed by the portable verifier: maximum
loop denominator count4, rank3, twelve groups (two bubbles, eight triangles,
two boxes). Four final particles do not imply six propagators in a loop.
The original ZZgg process retains its full numerical coverage and precise
local timing TODO. Fresh preparation includes mass setup and invokes no
amplitudes before the recorded timing workers; the postterminal packaging
helper's zero-native count does not describe every earlier executable startup.

## Exact input selections

The first-three timing packet is
`experiments/current-source-five-backend-timing-1/index.json`, SHA256
`3dc2c2b46295fce266706fd05b9444ee7408e41e3164daff77054189e2d79966`.
The separately generated four-lepton packet is
`experiments/current-source-four-lepton-timing-1/index.json`, SHA256
`2373175c50a62f0c5e0664c2337b438ae147ae31b1f4ff958ece374f041a2e89`.
Both immutable packets include their exact pure replay source. The analyzer is
`30721543c6184ca9a7002b155b6f70bee567663ae15ba1b85c136b5007b51619`.
`scientific-selection.json` retains their original seal filenames and also
selects the separate checked-CMS20 note's evidence index.

All four cases have registered scientific revision
`0fcd2cd873d61d134837e1d9e87afb13a47824fd` and exporter file inventory
`afb6e0bea64ea0932751ff4869c206bcea9477d40a39a54bb99d0f06d85b75a6`.
Actual export Git revisions are separate receipt fields: the Z export is
`85c6211b14…`, ggtt/EW exports are `6c77fe0604…`, and the fresh four-lepton
export is `b443c529633071b997bd21aa3654ef8b10b9e013`. Complete values remain in
JSON and source receipts. An inventory digest is never relabelled as a commit.
The envelope distinguishes original export numerical-card hashes from the
actual per-backend timing cards and records compiler flags, physical cards,
saved points, executable hashes and mapped runtime identities.

The optional CMS20 paragraph is a separate private, cache-disabled fixture
result bound to `normal-checked-cms20-qualification-1/evidence-index.json`,
SHA256 `f83d036d054adc58a3202c8614b7b9dcec549309b30be30390887f838822f3fc`.
It retains failed DP groups before successful genuine-QP rescue. It does not
qualify generic production, enabled caches, independent amplitude truth or
performance. Its reviewed narrow note bytes are preserved unchanged here.

## Reproduction

Run commands from the repository root with the pinned development environment.
Use fresh output directories. Report commands never launch a native amplitude,
compiler, runtime loader or network request. Python uses only retained portable
parsers and the analyzer; Matplotlib/Typst/Poppler are document tools.

```sh
source tools/tirust_env.sh
timing_report=docs/benchmarkfolder/experiments/current-source-five-backend-timing-report-1/rendered-1
python3 -B "$timing_report/prepare_timing_report.py" --root docs/benchmarkfolder \
  --output .tirust-build/reproduced-shared-host-timing-data
```

Compare the generated `timing-report-data.json` and `timing-summary.csv` with
their immutable copies here. To regenerate the plot and Typst tail in a fresh
directory, copy the generated four data files and the retained `timings.typ`,
`style.typ` and renderer there, then run:

```sh
export MPLCONFIGDIR="$PWD/.tirust-build/reproduced-timing-mplconfig"
python3 -B .tirust-build/reproduced-timing-tail/render_timing_comparison.py \
  --data .tirust-build/reproduced-timing-tail/timing-report-data.json \
  --output .tirust-build/reproduced-timing-tail/timing-comparison
typst compile .tirust-build/reproduced-timing-tail/timings.typ \
  .tirust-build/reproduced-timing-tail/timing-supplement.pdf \
  --input include-plots=true --creation-timestamp 1794355200
```

The canonical timing append owner authenticates the prior39-page archive and
this sealed tail. Its exact reproduction/verification command is supplied
with `build_timing_report.py` in the parent benchmark README. It preserves the
old39 pages using paired144dpi RGB pixels, per-page text, geometry and fonts;
it does not claim PDF object-graph or new annotation/link equivalence. Existing
NinjaRS PDF/source and the main historical Typst source remain unchanged.

Final precise performance remains Hua-Sheng's local TODO in
`campaigns/entry-latch-cache-guard-1/timing-handoff-1/LOCAL_TIMING_TODO.md`:
fresh local numerical qualification, the registered starting-revision baseline
and TIRust-first rescue-chain comparison, existing-backend regressions, and
the per-main-process upper95% paired runtime-ratio bound≤1.05. These twenty
descriptive shared-host lanes cannot satisfy that separate acceptance gate.

The exact style copy and `style-provenance.json` attribute pyAmpliCol revision
`7266c1600f6ec4e39be39986e6503a515c380c52`. No external timing values are used.
