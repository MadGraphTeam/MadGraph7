# Local numerical and precise timing TODO (Hua-Sheng) — campaign `tirust-2b9ba961-1`

Precise performance acceptance is **not established on the shared host**. It is a
TODO on Hua-Sheng's local machine. This file is the agent-ready task for this
campaign.

`tirust-2b9ba961-1` qualifies TIRust `2b9ba961f640a6f0ee3b7f376cd23c5b869f41c1` on TIRust
`main` (`a5d4afb` plus Symbolica community `d0424f9e` and oneloopmaster `07fc754f`), with
the bridge's method choice off (MadGraph `35e3408e7a`). The merged `madloop_ninjars` uses
return-code format 2: TIRust 213 in double, 323 in quadruple precision. It replaces
`tirust-a5d4afbf-1`. **It has one open blocking defect** (stress, below), whose bridge fix
may change TIRust's cost. If a later campaign is registered, time the newest one listed in
[`docs/benchmarkfolder/CURRENT_STATUS.md`](../../../CURRENT_STATUS.md) instead.

## What is already known (shared host, not acceptance)

- **Numerical lanes.** See the [numerical summary](../summaries/numerical-summary.md):
  every completed lane passes `3e-6`.
- **Double-double gate.** See the [DD-gate summary](../summaries/dd-gate-summary.md).
  Every replay is bit-identical, and DD lanes agree with 256-bit evaluation.
- **Provenance.** The [per-point provenance](../summaries/provenance-summary.md)
  classifies each point as binary64-only, DD, MPFR, genuine QP or failed.
- **Stress.** See the [stress classification](../summaries/stress-summary.md). The
  method-choice defect of `tirust-a5d4afbf-1` (CMS `forward-0.001`) is fixed. New and
  blocking: at the `e+ e- > t t~ a` threshold TIRust accepts a DP value 1.1% off with a
  claimed accuracy of 6.5e-9. The binary64 kinematics fix that point only to about 1%,
  and the bridge's tiers hide this from MadLoop's stability test. The 9|6|7|1 chain also
  returns a wrong CutTools value at two `OSThres`-projected points (outside TIRust).
- **Cost of method choice off.** The TIRust session measured +5% (stock `check profile`)
  to +22% (default card) TIRust time per `u u~ > z g g g` point.
- **Shared-host profile.** Not rerun on the final pins. The rehearsal's stock O3
  `u u~ > z g g g` profile measured TIRust/NinjaRS 1.177.
- **Helicity filters differ between the arms.** Since `madloop_ninjars`
  (`9a9b28d62e`), the candidate MadLoop learns and applies helicity filters with
  `LoopPoleMode = 0`, the campaign's setting. Baseline `9228fc58` bypasses them
  there (`BYPASS_FILTERS = … LoopPoleMode.GE.0`). The candidate therefore skips
  zero and equivalent helicities that the baseline computes, a MadLoop-side
  speed-up unrelated to TIRust. Learning evaluates each helicity separately on the
  first `CheckCycle` points; warmed pairs exclude it. Read the TIRust effect
  against the candidate's explicit-backend controls (same MadLoop, other
  reducers), not only against the baseline.

## Environment (exact)

GNU Fortran on Linux. Use the same toolchain as the campaign (gfortran 15.2.0,
Rust 1.97.1, Python 3.13.12). On the shared host that is `tools/tirust_env.sh`
sourced **outside** `nix develop`. The current flake resolves gfortran 15.3 /
Rust 1.98 / Python 3.14 instead, and inside the dev shell the script adds
nothing. If your machine differs, record the actual versions. Keep identical
flags for the baseline and candidate.

```bash
git clone https://github.com/MadGraphTeam/MadGraph7.git && cd MadGraph7
git checkout madloop_tirust        # or the exact commit named in CURRENT_STATUS.md
source tools/tirust_env.sh
export FFLAGS='-w -fPIC -O3'
```

Install HEPTools NinjaRS, COLLIER and Ninja as described in
[the comparison-backend recipe](../../../qualification/comparison-backends/README.md)
(do not repin NinjaRS).

## 1. Fresh TIRust installation and local measurement

The real `install tirust` CLI builds the pinned source into a fresh prefix and
writes a receipt that binds the source and native bytes. Each step refuses if
the exporter inventory differs from the registration.

```bash
python3 -B tools/tirust_benchmark_local.py install --campaign tirust-2b9ba961-1 \
  --installation hs-2b9ba961-install-1 --tirust-prefix HEPTools/local-2b9ba961/tirust --jobs 4
CPU=$(python3 -B -c 'import os; print(min(os.sched_getaffinity(0)))')   # choose a quiet core
python3 -B tools/tirust_benchmark_local.py prepare --campaign tirust-2b9ba961-1 \
  --measurement hs-2b9ba961-1 --tirust-prefix HEPTools/local-2b9ba961/tirust \
  --install-receipt docs/benchmarkfolder/local-installations/hs-2b9ba961-install-1/receipt.json --cpu "$CPU"
python3 -B tools/tirust_benchmark_local.py verify --measurement hs-2b9ba961-1
for stage in init model build numerical campaign summarize; do
  python3 -B tools/tirust_benchmark_local.py forward --measurement hs-2b9ba961-1 $stage --jobs 4
done
python3 -B docs/benchmarkfolder/build_local_report.py --measurement hs-2b9ba961-1 --render-label render-1
```

**`numerical`** runs all 66 lanes (11 processes × baseline CutTools plus candidate
1/6/7/8/9 × 100 points) serially on `$CPU`. On the loaded shared host, most lanes
took 1–3 minutes and the 2→4 lanes much longer (see `cold_seconds` in
`summaries/numerical-summary.csv`). Born, finite and both poles must agree at
`3e-6`. Report the `1e-8` misses separately and keep every failure.

**`campaign`** is the acceptance timing. For each of the eight main processes it
compares the TIRust-first chain `9|<baseline default chain>` with the
materialized default chain of baseline `9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f`:
- at least seven alternating warmed pairs;
- batches of about one second or more;
- the upper 95% paired candidate/baseline bound must be **≤ 1.05 for every main
  process**.

The explicit-backend regression controls follow. Measure them even if a main
gate fails. Record initialization separately.

## 2. TIRust precision cohorts (new: per-call route provenance)

`tools/tirust_benchmark_cohorts.py` now supports backend 9. It classifies every
point by its bridge route counters as `dp_native` (binary64 only), `dp_dd`,
`dp_mpfr`, `qp` (genuine binary128 rescue) or `failed`, then times each cohort
with seven alternating repetitions. A timed call that changes tier is a
migration and invalidates that batch. These are diagnostics, not acceptance.

```bash
for case in qcd_n3_uux_zgg qcd_n4_uux_zzgg ew_n3_epem_ttxa ew_n4_epem_mupmumaa; do
  python3 -B tools/tirust_benchmark_cohorts.py run --campaign tirust-2b9ba961-1 --case $case \
    --backend 9 --label hs-cohorts-$case-9 --cpu "$CPU"
done
```

## 3. What to publish

Commit the local summaries, paired intervals, failures and the rendered local
report. State the machine, CPU, compiler versions and load. Gates that are not
met remain open. Neither the shared-host results nor the cohort diagnostics
replace them.
