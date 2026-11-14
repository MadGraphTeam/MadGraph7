# Local numerical and precise timing TODO (Hua-Sheng) — campaign `tirust-9a781998-1`

Precise performance acceptance is **not established on the shared host**. It is a
TODO on Hua-Sheng's local machine. This file is the agent-ready task for this
campaign.

`tirust-9a781998-1` is a **rehearsal**: TIRust `9a781998` is the head of TIRust
PR #35, which is not yet merged into TIRust `main`. Once MadGraph pins a merged
`main` commit, a new campaign `tirust-<pin8>-1` is registered and copies this file.
Use the newest campaign listed in
[`docs/benchmarkfolder/CURRENT_STATUS.md`](../../../CURRENT_STATUS.md).

## What is already known (shared host, not acceptance)

- **Numerical lanes.** See the [numerical summary](../summaries/numerical-summary.md):
  every completed lane passes `3e-6`.
- **Double-double gate.** See the [DD-gate summary](../summaries/dd-gate-summary.md).
  Every replay is bit-identical, and DD lanes agree with 256-bit evaluation.
- **Provenance.** The [per-point provenance](../summaries/provenance-summary.md)
  classifies each point as binary64-only, DD, MPFR, genuine QP or failed.
- **Stress.** See the [stress classification](../summaries/stress-summary.md).

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
python3 -B tools/tirust_benchmark_local.py install --campaign tirust-9a781998-1 \
  --installation hs-9a781998-install-1 --tirust-prefix HEPTools/local-9a781998/tirust --jobs 4
CPU=$(python3 -B -c 'import os; print(min(os.sched_getaffinity(0)))')   # choose a quiet core
python3 -B tools/tirust_benchmark_local.py prepare --campaign tirust-9a781998-1 \
  --measurement hs-9a781998-1 --tirust-prefix HEPTools/local-9a781998/tirust \
  --install-receipt docs/benchmarkfolder/local-installations/hs-9a781998-install-1/receipt.json --cpu "$CPU"
python3 -B tools/tirust_benchmark_local.py verify --measurement hs-9a781998-1
for stage in init model build numerical campaign summarize; do
  python3 -B tools/tirust_benchmark_local.py forward --measurement hs-9a781998-1 $stage --jobs 4
done
python3 -B docs/benchmarkfolder/build_local_report.py --measurement hs-9a781998-1 --render-label render-1
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
  python3 -B tools/tirust_benchmark_cohorts.py run --campaign tirust-9a781998-1 --case $case \
    --backend 9 --label hs-cohorts-$case-9 --cpu "$CPU"
done
```

## 3. What to publish

Commit the local summaries, paired intervals, failures and the rendered local
report. State the machine, CPU, compiler versions and load. Gates that are not
met remain open. Neither the shared-host results nor the cohort diagnostics
replace them.
