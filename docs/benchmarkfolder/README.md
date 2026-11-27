# MadLoop for quarkonium at one loop: report

`benchmark.pdf` is the validation, reducer and performance report of the
one-loop virtuals of processes with S-wave and P-wave bound states in MadLoop
(branch `onia_loop`; usage and design in `docs/onia_loop.md`), with the S-wave
onium + jet virtuals against the MG5 2.5.5 onia outputs.

| Path | Content |
|---|---|
| `benchmark.typ`, `style.typ` | the report (Typst 0.15), in the style of the pyAmpliCol one-loop report |
| `data/*.json` | the tables of the report, written by `scripts/collect.py` |
| `data/raw/` | the campaign records behind them (`validate.json`; `validate_before_guard.json`, the first three cases before the bound-state reducer guard; `timing/*.json`: changes A to E; `timing_A_B_B2/`: the first three alone; `pwave/`: `pw_check.py` and `ps_term.py` of the P-wave cases, and the J/psi chi_cJ runs; `two_pwave/`: the two-P-wave comparison and runs, and `appb_*` of `research/two_pwave/appendix_b.py` and `research/pwave_fd/unequal_mass.py` against app. B of the two-P-wave manuscript; `validate_tirust*.json`: the J/psi eta_c case on unpinned TIRust builds; `v16all/`: every example rerun on TIRust da7902b + bridge v16, with the 2.5.5 points also in quadruple precision and with the heavy quarks excluded from the loops), scratch paths replaced by `WORK` |
| `scripts/collect.py` | campaign results to `data/*.json` |
| `research/literature.md` | analytic references and their conventions |
| `research/perf_audit.md`, `research/perf_audit_results/` | the MadLoop Fortran performance audit |
| `research/tirust_qp_repro/` | reproducers of the open TIRust items |
| `research/mg255_swave/` | the S-wave onium + jet comparison with MG5 2.5.5 (`data/cmp_*.json`) |
| `research/pwave_fd/` | the P-wave checks: finite differences, literature values, poles |
| `research/stability/` | the stability profiles (after appendix C of arXiv:1604.01363): `stability.py` |
| `research/estimate_gap/` | the trace of MadLoop's estimate gap at the softest u u~ > eta_c g point (double-precision loop coefficients of two nearly degenerate pentagons, amplified by the contraction) |
| `scripts/stability_plot.py`, `figures/stability.svg`, `figures/stability_octets.svg`, `figures/stability_v16.svg`, `figures/stability_v16_octets.svg` | their figures (the P-wave colour octets in the second; the last two: TIRust da7902b + bridge v16 against the builds of the records) and `data/stability.json` |

## Reproduction

From the repository root, with the toolchain of `tools/tirust_env.sh` and
TIRust installed (`install tirust`):

```sh
source tools/tirust_env.sh
# validation and reducers: every case, every library, 10 points each
python3 tools/onia_loop_validate.py --work WORK/validate --libs 9 3 8 6 7 1 \
    --npoints 10 --timeout 120 --json WORK/validate.json
# timing: one call per process, baseline checkout first (see the report)
python3 tools/onia_loop_timing.py --root base=BASELINE_CHECKOUT --root cand=. \
    --model loop_sm --process "g g > t t~ [virt=QCD]" --work WORK/gg_ttx \
    --libs "6|7|1" --npoints 100 --pairs 9 --json WORK/timing/gg_ttx.json
# the report
cd docs/benchmarkfolder
python3 scripts/collect.py --validation WORK/validate.json --reducers WORK/validate.json \
    --timing WORK/timing --mg255 research/mg255_swave/data --pwave data/raw/pwave \
    --git $(git rev-parse --short=10 HEAD) --date YYYY-MM-DD
typst compile benchmark.typ benchmark.pdf
```

The timing processes and settings are those of the timing table of the
report. The 2.5.5 and P-wave inputs are made by the scripts of
`research/mg255_swave` and `research/pwave_fd` (their READMEs). On this host `make` needs `SHELL` set to an existing bash
(`make SHELL=$(command -v bash)`), as `/bin/bash` is missing.

## The TIRust benchmark workspace

On `main` this directory is the git-ignored workspace of the TIRust benchmark
bundle (`docs/tirust-benchmarks.md`), which also has a `benchmark.typ`,
`benchmark.pdf`, `style.typ` and `README.md`. Its restore tool refuses to
overwrite changed files, so restore that bundle in a checkout without this
report.
