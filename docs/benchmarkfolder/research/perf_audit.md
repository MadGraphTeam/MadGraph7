# MadLoop performance audit (branch onia_loop @ 5c5b573794)

Audit subagent, 2026-11-21. Raw data, diffs and tools are in `perf_audit_results/`
(copied from the audit scratch directory).

## Method
- **Environment:** `tirust_env.sh` (gfortran 15.2) on an AMD EPYC 9754, a shared host at load 80–100, with `OMP_NUM_THREADS=1`.
- **Processes:** `loop_sm` `output standalone` for u u~ > z, g g > t t~, u u~ > z g g and g g > t t~ g (585 loops, 272 groups, 852 CT amplitudes, 18 Born amplitudes, 32 helicities).
- **Settings:** default `MLReductionLib=6|7|1` and `CTModeRun=-1`.
- **Driver:** `bench_main.f` reads 400 fixed points and installs a reference helicity filter. It times the steady state and writes every result in hex.
- **Comparison:** bit identity is checked with `cmp`. Timing is the median ratio of 7 alternating paired runs, plus `perf stat`.

## Key finding
The standalone loop output is compiled at -O0 except `polynomial.f`:
- `Template/LO/Source/make_opts` sets `FFLAGS=-w -fPIC`, and `GLOBAL_FLAG` is only written by the run interfaces.
- The DP-terminal proof (`dp_terminal_proof.py`) rejects any `GLOBAL_FLAG` in `make_opts`, so adding one there silently disables the PR #508 updaters.
- MC@NLO output is not affected.

## Profile of g g > t t~ g (Ninja)

| Phase | Base | After the full stack |
|---|---|---|
| `UPDATE_WL` | 28.2% | 40.6% |
| `DP_TERMINAL_WL` | 16.7% | 26.8% |
| memset | 5.3% | 0% |
| `CREATE`+`MERGE` | 6.6% | 7.0% |
| `SLOOPMATRIX` CT colour loop | 16.1% | 2.0% |
| split-order index functions | 2.8% | 0% |
| IEEE save/restore | 3.0% | 0.2% |
| HELAS | 7.9% | 5.2% |
| Ninja | 9.5% | 13.9% |

## Ranked proposals
All are bit-identical. Ratios are B/A runtimes; below 1 is faster.

| # | Change | gg>tt~g | uu~>zgg | gg>tt~ |
|---|---|---|---|---|
| A | `MADLOOP_OPTIMIZATION=-O2` before `$(FFLAGS)` in the two `$(DOTO)` rules of the loop makefile | 0.828 | 0.937 | 0.925 |
| D | IEEE rounding/halting checks hoisted to `COEF_CONSTRUCTION_*` (on A+B+B2+C) | 0.872 | 0.946 | 0.969 |
| B | CT colour contraction: colour factors and SO indices tabulated at init, register sums when NSQUAREDSO=1 | 0.903 | 0.969 | 0.978 |
| E | DHELAS/MODEL at -O2 (on A+B+B2+C+D) | 0.920 | 0.901 | 0.912 |
| B2 | split-order tables in `CREATE_LOOP_COEFS` | 0.952 | 0.992 | 0.999 |
| C | rank-limited clearing of the DP updaters' `OUT` | 0.946 | 0.989 | 0.995 |
| F | `-march=x86-64-v3 -ffp-contract=off` (opt-in) | 0.897 | 0.923 | 0.904 |
| G | `-march=native -ffp-contract=off` (opt-in) | 0.847 | 0.907 | 0.871 |

Stacked:

| Stack | gg>tt~g | uu~>zgg | gg>tt~ |
|---|---|---|---|
| A+B+B2+C+D | 0.710 | 0.856 | 0.878 |
| A+B+B2+C+D+E | 0.643 (84.1 → 54.6 ms) | 0.757 (25.0 → 19.0 ms) | 0.811 (1.086 → 0.881 ms) |

## Caveats
- **C:** production results are identical, but it breaks 20 PR #508 tests that lock in full clearing.
- **D:** needs changes to the generator (`write_dp_terminal_wl_updater`, `prepare_dp_terminal_batch`) and to the proof's capture grammar.
- **E:** needs the LO `make_opts` templates or the proof's whitelist.
- **B2:** one DP-terminal test harness must initialise the new COMMON.
- **IOTests:** the goldens change.

## Not prototyped
- A vectorised bounds scan of the DP-terminal guard (3–6%).
- A colour-summed Born vector for the CT loop.
- Monomial reuse in `EVAL_POLY`.
- Skipping exact-zero blocks.
- Reusing `WL` slots.
