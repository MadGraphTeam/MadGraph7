# Isolated DP terminal-wavefunction experiment

This directory studies a private final double precision wavefunction update in
the frozen `qcd_n1_uux_z` generated process. It does not modify the production
generator, the public dense updater, or the earlier quadruple precision
optimization. None of its timings are acceptance-campaign measurements.

The final `UPDATE_WL_1_1` calculates a 4-by-4 matrix whose only consumer,
`CREATE_LOOP_COEFS` through `MERGE_WL`, reads its diagonal. The candidate keeps
the complete zero initialization and every consumed expression in the original
order, but evaluates only diagonal arithmetic when a narrow runtime guard
succeeds. Every uncertain case calls the untouched dense updater.

`proof-audit.md` records the source and arithmetic argument. Admission requires
the exact 4-by-4 shape, binary64 representation, nearest rounding, all exception
traps disabled, and an INEXACT flag already set immediately before the call.
Each input component must be signed zero or have unbiased exponent between
-400 and 400 inclusive. These bounds exclude new invalid, divide-by-zero,
overflow and underflow exceptions from omitted arithmetic; already-sticky,
nontrapping INEXACT accounts for possible discarded rounding exceptions.

The root-level prototype uses an elemental bit classifier. `packed-scan/` is
an independently retained variant with the same admission rule and arithmetic,
using bulk bit transfers and an integer scan to reduce classifier overhead.
Both passed 1,536,000 comparisons across shapes, operand fixtures, rounding
modes and all pre-existing flag combinations, plus 32 paired exception-trap
probes. Each was compiled under four GNU Fortran 15.2.0 configurations. Tests
compare consumed output bits, complete dense output on fallback, unchanged
inputs, every IEEE flag and the rounding mode. Raw commands and outcomes are
in each variant's `guard-results.json`.

Synthetic kernel timings include the caller's `IEEE_GET_FLAG` query. Seven
alternating pairs gave these median candidate/dense ratios:

| Guard | `-O0` | `-O3` |
| --- | ---: | ---: |
| Elemental | 1.2996 | 0.5485 |
| Bulk scan | 0.4763 | 0.7438 |

The slower elemental `-O0` result is retained. **The generated process compiles
`polynomial.f` with its existing special `-O3` rule.** Its caller and other
ordinary process sources have no optimization flag. Thus the `-O0` kernel
column is a compiler-sensitivity diagnostic, not the actual generated
polynomial configuration. Neither column predicts complete-amplitude runtime.

`prepare_amplitude.py` creates private copies of the frozen output and checks
every recorded source, compiler recipe, injected compiler environment and
static archive. An untouched relink must first reproduce the original
executable byte-for-byte. It then changes only `polynomial.o` and
`coef_construction_1.o`, preserving the original flags, worker and reduction
libraries. The actual caller captures INEXACT directly before its private
terminal call. Separate admission-trace builds print the observed guard
decision; they are excluded from timing comparisons.

The first fixed-form conversion omitted the label columns for `900 CONTINUE`.
Its compiler rejection and exact source are retained under
`amplitude-build-elemental-initial-label-error/`. No numerical process ran
that failed build. The corrected conversion preserves the label and subsequent
ordinary and trace builds succeeded.

Reproduce from the MadGraph root after sourcing `tools/tirust_env.sh`:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 docs/benchmarkfolder/experiments/dp-terminal/prepare.py
PYTHONDONTWRITEBYTECODE=1 python3 docs/benchmarkfolder/experiments/dp-terminal/run_guards.py
PYTHONDONTWRITEBYTECODE=1 python3 docs/benchmarkfolder/experiments/dp-terminal/prepare_packed.py
PYTHONDONTWRITEBYTECODE=1 python3 docs/benchmarkfolder/experiments/dp-terminal/packed-scan/run_guards.py
```

The generators require the explicitly identified frozen generated output.
Do not run timing diagnostics during another campaign's timing window.
Generated-amplitude native executions use the shared numerical lock and the
same pinned split-workspace DSO for the original and terminal candidates.
Production promotion additionally requires the DP-specific compiled-consumer
and alias proof, complete-amplitude equivalence, and a measured improvement.

## Complete generated-amplitude results

Both ordinary variants and the separate trace executable passed the 100-point
test stream. Their calibration and observed coefficients match the unchanged
candidate bit for bit, and every call reported the explicitly selected DP
backend. The trace recorded 1,212 calls, all with pre-existing INEXACT and all
taking the guarded path; the count includes initialization and warmup.

The subsequent seven-pair comparisons used 200 traversals of the same 100
points per measured pass, balanced execution order, CPU 9, unchanged compiler
recipes and the same pinned split-workspace DSO for both candidate versions.
The original generated NinjaRS baseline remained separate. Each process also
performed a complete calibration pass before its observed pass. Other agents'
compiler and native jobs were held during these diagnostics, but this shared
host experiment is **not an acceptance campaign**.

| Guard | Terminal/original TIRust ratio, 95% CI | Terminal/original NinjaRS ratio, 95% CI |
| --- | --- | --- |
| Elemental | 0.88866 [0.87559, 0.90193] | 0.97852 [0.96888, 0.98825] |
| Bulk scan | 0.94463 [0.93676, 0.95256] | 1.04235 [1.03396, 1.05081] |

All calibration and measured coefficients remain bit-identical between each
terminal candidate and its original TIRust control. The elemental candidate
had median 90.82 microseconds per amplitude, versus 102.43 for its original
TIRust control and 92.71 for the original NinjaRS baseline in that series.
The bulk variant's upper bound exceeds 1.05 relative to the NinjaRS baseline;
that result is retained. The unchanged TIRust control was also slower than
NinjaRS in both series (ratios 1.10111 and 1.10346). These diagnostics do not
replace the earlier qualified c91 failure or change any legacy summary.

`analysis-summary.json` checks the complete retained observations, including
calibration coefficients. Each run directory contains the exact worker/card,
runtime-library identities, raw values/status/timings, source auditor identity
and copied runner. The independent consumer proof checks 111 pinned inputs,
49 actually compiled units, 30 include files and the exact DP COMMON aliases;
36 mutation tests pass. It is deliberately specific to this frozen export.

The three initial parity runs used auditor SHA `34642697...`; that exact script
is retained as `consumer-audit/audit_consumers-before-gnu-makefile.py`. A final
hardening rejects unknown `GNUmakefile` precedence. The two paired series used
the final `809409a5...` auditor. `final-source-attestations.json` additionally
rechecks all unchanged compiled clones using the final auditor. No numerical
source or generated executable changed between those source audits.
