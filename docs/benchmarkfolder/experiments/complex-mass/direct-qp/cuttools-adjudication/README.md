# CutTools point-20 diagnosis

This isolates the forced-QP CutTools disagreement for the original saved CMS
point 20, without changing the production reducer, its installed library, the
physical point, the model, the numerator, or the primary campaign. Every result
here is an unchecked fixed-QP diagnostic; none establishes primary-policy or
runtime acceptance.

CutTools gives identical `result.dat` bytes in the baseline and candidate.
With its original scalar threshold the raw Laurent coefficients are:

| Quantity | Original CutTools | CutTools with scalar cutoff `1e-10` | TIRust QP |
|---|---:|---:|---:|
| Born | `0.03386428695052817` | same | same |
| Finite | `0.0081920517374233955` | `-0.034460031100884962` | `-0.034459616824384275` |
| Single pole | `-0.0023354669830583240` | `-0.0049675395394987807` | `-0.0049675129209920781` |
| Double pole | `-0.000081389964476063457` | `-0.00016277772309331913` | `-0.00016277686432570026` |

The large discrepancy is caused by a separate CutTools scalar threshold.
`vendor/CutTools/src/cts/cts_cuttools.f90:77-84` suppresses scalar diagnostics and
sets OneLOop's absolute on-shell cutoff to `1e-6 GeV²`. MadLoop's `OSThres`
does not control that setting. The original physical transfer is approximately
`-2.5e-7 GeV²`, below the scalar cutoff. The QP scalar implementation classifies
bubbles and triangles using this absolute threshold; it selects zero/scaleless
branches for some nonzero invariants. The original run emits 16 previously
suppressed `tria0` errors and 40 soft-box warnings when its messages are enabled.

The isolated generated `INITCT` calls the existing private OneLOop API after
`CTSINIT`. It prints the threshold and all scalar inputs, then evaluates with
an explicitly recorded scalar cutoff. No scalar or tensor algorithm is changed.
The original cutoff exactly reproduces the uninstrumented bad result. Reducing
only that cutoff to `1e-10` removes all 56 diagnostics, restores the missing pole
contributions, and corrects the finite sign. All 2,708 ordered scalar input
records, including 1,076 invariants, are identical across this intervention.
The largest nonzero external virtuality is `2.64385014548454e-12 GeV²`, below
the reduced cutoff; the physical transfer remains above it. A repeat at `1e-9`
gives identical result bytes. This checks that the correction does not result
from treating external rounding noise as a physical off-shell regulator.

A residual relative finite discrepancy of `1.2022e-5` remains after the scalar
cutoff correction and still fails the exact MadLoop `3e-6` comparator. Forward,
reverse, and combined CutTools QP modes 4, 5, and 6 reproduce that residual;
they do not independently test the input kinematics. The generated legacy
CutTools interface passes DP momenta and performs DP prefix sums even in QP
mode. CutTools promotes these denominators before QP reduction. Its numerator
uses the separate repaired QP external momenta because
`UseQPIntegrandForCutTools=.TRUE.`. The independently recorded repaired-input
trace for TIRust is available in `../actual-matrix-trace/`.

The isolated `qp-transport/` experiment resolves that residual. A private copy
of the existing `CTSXCUT` routine, linked under a new symbol, receives MadLoop's
existing repaired QP momenta and QP prefix sums. Only the momentum transport
changes: the mass inputs, numerator, scalar kernels, reduction algorithm and
installed CutTools archive remain unchanged. With the reduced scalar cutoff,
the raw finite result becomes `-0.034459616825017546`; finite and pole
coefficients agree with TIRust within `2.11e-11` relative. Reversed and combined
QP modes agree within `1.07e-11`. The scalar external virtuality drops from
`2.64e-12` to `2.52e-29 GeV²`, while the physical transfer is retained.
Keeping the original scalar cutoff still gives the large disagreement even
with QP transport. The two controlled interventions therefore identify separate
legacy causes: scalar classification and DP denominator input precision.
This diagnoses the original failure; it is not a production CutTools fix or a
replacement for primary-policy stress qualification.

The double pole has an independent check within MadGraph. The QED part of
`Template/NLO/SubProcesses/fks_singular.f:7789-7804` gives
`-sum(Q_i²) * Born * abs(alphaEW)/(2*pi)` for massless charged external legs.
The four leptons give `sum(Q_i²)=4`. The generated `GAL(1)` coupling uses
`abs(alphaEW)`; its imaginary part here is only `3.45e-20`, so this equals its
real part at the recorded precision. The printed CMS model coupling predicts
`-0.0001627768643257003`; TIRust differs by `1.67e-16` relative. This check
uses the model and infrared formula, rather than agreement among reducers.

`analyze.py` revalidates physical point/card bindings, source and object hashes,
the exact original-result reproduction, the scalar input stream, runtime
cutoffs, QP settings, all raw results, and the independent double pole.
`summary.json` retains every failed comparison. The diagnostic source builds
change only `CT_interface.o` in separate ignored copies; all other numerical
objects and static archives are byte-identical. The first build omitted the
generated Ninja module include flags and failed before execution; its build log
is retained in `build-baseline-missing-include-failure/`.

From the repository root, after preparing the two direct-driver copies
described in `../README.md`, use fresh labels and unused ignored build paths:

```sh
source tools/tirust_env.sh
python docs/benchmarkfolder/experiments/complex-mass/direct-qp/cuttools-adjudication/run.py build --source baseline
python docs/benchmarkfolder/experiments/complex-mass/direct-qp/cuttools-adjudication/run.py build --source candidate
python docs/benchmarkfolder/experiments/complex-mass/direct-qp/cuttools-adjudication/run.py run --source baseline --label baseline-original-threshold --threshold 1e-6
python docs/benchmarkfolder/experiments/complex-mass/direct-qp/cuttools-adjudication/run.py run --source baseline --label baseline-threshold-1e-10 --threshold 1e-10
python docs/benchmarkfolder/experiments/complex-mass/direct-qp/cuttools-adjudication/run.py run --source baseline --label baseline-threshold-1e-9 --threshold 1e-9
python docs/benchmarkfolder/experiments/complex-mass/direct-qp/cuttools-adjudication/run.py run --source baseline --label baseline-threshold-1e-10-reverse --threshold 1e-10 --mode 5
python docs/benchmarkfolder/experiments/complex-mass/direct-qp/cuttools-adjudication/run.py run --source baseline --label baseline-threshold-1e-10-both --threshold 1e-10 --mode 6
python docs/benchmarkfolder/experiments/complex-mass/direct-qp/cuttools-adjudication/run.py run --source candidate --label candidate-original-threshold --threshold 1e-6
python docs/benchmarkfolder/experiments/complex-mass/direct-qp/cuttools-adjudication/run.py run --source candidate --label candidate-threshold-1e-10 --threshold 1e-10
python docs/benchmarkfolder/experiments/complex-mass/direct-qp/cuttools-adjudication/analyze.py
```
