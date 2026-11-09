# QCD 2→3 finite-coefficient adjudication

The strict-policy `u u~ > z g g [virt=QCD]` pilot passes the original MadLoop
`3e-6` symmetric comparison for all five reducers. The stronger `1e-8` ordinary
target is missed at the same 28 points by every non-CutTools reducer when
compared with CutTools DP. These are finite-coefficient discrepancies; total
and split-order 1 repeat the same result. Baseline and candidate CutTools DP
answers are identical throughout the saved pilot.

The largest difference is point 93:

| Reducer | Accepted precision | Finite coefficient |
|---|---|---:|
| CutTools (baseline and candidate) | DP | 2.1732930914093367e-7 |
| Ninja | DP | 2.1732808952118478e-7 |
| COLLIER | DP | 2.1732808949363016e-7 |
| NinjaRS | DP | 2.1732808938689157e-7 |
| TIRust | DP | 2.1732808930943623e-7 |

The worst relative difference is `5.61282554230633e-6`; the corresponding
symmetric difference is `2.8064206471279104e-6`, below the regression gate.
CutTools' internal estimate is `4.074658777825374e-7`, below the shared
`MLStabThres=1e-6`. Agreement among the other four reducers suggests a CutTools
DP reference error, but this initial observation is not an independent truth
determination.

Among non-CutTools reducers, only point 39 misses the `1e-8` target: Ninja and
NinjaRS finite answers differ from COLLIER/TIRust by roughly `4e-8` under the
symmetric metric. COLLIER and the already QP-accepted TIRust answer differ by
`5.909428562178099e-10`. The 100-point TIRust pilot has 87 DP and 13 QP accepted
outcomes; those counts do not enumerate internal attempts.

The independent four-point QP follow-up has completed. CutTools/Ninja,
CutTools/TIRust and Ninja/TIRust agree to maximum relative differences of
`9.79365e-11`, `9.88681e-11` and `9.31604e-13`, respectively, across all retained
Born and Laurent columns. Every pair passes the `1e-8` target. At point 93,
CutTools QP gives finite `2.173280894918967e-7`, confirming the other reducers'
cluster rather than the original CutTools DP reference.

Against this QP reference on the four selected points, the original CutTools
DP result has maximum relative difference `5.61199e-6` (symmetric
`2.80600e-6`), which still passes the `3e-6` regression gate. Original Ninja
and NinjaRS DP results miss the stronger target only at point 39, with
symmetric differences `4.30631e-8` and `4.28885e-8`. Original COLLIER and
TIRust results pass the target throughout this subset. Agreement of three
independent QP lanes supports this adjudication; the fixed-QP outputs remain
explicitly unchecked and do not certify automatic rescue behavior.

The authorized CutTools QP extension also completed all 28 original target
misses. Its four overlapping points reproduce the first QP run exactly.
Comparison against the unchanged saved ordinary results gives:

| Original ordinary result | Maximum relative difference from CT QP | Points missing `1e-8` symmetric target |
|---|---:|---:|
| CutTools DP (baseline and candidate) | 5.61199e-6 | All 28 |
| Ninja | 8.61261e-8 | 39 only |
| COLLIER | 1.08302e-9 | None |
| NinjaRS | 8.57770e-8 | 39 only |
| TIRust | 8.39562e-10 | None |

All these comparisons pass the existing `3e-6` regression criterion. Thus the
large original discrepancy is attributable to the CutTools DP reference, with
smaller Ninja/NinjaRS DP target misses at point 39. TIRust's ordinary saved
results meet the target against this QP reference at all 28 investigated
points. This conclusion does not replace the retained original comparisons,
extend the QP crosscheck to the other 72 points, or change the shared policy.

## Follow-up and scope

`run.py` prepares exact original decimal inputs for points 39, 60, 91 and 93,
then requests fixed QP from baseline CutTools, candidate Ninja and candidate
TIRust. It copies the existing source-bound executables and mutable process
files into isolated working directories, then recompiles only the protocol
driver with `FORCE_STABILITY_CHECK(.FALSE.)`. Every numerical object, archive
and compiler policy must remain unchanged. Read-only
dependency aliases retain the exact original link-input identities. The
current protocol keeps the original DP initialization (`CTModeInit=1`) and
sets `CTModeRun=4`; the frozen physical card, scale,
pole settings, and `MLStabThres=1e-6` remain the same. All selected incoming
totals are `(1000,0,0,0)` GeV, so the worker's actual scale formula gives
`MU_R=1000` GeV and `MU_R**2=1e6` GeV². Original numerical cards and filters are
restored in each isolated copy, including after failed workers. Only TIRust
requires the shared Symbolica numerical lock: generated sole-backend CutTools
and Ninja dispatch does not call the Symbolica-backed numerical routines.

Fixed-mode status 120 provides unchecked QP coefficients. It is explicitly a
numerical diagnostic, without independent stability certification or timing
qualification. Physical/card/protocol hashes, original point indices, exact
card overrides, raw outputs, failures and QP comparisons are retained. The
original pilot and its ordinary-target misses are not overwritten. This
adjudication does not change the frozen acceptance policy or target.

After agreement on the four diagnostic points, CutTools QP evaluated all
28 original target-missing points for comparison with every saved pilot row.
Native execution began after the main campaign released its timing window.
Consult `summary.json` and each
retained run's `result.json` or `failure.json` for subsequent observations.

The initial `runs/minimal/baseline-1` attempt used `CTModeInit=4` and had not
emitted the worker's READY message after 302.6 seconds. It was interrupted at
the agreed five-minute bound. `interruption.json`, its original protocol,
configuration and output remain retained. An explicit administrative abort
on its own FIFO invoked the worker's process-group cleanup; this is not an
observed numerical accuracy failure. Subsequent `*-init-dp` directories retain
the initialization-only amendment; these also failed to reach READY and were
administratively stopped after the protocol cause was identified. Fixed QP
returns accuracy `-1`, and the original unconditional forced check converts its
steady status into T=4 on every warmup call, preventing the original worker
from reaching READY. These are protocol-confounded attempts, not QP numerical
failures. The corrected `*-unchecked` directories retain the exact derived
driver, build log, before/after object and library hashes, and compilation
record. Their only driver change disables this incompatible forced check.
Genuine QP final evaluation, numerator precision, helicity/color sums and
physical inputs are unchanged. The matching original worker template and
Python transport are frozen under `protocol/`, independently of later main
campaign protocol corrections.

```sh
source tools/tirust_env.sh
python docs/benchmarkfolder/experiments/qcd-n3-adjudication/run.py prepare
# Run only after the campaign releases its exclusive numerical window.
python docs/benchmarkfolder/experiments/qcd-n3-adjudication/run.py run --backend 1
python docs/benchmarkfolder/experiments/qcd-n3-adjudication/run.py run --backend 6
python docs/benchmarkfolder/experiments/qcd-n3-adjudication/run.py run --backend 9
python docs/benchmarkfolder/experiments/qcd-n3-adjudication/run.py summarize
# Conditional follow-up after checking the QP comparisons:
python docs/benchmarkfolder/experiments/qcd-n3-adjudication/run.py run --group target-misses --backend 1
python docs/benchmarkfolder/experiments/qcd-n3-adjudication/run.py summarize
```

The protocol received an independent read-only review. Before execution it was
corrected to exclude address-randomized `ldd` text from library equality, keep
state restoration within the numerical lock, and validate retained subset
and card bindings before comparisons. A second review found no numerical
provenance blocker in the isolated driver-only compilation. The final cleanup
uses the worker context manager to preserve an original failed-run exception
while terminating that worker; successful retained runs predate this cleanup
change and preserve their exact executed protocol copies.
