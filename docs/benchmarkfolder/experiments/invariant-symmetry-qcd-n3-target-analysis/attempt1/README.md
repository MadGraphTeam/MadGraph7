# Closed invariant-symmetry QCD three-body target analysis

All six retained lanes pass the registered `3e-6` comparison for all 100
`u u~ > z g g [virt=QCD]` points. Ninja, COLLIER, NinjaRS and TIRust each miss
the tighter `1e-8` target at the same **28 points**, exclusively in the finite
coefficient. This offline investigation changes no comparison or campaign gate
and provides no performance result.

The exact target points are:

```text
6 18 20 25 26 27 34 35 36 37 38 39 41 44 48 52 60 61 64 70
77 79 87 88 91 92 93 100
```

Order 0 (total) and order 1 (the sole split order) repeat identical answers:
there are **56 missed coefficient comparisons per reducer, not 56 physical
points**. Born is bit-identical across all reducers. The largest symmetric
single-pole discrepancy against CutTools is `2.758e-11`; the largest double-pole
discrepancy is `1.330e-11`. Both poles pass `1e-8` everywhere.

The registered metric is `abs(a-b)/abs(a+b)`, with the existing small-pole
exception. The raw metric `abs(a-b)/max(abs(a),abs(b))` is also retained and must
not be confused with the acceptance metric.

| Compared with baseline CutTools | Maximum symmetric difference | Point | Accepted precision there |
|---|---:|---:|---|
| Current CutTools | 0 | all | DP / 211 |
| Ninja | 2.805933484e-6 | 93 | DP / 216 |
| COLLIER | 2.805996878e-6 | 93 | DP / 217 |
| NinjaRS | 2.806242448e-6 | 93 | DP / 218 |
| TIRust | 2.806420647e-6 | 93 | DP / 219 |

At point 93, CutTools finite is `2.1732930914093367e-7` and TIRust finite is
`2.1732808930943623e-7`. Their raw relative difference is `5.612825542e-6`,
while the registered symmetric difference is `2.806420647e-6`, below `3e-6`.
This is a tighter-target miss, not a registered-criterion failure.

Every non-CutTools finite result lies on the same side of CutTools at all 28
target points. The non-CutTools pairwise disagreement is confined to point 39:

| Pair | Maximum symmetric difference | Target misses |
|---|---:|---|
| Ninja / COLLIER | 4.252154268e-8 | 39 |
| Ninja / NinjaRS | 8.595156396e-8 | 39 |
| Ninja / TIRust | 4.311248554e-8 | 39 |
| COLLIER / NinjaRS | 4.343002128e-8 | 39 |
| COLLIER / TIRust | 5.909428562e-10 | none |
| NinjaRS / TIRust | 4.283907842e-8 | 39 |

Outside point 39, all these pairs agree within `2.118323614e-9` over the other
99 points and all coefficients. At point 39, Ninja and NinjaRS lie on opposite
sides of the closer COLLIER/TIRust pair:

| Reducer | Finite at point 39 | Accepted status | Retained accuracy estimate |
|---|---:|---:|---:|
| CutTools | -1.6424130015341697e-5 | 211 / DP | 7.943028033e-7 |
| Ninja | -1.6424145149169340e-5 | 216 / DP | 1.315424646e-7 |
| COLLIER | -1.6424143752409420e-5 | 217 / DP | 1.782798110e-10 |
| NinjaRS | -1.6424142325807658e-5 | 218 / DP | 1.433644719e-7 |
| TIRust | -1.6424143732997960e-5 | 323 / QP | 0 |

These correlations identify useful diagnostic points; they do not establish
finite-term truth. The accuracy estimates are MadLoop stability diagnostics,
not rigorous error bounds. A stored zero estimate is not exact arithmetic.

CutTools, Ninja, COLLIER and NinjaRS are accepted in DP on all 100 points.
TIRust is accepted in DP on 87 and QP on 13:

```text
16 18 21 32 39 52 54 55 65 94 97 98 99
```

Only **18, 39 and 52** are also target-miss points. Statuses describe the final
accepted backend/precision; internal attempts and working-bit tiers are not
observed. The opened result does not record IEEE flags or native effective
kinematic/model inputs.

## Comparison with the earlier closed campaign

All **1,200 accepted rows**, including **4,800 coefficient binary64 values**,
all statuses, precision labels and accuracy bits, are exactly identical to the
closed `split-dp-terminal-1` QCD three-body result. Embedded raw warmup records
also match. Baseline and current CutTools match each other exactly. Therefore
the current campaign reproduces the previous discrepancy pattern; this
comparison is not a new validation of physical input or compiled-library
identity.

The earlier pinned triage packet reports independent fixed-QP diagnostics on
points 39/60/91/93 and a 28-point CutTools QP follow-up. This packet records that
historical context only. Those rows have **status 120, accuracy -1 and no
independent stability certification**. The present replay does not revisit
their full provenance chain, prove native effective-input equality, or establish
warning-free completeness. They are not relabeled as checked results or used
to close the current tighter target.

## Remaining investigation

Prioritize point 93 (largest CutTools discrepancy), point 39 (largest non-CutTools
discrepancy), and a quiet control. A separately reviewed higher-precision
diagnostic must preserve the scientific inputs and observe the actual Born,
repaired DP/MP momenta, masses, couplings and scale before comparisons. Retain
all IEEE state and warnings; reject `##W03` omitted-contribution output before
coefficient comparisons. Status 120 stays unchecked even when warning-free.
An internal 512-bit success or a reducer majority alone does not establish
truth. No diagnostic is launched by this packet.

## Offline reproduction and scope

From the repository root:

```sh
source tools/tirust_env.sh
PYTHONDONTWRITEBYTECODE=1 python3 -B docs/benchmarkfolder/experiments/invariant-symmetry-qcd-n3-target-analysis/attempt1/analyze.py
```

`index.json` binds this packet. `analysis.json/inputs` names seven exact,
hash-pinned closed inputs, led by the current result
`59bf41a90d581fb757b2240b57162fc85c5e1f29ec14a1d54713a2255188b373`
and previous result
`8eb88c74c3c9496440c38fef052f335e4ffca851d36baa3a17a129177ffeecde`.
The script reparses both embedded warmup traversals, verifies all original
residuals, recomputes all ten pairwise comparisons, checks exact accepted-row
identity and reproduces the stored analysis. It uses only the Python standard
library and opens no active QCD four-body output, mutable campaign summary or
status, installed library, ignored binary, or native process. No source/card or
protected evidence was changed. This is neither full-matrix qualification nor
a performance measurement.
