# QCD three-particle ordinary-target diagnosis

All five reducers pass the registered `3e-6` criterion on all 100 saved `u u~ > z g g [virt=QCD]` points. The original `1e-8` target misses remain recorded. This read-only analysis replays the closed campaign data and previously retained higher-precision diagnostics; it runs no amplitude, compiler, or reducer and supplies no performance evidence.

Every non-CutTools reducer misses the target against the baseline CutTools result at the same **28 points**, exclusively in the finite coefficient. Total order 0 and split order 1 repeat the same coefficients: 56 missed coefficient comparisons per reducer. Born and both poles pass the target. Current baseline/candidate CutTools coefficients agree exactly at all 100 physical points (200 answer-order rows).

The criterion is `abs(a-b)/abs(a+b)<tolerance`, with the existing small-pole exception. The tables below use this symmetric metric. Machine-readable output also retains the distinct raw metric `abs(a-b)/max(abs(a),abs(b))`; no criterion is changed.

## Exact target points and accepted status

CutTools, Ninja, COLLIER and NinjaRS are accepted in DP at every listed point, with status codes **211, 216, 217, 218**, respectively. TIRust status **219** denotes accepted DP; **323** denotes accepted QP. These are final accepted outcomes, not counts of internal attempts; FPE flags and internal retries were not observed.

| Point | TIRust accepted precision / status |
|---:|:---|
| 6 | DP / 219 |
| 18 | QP / 323 |
| 20 | DP / 219 |
| 25 | DP / 219 |
| 26 | DP / 219 |
| 27 | DP / 219 |
| 34 | DP / 219 |
| 35 | DP / 219 |
| 36 | DP / 219 |
| 37 | DP / 219 |
| 38 | DP / 219 |
| 39 | QP / 323 |
| 41 | DP / 219 |
| 44 | DP / 219 |
| 48 | DP / 219 |
| 52 | QP / 323 |
| 60 | DP / 219 |
| 61 | DP / 219 |
| 64 | DP / 219 |
| 70 | DP / 219 |
| 77 | DP / 219 |
| 79 | DP / 219 |
| 87 | DP / 219 |
| 88 | DP / 219 |
| 91 | DP / 219 |
| 92 | DP / 219 |
| 93 | DP / 219 |
| 100 | DP / 219 |

`analysis.json/target_point_results` retains all five finite/Born/pole values, accuracy estimates, exact statuses and all ten pairwise finite differences at every listed point. Both original answer orders remain retained.

## Pairwise comparison on the complete 100 points

| Reducers | Largest symmetric difference | Points missing `1e-8` |
|---|---:|:---|
| CutTools / Ninja | 2.80593348e-06 | same 28 |
| CutTools / COLLIER | 2.80599688e-06 | same 28 |
| CutTools / NinjaRS | 2.80624245e-06 | same 28 |
| CutTools / TIRust | 2.80642065e-06 | same 28 |
| Ninja / COLLIER | 4.25215427e-08 | 39 |
| Ninja / NinjaRS | 8.59515640e-08 | 39 |
| Ninja / TIRust | 4.31124855e-08 | 39 |
| COLLIER / NinjaRS | 4.34300213e-08 | 39 |
| COLLIER / TIRust | 5.90942856e-10 | none |
| NinjaRS / TIRust | 4.28390784e-08 | 39 |

The largest original difference is point 93. CutTools DP gives finite `2.1732930914093367e-7`; TIRust DP gives `2.1732808930943623e-7`, a symmetric difference of `2.8064206471e-6`. This remains below `3e-6`. A four-reducer cluster alone does not establish truth.

## Retained independent higher-precision evidence

The previous diagnostic evaluated the exact same saved decimal inputs and physical card with CutTools, Ninja and TIRust in genuine fixed QP at points 39, 60, 91 and 93. All three pairs pass `1e-8`; their largest raw relative difference is below `9.9e-11`. A subsequent CutTools QP run covered all 28 target points and reproduced the four overlap points bit for bit. The fixed-QP status is **120**, with accuracy `-1`: these coefficients have **no independent stability certification** and are not evidence that the automatic rescue policy ran QP.

Current CutTools, Ninja, NinjaRS and TIRust coefficients and statuses are binary64-identical to the historical ordinary observations over all 100 points. COLLIER differs only at points 1 and 3 (at most `6.78e-21` absolute), outside the target set; every target-point coefficient is identical. The replay checks point/card hashes, subset mappings, settings, original raw QP rows and source/build records, then compares the current coefficients directly:

| Current ordinary result | Max symmetric difference from CT QP on 28 points | Target misses |
|---|---:|:---|
| CutTools | 2.80600087e-06 | all 28 |
| Ninja | 4.30630515e-08 | 39 |
| COLLIER | 5.41508829e-10 | none |
| NinjaRS | 4.28885125e-08 | 39 |
| TIRust | 4.19781169e-10 | none |

The independent QP agreement supports a CutTools DP reference bias for the large discrepancies, and a smaller Ninja/NinjaRS DP discrepancy at point 39. TIRust does not diverge from the retained QP evidence on any of these 28 points. This conclusion is restricted to the investigated subset and the unchecked higher-precision diagnostic; it does not overwrite the original target misses, certify other points, or alter any campaign acceptance result.

## Universal QCD double-pole crosscheck

The external massless colored legs are two quarks and two gluons, so `2 CF + 2 CA = 26/3`. The generated model, identical Born/loop averaging and MadLoop normalization give

```text
double pole = -Born × (26/3) × G²/(8π²)
G = 2 sqrt(alpha_s × π), alpha_s = 0.118
G reconstructed in binary64 = 1.2177157847767197
```

The retained worker sets `MU_R=1000` GeV from the incoming momenta and calls `UPDATE_AS_PARAM`; this path refreshes dependent couplings without running alpha_s. Complete copied source files are checked against the retained generation/compilation hashes. `source/fks_singular.f` lines 7752–7775 supplies the repository pole formula; `intparam_definition.inc` and `couplings.f` supply the coupling construction; `TIR_interface.f`, `loop_matrix.f` and `born_matrix.f` establish the normalization/averaging.

| Reducer | Largest `abs(observed−expected)/abs(expected)` over 100 points |
|---|---:|
| CutTools | 4.90509381e-13 |
| Ninja | 1.53867413e-11 |
| COLLIER | 2.66147872e-11 |
| NinjaRS | 5.75122065e-12 |
| TIRust | 1.32592224e-12 |

This independent double-pole diagnostic is consistent with all reducers and localizes the ordinary target issue to the finite term. **G is reconstructed from pinned source/card inputs, not directly traced from the native process.** The Born result uses the supplied `P_USER` before the loop momentum-repair step, so this is a normalization diagnostic rather than an exact native-input identity proof. It establishes neither finite nor single-pole truth.

## Reproduce without native execution

From the repository root:

```sh
python3 -B docs/benchmarkfolder/campaigns/split-dp-terminal-1/experiments/qcd-n3-target-triage-1/analyze.py
```

`index.json` binds this package; `inputs.json` binds 219 closed retained input files. The verifier checks every hash, reparses the full warmup traversals and historical raw rows using the retained parser/comparator, verifies the copied normalization-source bindings and exactly regenerates `analysis.json`. It needs no ignored build, current numerical card, installed library or network. Active QCD/EW four-particle files and mutable supervisor status are excluded. The final current campaign remains incomplete and has no formal performance result.
