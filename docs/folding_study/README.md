# Folding, Simpson and quasi-random rules for the MC@NLO S events

Study of 2026-10-19 (branch `claude/simpson-folding-counter-term-f515fd`).

Questions:
1. Does the folding use a Simpson-type rule for the integral of the counterterms over the radiation variables?
2. If not, how many points per Born point would a Simpson rule need for 1% precision?
3. Can quasi-random points or similar tricks reduce the negative weights at less CPU than today's folding?

The integral has to be exact *on average* over Born points, not per Born point (see the exactness rule in the memory notes).

## 1. What the folding does today

`mint_module.f90` (`get_random_x_next_fold`) splits the vegas grid of each folded dimension (ξ_i, y_ij, φ_i) into `ifold` groups of cells. In vegas-flat coordinates, image k sits at t_k = (U + k − 1)/n, with U uniform and equal weights 1/n.

So the folding is a **randomly shifted rectangle rule**: a Cranley–Patterson rotated product lattice. It is not Simpson. It is exact on average (each node is uniformly distributed), and for a smooth non-periodic integrand its error falls like 1/n per dimension.

The other facts that matter are:

* **One set of FKS configurations per MINT point.** With `sum=3`, one set of FKS configurations (one per soft singularity) is picked by MC (`get_MC_integer`), once per MINT point, before the fold loop. All images therefore share it.
  * The n-body part (B+V+integrated soft/collinear) is weighted by 1/(N_sets p_k), and the n+1-body S terms by 1/p_k.
  * This discrete MC is a noise source that no rule in (ξ, y, φ) can touch.
* **The n-body is recomputed at every image** and weighted by that image's vegas × FKS jacobian. The resulting density is not flat in the radiation variables.
  * It acts as a control variate: replacing it by its exact radiation integral makes the S negatives much worse (t t̄: r_S 0.0006 → 0.04 at 8×4).
  * This is the same mechanism as "spreading" in the virt-trick study.
* Fixed order never folds (`driver_mintFO` forces 1,1,1).

## 2. What was implemented (Template/NLO)

Every new switch is in `FKS_params.dat` and off by default; with the defaults, the cross section and events are identical to before.

| switch | meaning |
|---|---|
| `FoldTransform` | node map for ξ and y. 0: today's lattice; 1: tent (baker) map; **2: cubic t²(3−2t), weights 6t(1−t), a randomized rule of Simpson order**; 3: quintic; 4: t − sin(2πt)/2π. |
| `FoldLattice` | N > 0: rank-1 (Fibonacci) lattice with N points in (ξ, y), a quasi-random rule, instead of the product lattice. |
| `FoldRealOnce` | H-event contributions (real ME and n+1-body MC terms) on one image only, picked according to the image weights and rescaled to their sum. S terms stay on all images. |
| `FoldSumSets` | S terms of **all** FKS sets summed at every image. H terms still come from the set picked by MC, divided by its probability. |
| `FoldDump` | study only: 1 = evaluate the rules of `fold_dump.dat` on the same Born points; 2 = the same, with replica r forced onto FKS set r. |

The randomized rules and both tricks are exact: every node is uniformly distributed (the transforms carry their jacobians), and the image or set used for the H events is drawn with a known probability.

The deterministic composite Simpson rule is available only in the dump (`FoldTransform` 5 in `fold_dump.dat`), since a fixed rule cannot be exact on average.

Two pre-existing issues were fixed along the way:
* `weight_lines_allocated` grew the contribution arrays one entry at a time. That is O(n²), and `icontr_sum` is n×n, so it was 28× slower with 2000 images per point. It now grows geometrically.
* `add_wgt` now works on copies of its weights. Callers pass literal `0d0`, and rescaling them in place crashed with SIGBUS.

## 3. Dumps: the same Born points under each rule

Processes: `p p > t t~ [QCD]` (channels gg and uū) and `p p > e+ e- j [QCD]` (channels ug and uū), with MINT0 grids from the standard setup, in MINT1 mode (virt-trick average, as used for the envelope).

r_S is the negative fraction of the S-event weight. r adds the H events. N_eff = (1 − 2r)².

### 3a. Deterministic Simpson is biased on the average

| rule | nodes | bias of ⟨S⟩, t t̄ | bias of ⟨S⟩, e+e- j |
|---|---|---|---|
| Simpson 2×2 intervals | 9 | −12.6% | +36% |
| Simpson 4×4 | 25 | −11.2% | +22% |
| Simpson 8×8 | 81 | −6.7% | – |

The integrand in vegas-flat variables has integrable endpoint singularities (soft and collinear) and step functions: MC dead zones, ξ_cut and δ cuts, PDF thresholds. A fixed Simpson rule therefore converges slowly, and it does so with a bias that does not average away. **It fails the exactness requirement.**

### 3b. Precision per Born point, at a fixed FKS set

From `fixedset_*.txt`: the median relative error of the S weight of one set, against a 256-node cubic reference.

| rule | nodes | t t̄ | e+e- j |
|---|---|---|---|
| lattice (today) 2×2 | 4 | 21% | 46% |
| lattice 8×4 | 32 | 12% | 29% |
| cubic 4×4 | 16 | 10% | 27% |
| cubic 8×4 | 32 | 8% | 25% |
| rank-1 cubic N=21 | 21 | 7% | 26% |

Two independent 256-node cubic estimates of the same set and point differ by a median of 18–21% in e+e- j, i.e. about 14% per estimate.

The convergence is Monte-Carlo-like (≈ N^-1/2): the rules of Simpson order buy only a constant factor, because the integrand is not smooth.

Extrapolating, **1% per Born point needs about 2000 nodes for t t̄ and more than 10⁴ for e+e- j**. Both are far beyond any sensible budget. As the question anticipated, per-point precision is not the right target: only the sign of each S event matters.

### 3c. Negative S weights versus nodes

From `rules_tt.txt` and `fixedset_eej.txt`. "MC" is today's random pick of one FKS set; "SUM" is `FoldSumSets`.

| rule | nodes | t t̄ r_S (MC) | e+e- j r_S (MC) | e+e- j r_S (SUM) |
|---|---|---|---|---|
| lattice 1×1 | 1 | 0.062 | 0.249 | 0.175 |
| lattice 2×2 | 4 | 0.0078 | 0.238 | 0.194 |
| lattice 4×4 | 16 | 0.0013 | 0.120 | 0.067 |
| lattice 8×4 | 32 | 0.0006 | 0.094 | 0.040 |
| cubic 2×2 | 4 | 0.0016 | 0.138 | 0.058 |
| cubic 4×4 | 16 | 0.0003 | 0.084 | 0.031 |
| rank-1 cubic N=13 | 13 | 0.0002 | 0.105 | 0.052 |
| cubic 16×16 | 256 | 0.0002 | 0.073 | 0.030 |

* **t t̄**: the cubic 2×2 rule gives 5× fewer S negatives than the 2×2 lattice, and cubic 4×4 does as well as lattice 8×8. In t t̄, however, almost all negatives come from H events (r ≈ 0.17), so the S part saturates at 2×2–4×4.
* **e+e- j**: the S negatives plateau at r_S ≈ 0.07 even with 256 nodes when one FKS set is picked per point. Summing the sets lowers that floor to 0.03.
  * The cubic rule is worth about 2× in nodes.
  * The quasi-random rank-1 lattice is no better than the product lattice of the same size.
  * Tent, quintic and sin² are between lattice and cubic.

## 4. End-to-end runs

Setup:
* seed 21 for every variant;
* 20k events for t t̄ and 10k for e+e- j and e+e- → b b̄ b b̄ (default cards, √s = 1 TeV, m_b = 4.7 GeV);
* r is counted from the XWGTUP signs;
* CPU is MINT1 + MINT2 `Time spent in Total` summed over the jobs;
* N_eff = N(1 − 2r)².

**Load control.** A first pass ran partly while Microsoft Defender pushed the load average to 150–260 on 18 cores, which inflated the process CPU time by up to 1.7×. Two runs show it:
* the same-seed lattice and cubic 2,2,1 runs did identical numbers of MINT1 points but differed by 1.7× in CPU per point;
* the e+e- j and t t̄ 8,4,1 baselines were 9% and 2× too expensive.

All t t̄ and e+e- j points below were therefore re-run consecutively at normal load. Same-seed re-runs reproduce σ and r exactly. The FoldRealOnce variants are left out (see below).

Plots: `rS_vs_cpu.png` (t t̄ and e+e- j) and `rS_vs_cpu_ee4b.png`. The grey curves are lines of constant CPU/N_eff; they use the process-averaged S fraction and r_H.

**p p > t t~**

| folding | method | r | r_S | r_H | CPU/event | CPU/N_eff |
|---|---|---|---|---|---|---|
| 1,1,1 | lattice | 0.225 | 0.081 | 0.637 | 1.17 ms | 3.87 ms |
| 1,1,1 | SumSets | 0.211 | 0.053 | 0.638 | 2.05 ms | 6.13 ms |
| 2,2,1 | lattice | 0.197 | 0.017 | 0.644 | 3.56 ms | 9.7 ms |
| 2,2,1 | lattice+SumSets | 0.193 | 0.013 | 0.629 | 7.04 ms | 18.7 ms |
| 2,2,1 | cubic | 0.198 | 0.015 | 0.643 | 3.12 ms | 8.54 ms |
| 2,2,1 | cubic+SumSets | 0.198 | 0.010 | 0.650 | 6.8 ms | 18.7 ms |
| 4,4,1 | lattice+SumSets | 0.190 | 0.013 | 0.641 | 26.4 ms | 69 ms |
| 4,4,1 | cubic | 0.192 | 0.009 | 0.639 | 12 ms | 31.6 ms |
| 4,4,1 | cubic+SumSets | 0.194 | 0.015 | 0.629 | 25.2 ms | 67.4 ms |
| 8,4,1 | lattice | 0.195 | 0.011 | 0.643 | 26 ms | 69.8 ms |

**p p > e+ e- j**

| folding | method | r | r_S | r_H | CPU/event | CPU/N_eff |
|---|---|---|---|---|---|---|
| 1,1,1 | lattice | 0.304 | 0.253 | 0.450 | 49.4 ms | 320 ms |
| 1,1,1 | SumSets | 0.273 | 0.211 | 0.432 | 88.2 ms | 427 ms |
| 2,2,1 | lattice | 0.238 | 0.147 | 0.428 | 189 ms | 685 ms |
| 2,2,1 | lattice+SumSets | 0.234 | 0.123 | 0.436 | 344 ms | 1.22 s |
| 2,2,1 | cubic | 0.257 | 0.167 | 0.445 | 179 ms | 754 ms |
| 2,2,1 | cubic+SumSets | 0.231 | 0.122 | 0.427 | 336 ms | 1.16 s |
| 4,4,1 | lattice+SumSets | 0.220 | 0.099 | 0.433 | 1.38 s | 4.38 s |
| 4,4,1 | cubic | 0.235 | 0.113 | 0.452 | 696 ms | 2.48 s |
| 4,4,1 | cubic+SumSets | 0.218 | 0.086 | 0.444 | 1.36 s | 4.27 s |
| 8,4,1 | lattice | 0.236 | 0.118 | 0.458 | 1.47 s | 5.27 s |

**e+ e- > b b~ b b~**

| folding | method | r | r_S | r_H | CPU/event | CPU/N_eff |
|---|---|---|---|---|---|---|
| 1,1,1 | lattice | 0.318 | 0.284 | 0.429 | 23.9 ms | 181 ms |
| 1,1,1 | SumSets | 0.322 | 0.272 | 0.468 | 36.9 ms | 292 ms |
| 2,2,1 | lattice | 0.300 | 0.245 | 0.435 | 92 ms | 575 ms |
| 2,2,1 | lattice+SumSets | 0.273 | 0.202 | 0.423 | 134 ms | 651 ms |
| 2,2,1 | cubic | 0.287 | 0.225 | 0.425 | 93 ms | 512 ms |
| 2,2,1 | cubic+SumSets | 0.262 | 0.181 | 0.421 | 138 ms | 606 ms |
| 4,4,1 | lattice+SumSets | 0.259 | 0.164 | 0.430 | 532 ms | 2.29 s |
| 4,4,1 | cubic | 0.284 | 0.214 | 0.427 | 351 ms | 1.89 s |
| 4,4,1 | cubic+SumSets | 0.248 | 0.146 | 0.425 | 538 ms | 2.12 s |
| 8,4,1 | lattice | 0.278 | 0.214 | 0.417 | 746 ms | 3.79 s |

**p p > z j j** (nic6, one 32-core job per variant run in parallel; 10k events, `req_acc` fixed at 0.0224 = 1/√2000; anti-kT R = 0.4, p_T,j > 30 GeV; plot `rS_vs_cpu_zjj.png`):

| folding | method | r | r_S | r_H | CPU/event | CPU/N_eff |
|---|---|---|---|---|---|---|
| 1,1,1 | lattice | 0.364 | 0.216 | 0.638 | 422 ms | 5.67 s |
| 1,1,1 | SumSets | 0.355 | 0.188 | 0.634 | 893 ms | 10.7 s |
| 2,2,1 | lattice | 0.336 | 0.143 | 0.615 | 1.59 s | 14.8 s |
| 2,2,1 | lattice+SumSets | 0.347 | 0.112 | 0.653 | 3.05 s | 32.4 s |
| 2,2,1 | cubic | 0.342 | 0.142 | 0.629 | 1.55 s | 15.5 s |
| 2,2,1 | cubic+SumSets | 0.332 | 0.115 | 0.611 | 3.65 s | 32.2 s |
| 4,4,1 | lattice+SumSets | 0.331 | 0.097 | 0.618 | 12.2 s | 107 s |
| 4,4,1 | cubic | 0.334 | 0.106 | 0.621 | 6.12 s | 55.6 s |
| 4,4,1 | cubic+SumSets | 0.335 | 0.090 | 0.628 | 13.9 s | 127 s |
| 8,4,1 | lattice | 0.331 | 0.113 | 0.605 | 12.8 s | 112 s |

* Here the H events are 61–65% negative and make up about 42% of the events, so they dominate r.
* FoldSumSets lowers r_S at every folding, but it costs about 2× CPU and r barely moves.
* 2,2,1 already reaches r = 0.336, against 0.331 for 8,4,1, at 7.5× less CPU per N_eff (14.8 s vs 112 s).
* Cubic and lattice are equal at 2,2,1.
* The CPU of jobs sharing a node varies by about ±20% (e.g. the two 2,2,1 + SumSets runs).

**FoldRealOnce** (first pass, not in the plots):
* On t t̄ it halved the CPU of 8,4,1 at the same r.
* On e+e- j (cubic 2,2,1 + SumSets + RealOnce), σ came out 3% low: 1147 ± 7 pb against 1183. On a fixed grid in a single channel it agrees within 1.4σ, but it doubles the error.
* The H weight of a point sits on one image and becomes heavy-tailed. MINT's inverse-variance combination of iterations then favours the low-fluctuating ones.
* It also doubles the MINT2 upper-bound failures (t t̄ 8,4,1: 0.17% → 0.40%; the default 1,1,1 has 0.8%).

## 5. Conclusions

1. **Today's folding is not Simpson.** It is a randomly shifted equal-weight lattice in vegas-flat (ξ, y, φ), which is exact.
2. **Deterministic Simpson is biased on the average** (−12% to +36%), so it is rejected.
3. **1% per Born point is not reachable** with any rule: about 2000 nodes for t t̄ and more than 10⁴ for e+e- j. The integrand is not smooth, and the per-point noise is dominated by the MC over FKS sets.
4. **`FoldSumSets` is the main gain.** At 2,2,1 it gives as low a negative fraction as today's 8,4,1, or lower, at a fraction of the cost:

   | process | 8,4,1 lattice | 2,2,1 cubic + SumSets |
   |---|---|---|
   | t t̄ | r 0.195 at 70 ms | r 0.198 at 18.7 ms (3.7× cheaper) |
   | e+e- j | r 0.236 at 5.27 s | r 0.231 at 1.16 s (4.5× cheaper) |
   | e+e- → b b̄ b b̄ | r 0.278 at 3.79 s | r 0.262 at 0.61 s (6.3× cheaper) |

   All values are CPU per N_eff. At 4,4,1, SumSets gives the lowest r in e+e- j (0.218) and e+e- → b b̄ b b̄ (0.248), still cheaper than 8,4,1. σ agrees everywhere.
5. **The cubic rule (`FoldTransform=2`) costs the same as the lattice and is a modest extra on top.**
   * e+e- → b b̄ b b̄: it lowers r_S in every pair (2,2,1: 0.245 → 0.225; with SumSets 0.202 → 0.181; 4,4,1 + SumSets 0.164 → 0.146).
   * e+e- j: it is neutral at 2,2,1 with SumSets (0.123 vs 0.122) and worse without (0.147 vs 0.167). It is better at 4,4,1 + SumSets (0.099 vs 0.086).
   * t t̄: slightly better. The quasi-random rank-1 lattices bring nothing.
6. **`FoldRealOnce` stays experimental** (heavy-tailed H weights, −3% σ on e+e- j).
7. **Folding never pays in CPU per N_eff for these processes:** 1,1,1 stays the cheapest. The options above matter when a negative weight is expensive downstream (shower and detector simulation).
8. **The H events set the floor** (r_H about 0.64 in t t̄, about 0.43 in e+e- j and e+e- → b b̄ b b̄) and are untouched by any folding. Further gains need work on the H side.

Pre-existing bugs fixed on the way:
* `special_check_SoftSing` called `add_wgt` without the orders argument. It read garbage and crashed once FoldSumSets made that path common.
* `weight_lines` grew its arrays one contribution at a time, which is O(n²).
* `add_wgt` now works on copies of its weights instead of writing to its arguments.

With FoldSumSets, the Born cache has to be reset for each set and before the event is written (the Born momenta of different sets differ in the last bits).

