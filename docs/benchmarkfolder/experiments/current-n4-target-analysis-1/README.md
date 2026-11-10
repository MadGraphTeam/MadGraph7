# Closed current n4 tighter-target analysis

Only the published ordinary-matrix capsule `8f5f7110…4496be` supplies numerical
inputs. The exact retained result hashes and parser hash are in `analysis.json`.
The capsule replay, each lane's full stored comparison, and both identical
answer orders were checked before counting unique physical points. Baseline
and current CutTools values, statuses, and accuracy fields match bit for bit.
All six lanes in both cases still pass the registered `3e-6` criterion.

Counts below use `abs(a-b)/abs(a+b)` and the frozen negligible-pole rule at
`1e-8`. Each physical-point miss appears in two identical answer-order rows.

| Case | Backend | Finite misses | Single-pole misses | Double-pole misses |
|---|---|---:|---:|---:|
| QCD n4 | Ninja | 44 DP | 0 | 0 |
| QCD n4 | COLLIER | 40 DP | 0 | 0 |
| QCD n4 | NinjaRS | 40 DP | 0 | 0 |
| QCD n4 | TIRust | 5 DP + 34 QP | 0 | 0 |
| EW n4 | Ninja | 5 DP | 1 DP (69) | 0 |
| EW n4 | COLLIER | 4 DP | 0 | 1 DP (38) |
| EW n4 | NinjaRS | 5 DP | 1 DP (69) | 1 DP (69) |
| EW n4 | TIRust | 1 DP + 3 QP | 0 | 0 |

Born has no misses. QCD n4 has 45 finite-miss points in the union, with 38
common to all four non-CutTools backends. Its largest baseline discrepancy is
COLLIER point 61 (`4.150911600e-7`); the largest non-CutTools pair discrepancy
is Ninja/COLLIER point 89 (`2.166777164e-7`). All six non-CutTools finite pairs
are below `1e-8` at 91 points. COLLIER/TIRust exceed it only at point 93
(`1.100865060e-8`, accepted DP/QP).

EW n4 has common finite misses at points 14, 81, 85, and 87; Ninja and NinjaRS
also miss at 69. The largest discrepancy is NinjaRS's single pole at 69
(`1.526551658e-7`); the worst finite discrepancy is at 81 (`1.143672333e-7`).
At 69 the finite results form two close pairs, Ninja/NinjaRS and COLLIER/TIRust;
their largest separation is `1.579171974e-8`. All six non-CutTools finite pairs
are below target at the other 99 points.

These are discrepancy patterns, not independent truth determinations. DP/QP
labels identify accepted host output; internal attempts and hardware IEEE state
are unobserved. Further investigation needs separately admitted independent
higher-precision evidence with common effective inputs. No native evaluation,
prior private QCD n4 diagnostic, or source/card/policy change enters this packet.

Verify the packet hashes and deterministically recompute both JSON files:

```sh
python3 -B docs/benchmarkfolder/experiments/current-n4-target-analysis-1/analyze.py --verify
```
