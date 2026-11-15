# Stock O3 `check profile`: g g > g g g g [virt=QCD]

Provisional shared-host timings, not local timing acceptance. Source `0eb109bbb0`
(model `loop_sm`, optimized output, LoopPoleMode -1, 1000 GeV, seed 20261113).
TIRust used `libmadloop_tirust.so` `a01c7639…` (pin 9a781998 with the 512-bit QP
retry). Per-run values: [`profile-gg4g-o3.json`](profile-gg4g-o3.json).

## Commands

```bash
source tools/tirust_env.sh
export FFLAGS='-w -fPIC -O3' LDFLAGS='-lstdc++ -Wl,--no-relax'
# MG commands (one export, all reducers linked; later runs pass -reuse to the same export):
#   set loop_optimized_output True; set output_dependencies external; set ninja/collier/ninjars/tirust <libs>
#   import model loop_sm
#   check profile -reuse 1 g g > g g g g [virt=QCD] --reduction=N --loop_pole_mode=-1 --split_orders=-1 --energy=1000 --seed=20261113
```

Generation, export and compilation took 2.6 h. The default link fails: the loop
wavefunction commons `ml5_0_wl_` (2.75 GB) and `ml5_0_mp_wl_` (5.5 GB;
NLOOPWAVEFUNCS = 51123) place data beyond 2 GB, and the linker cannot convert a
GOTPCREL relocation. `-Wl,--no-relax` (as in `cudacpp_overlay.mk`) links it and
applies equally to every reducer.

Three repetitions. In each, the five reducers ran concurrently, each pinned to one
core in a different L3 cluster, rotated between repetitions. Host load was 190–280
of 384 CPUs.

## Results (median of 3 runs [min–max])

| Reducer | Helicity sum / PS point | Coefs | Loop eval | One-helicity loop eval | Init | Max RSS | Stability (1 pt) |
|---|---|---|---|---|---|---|---|
| NinjaRS | 29.1 s [29.1–29.7] | 28.2 s | 1.02 s | 114 ms [109–120] | 169 s | 2.10 GB | 7.3e-10 |
| Ninja | 29.6 s [29.5–29.7] | 28.4 s | 1.13 s | 178 ms [166–251] | 178 s | 2.10 GB | 3.4e-10 |
| CutTools | 30.0 s [29.6–30.7] | 28.5 s | 1.54 s | 529 ms [519–540] | 251 s | 2.10 GB | 1.4e-8 |
| TIRust | 30.6 s [30.2–31.6] | 28.3 s | 2.32 s | 1.29 s [1.24–1.30] | 156 s | 2.51 GB | 1.2e-10 |
| COLLIER | 31.1 s [30.9–31.7] | 28.4 s | 2.78 s | 1.81 s [1.80–1.81] | 508 s | 2.18 GB | 1.7e-11 |

- 50 of 64 helicities contribute. Coefficient construction takes 91–97% of the
  helicity-summed time, so the five reducers are within 7% of each other.
- The helicity-sum loop evaluation is the difference of two 50-point runs of about
  30 s per point and is uncertain at roughly ±0.5 s. The one-helicity stages (100
  points, coefficients about 0.58 s) separate the reducers' reduction cost cleanly.
  TIRust is 2.4× CutTools, 7× Ninja and 11× NinjaRS, and 0.71× COLLIER.
- TIRust has the lowest initialization time and uses about 0.4 GB more memory.
- Stability is the median rotation-test accuracy of a single point (no unstable point),
  identical across repetitions; it is indicative only.
