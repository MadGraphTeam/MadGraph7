# Recovered evidence for the three post-report optimization commits

The commits b350286280 (exact-input tensor cache, retained rescued subintegrals,
check store, `tier_margin_bits=0`), ab31cdf840 (pooled auxiliary executors) and
b71c59086d (pin 9a781998, lazy master evaluators, `double_double_headroom_bits=26`)
quote test and replay figures from logs that the Claude session "Speed up TIRust
in MadLoop and its initialization" kept under `/tmp`. On 2026-11-13 they were
copied to this host's ignored build directory,
`.tirust-build/recovered-450e7a-20261113/`. The copy has 763 files, about
2.8 GB including two call traces, and a `SHA256SUMS` manifest
(`scratchpad/EVIDENCE.md` is that session's index). The table below maps each
quoted figure to its log. Figures marked *re-run* were reproduced here with
[`tools/tirust_capture`](../../../../tools/tirust_capture/README.md).

| Claim | Log (under the recovered directory) | Status |
|---|---|---|
| Bridge unit tests 16/16 | `tasks/bj23p2y6n.output` | 16 passed, 0 failed |
| TIRust suites 710 / 711 / 712 / 713 | `scratchpad/suite{,2,3,4}.log` | 47/47/47/48 suites, 0 failed, 31 ignored each |
| 464,000 replayed calls (12 default-card Zggg points), every tensor unchanged | `scratchpad/ml/out/trace/def9.bin`; `scratchpad/rp/dh-def9-{40,26}.txt` | 463,901 bit-identical in both headroom variants; the rest differ by ≤ 4.9e-29 from the older recording |
| 53,200 stock-profile calls; 53,198 bit-identical between DD and 256-bit, the other 2 ≤ 3e-28 | `scratchpad/prof/trace/t.2483917`; `scratchpad/rp/dh-prof-{40,26}.txt` | *re-run*: the headroom-40 replay reproduces the recording bit for bit; headroom 26 differs on 2 calls by ≤ 5.4e-29 (per-degree metric) |
| 30-point MadLoop table for the five reducers (b350286) | `scratchpad/ml/out/final1/*.{dat,log,time}` | retained, not re-run |
| Concurrent stock O3 `check profile` loop estimates (b71c590) | `scratchpad/prof/outputs/{cuttools4,ninja4,collier4,ninjars4,tirust4}/` | retained; FULL−SKIP estimates, not acceptance timings |

The re-runs also give a direct 256-bit reference for every lane the double-double
tier accepts. They use a replay with `double_double_headroom_bits=105`, so no
lane stays in DD:

- **Profile trace (old 2^-66 gate):** 1,261 DD calls; maximum per-degree
  difference 5.2e-22.
- **Default-card trace:** 423 DD calls under the production settings; maximum
  difference 7.4e-28.

These are single-process Zggg (`u u~ > z g g g`) samples. Qualification of the
headroom-26 gate across all eleven processes and the stress points uses traces
captured from this campaign. See the campaign README.

The `check profile` loop estimates swing strongly between runs: NinjaRS
4–59 ms and COLLIER 292–361 ms across five repeats. Use paired runs only.
