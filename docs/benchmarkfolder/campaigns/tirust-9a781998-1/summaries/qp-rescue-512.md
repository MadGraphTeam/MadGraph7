# Genuine-QP 512-bit retry: stress rerun on the patched bridge

The bridge change "retry genuine-QP Gram refusals once at 512 bits" (vendor
`lib.rs` `15be50e0…`, previously `e5bdb54a…`) was checked against this campaign's
recorded stress lanes. Its compiled stress clones were rerun with the patched
`libmadloop_tirust.so` (`a01c7639…`, previously `6c11cfd9…`) on the same warm-up
and stress points. The clones keep their pre-merge MadLoop source (return-code
format 1). This is a delta check, not a requalification: the final-pin campaign
requalifies the patched bridge.

## Commands

```bash
source tools/tirust_env.sh; export FFLAGS='-w -fPIC -O3'
printf 'install tirust --prefix=%s/HEPTools --jobs=16\nquit\n' "$PWD/.tirust-build/q-9a781998-qp512" > install.mg5
python3 bin/madgraph install.mg5          # smoke --require-poles passes
python3 vendor/tirust/check_fixtures.py <lib>/libmadloop_tirust.so <tirust>/crates/iregi-rust/tests/fixtures/method_choice_lanes.json [--qp]
python3 -B tools/tirust_stress_rerun.py --campaign tirust-9a781998-1 --library <lib> --out <out>/<case> --case <case> [lanes]
```

The bridge's `cargo test --release` passes 19 tests with `poles-only` and 17
without, including the two new tests: the 512-bit retry of a captured H H box,
checked against 1024 bits, and pole-only parity. The pinned C-ABI fixtures pass
27 DP and 54 QP lanes with values identical to the previous bridge.

## Results

Lanes: TIRust forced QP (`candidate-9-qp`), TIRust automatic (`candidate-9-auto`)
and the `9|6|7|1` chain, for the 9 stress cases (38 points). For CMS, only the two
TIRust lanes were rerun, because CutTools hangs at `forward-1e-08` in the chain.
Every lane and point not listed below is bit-identical to the recorded run
([`qp-rescue-512.json`](qp-rescue-512.json)).

| Case | Point | Lane | Before | After |
|---|---|---|---|---|
| li_gg_hh | stress-small-gram | TIRust forced QP | QP failure latch; incomplete sum (3.53e-6, about 11× too small) | 3.98229627107805e-5, no latch. Identical to NinjaRS QP; agrees with CutTools QP and Ninja QP |
| ew_n3_epem_ttxa | stress-threshold | TIRust automatic | refused (420) | accepted QP rescue (323) |
| ew_n3_epem_ttxa | stress-threshold | chain 9\|6\|7\|1 | rescued by Ninja (328) | TIRust QP rescue (323) |
| ew_n3_epem_ttxa | stress-threshold | TIRust forced QP | QP failure latch | no latch, same value as automatic |
| ew_n3, ew_n4, qcd_n3, qcd_n4 | soft-collinear, QCD thresholds | TIRust forced QP and automatic | latched / refused | still latched / refused; the incomplete forced-QP sums change |

At the t t̄ γ threshold, the checked reducers disagree with each other, so the
stress tool has no reference there. The new TIRust value (finite part
6.002244658e11) agrees with NinjaRS QP to 1.1e-9, with CutTools QP to 3.7e-6, and
with Ninja QP to 1.7e-5 in the finite part (a pole component differs more). The
chain now accepts this value in place of Ninja's.

**Stress bar.** No accepted TIRust value disagrees with a reference. TIRust now
accepts 28 of 38 points and refuses 10:
- 7 where every reducer fails;
- 2 the chain rescues (EW μ⁺μ⁻ small Gram, CMS `forward-1e-06`);
- CMS `forward-1e-08`.
