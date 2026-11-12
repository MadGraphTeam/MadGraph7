# Hua-Sheng's local timing TODO — signed OEM source

Precise timings remain **TODO on Hua-Sheng's local machine**. The new
[`tirust-oem-signed-1` manifest](../manifest.json) has SHA256
`cd6bca8e34198963b0656bdad19a822a1df15d846c5f14ee06d469fe4f385eec`.
Its [initial pending summary](initial-summary.json) is an immutable copy of the
unmeasured registration summary, SHA256
`f8690d18af61084e583cb25c3fa3d23792bd8cf67feb0257d670daa95cfe65eb`.
Registration copies inputs and source snapshots; it does not generate, compile
or execute a numerical process. The metadata-only summary used `--cpu 12` to
avoid a CPU-selection probe; it records no measured affinity or reservation.
Choose and register an allowed CPU on the local machine below.

This successor binds MadGraph source commit
`e2023308563020688c553e0970c6e37ad607a570`, exporter inventory
`14e782c6a07a213e1ff73fdce9e816dd907d571fc684da9711d5bc970976a8d2`,
and numerical identity
`59e8206d60bf1b21bf85cbd8ea40b6e9dde74b2943ec07001278f2da6e46338a`.
Its signed TIRust pin is `10ce34ed8ad5b6f5f989a0e220eb3a244a71d1a1`.
The registered installed DSO SHA256 is
`85ee0cdf860e5ad4c7a5c26163802673de22404690a10c7c53fb73ebe1ff5bb0`.
Installation qualification is separate from process numerical/runtime acceptance.
The separate M1 reference source is not this installed runtime. Later
documentation-only commits may retain this exact exporter inventory; changed
generator, bridge or native bytes require a new verified binding.

All eleven cases require fresh local numerical qualification: baseline CutTools
and candidate reducers 1/6/7/8/9, each on the exact saved 100 physical points.
The eight agreed main processes retain baseline
`9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f`; mixed QCD/QED and loop-induced
controls remain in the frozen configuration. The predecessor's physical inputs,
model pin, cards and policy are inherited; its 66 ordinary passes and timing
measurements do not qualify this successor. Every new numerical lane and precise
runtime acceptance item is **PENDING**. The old 44-page PDF, capsules and frozen
preceding handoff remain unchanged historical evidence.

Run from the updated checkout root on supported GNU/Linux with the pinned Nix
development shell. First reproduce the comparison libraries using the
[comparison-backend instructions](../../../qualification/comparison-backends/README.md#reproduce-the-installations).
NinjaRS retains its independent scalar-library revision. Keep all declared
predecessor/failure and physical-card ancestry files in the delivered checkout.
Use fresh installation, measurement and render IDs, then run:

```bash
nix develop
source tools/tirust_env.sh
python3 -B tools/tirust_benchmark_local.py install \
  --campaign tirust-oem-signed-1 --installation hs-oem-signed-native-1 \
  --tirust-prefix HEPTools/local-oem-signed-native-1/tirust --jobs 4
OEM_LOCAL_CPU=$(python3 -B -c 'import os; print(min(os.sched_getaffinity(0)))')
python3 -B tools/tirust_benchmark_local.py prepare \
  --campaign tirust-oem-signed-1 --measurement hs-oem-signed-local-1 \
  --tirust-prefix HEPTools/local-oem-signed-native-1/tirust \
  --install-receipt docs/benchmarkfolder/local-installations/hs-oem-signed-native-1/receipt.json \
  --cpu "$OEM_LOCAL_CPU"
python3 -B tools/tirust_benchmark_local.py verify --measurement hs-oem-signed-local-1
python3 -B tools/tirust_benchmark_local.py forward --measurement hs-oem-signed-local-1 init
python3 -B tools/tirust_benchmark_local.py forward --measurement hs-oem-signed-local-1 model
python3 -B tools/tirust_benchmark_local.py forward --measurement hs-oem-signed-local-1 build --jobs 4
python3 -B tools/tirust_benchmark_local.py forward --measurement hs-oem-signed-local-1 numerical
python3 -B tools/tirust_benchmark_local.py forward --measurement hs-oem-signed-local-1 campaign
python3 -B tools/tirust_benchmark_local.py forward --measurement hs-oem-signed-local-1 summarize
python3 -B docs/benchmarkfolder/build_local_report.py \
  --measurement hs-oem-signed-local-1 --render-label render-1
```

The actual CLI installation receipt binds newly built local native bytes.
`prepare` preserves the frozen baseline record and creates isolated local roots;
do not substitute unscoped core `tirust_benchmark.py init`, which can rewrite the
old root baseline record on a fresh checkout. These local installation,
preparation, numerical and rendering commands still need actual execution against
this successor; this handoff does not claim that they have passed locally.

Compare Born, finite and both poles using the frozen `3e-6` criterion and retain
`1e-8` target misses. Keep points, cards, COLLIER pole overrides, compiler flags,
helicity/color sums, cache policy and natural rescue/stability checks matched.
Precise performance requires at least seven alternating warmed pairs with
approximately one-second batches; report initialization separately. Every main
TIRust-first versus materialized-baseline-default-chain comparison must have
upper 95% paired runtime-ratio bound `<=1.05`; existing-backend regressions are
separate. Retain refusals and failed or incomplete measurements. Separate
native-DP/rescue-cohort timings remain an additional TODO until the local cohort
adapter is qualified. Timing success does not qualify unresolved stress accuracy
or complete the overall integration goal.
