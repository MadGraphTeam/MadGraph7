# Hua-Sheng's final local validation TODO — native tensor tiers

**Final numerical and runtime acceptance is pending.** This checklist selects
campaign `tirust-native-tiers-1`, registered for source `77eb57b82d4858ff302fa9870778e01cc86d7326`.
Its manifest SHA256 is `1ed7ff89800de0a1ec361c15db3795adb2e4173bd52d404fd36f2efb0d62b0dc`; registration freezes inputs and
source identities without executing processes. The [manifest](campaigns/tirust-native-tiers-1/manifest.json)
and [initial pending summary](campaigns/tirust-native-tiers-1/timing-handoff-1/initial-summary.json)
retain that registration. No predecessor passes transfer. Use this source and
its newly built local libraries for all runs below.

The immediate physical-input predecessor is the measured
[`entry-latch-cache-guard-1`](campaigns/entry-latch-cache-guard-1/manifest.json),
which has actual retained card observations for all eleven cases. The
[earlier signed-source registration](campaigns/tirust-oem-signed-1/manifest.json)
is a separate reference: it has no measured observations to inherit. Its
handoff and the measured predecessor's numerical passes do not qualify this
new source.

The native-tiers implementation was pushed in `8b74cdbb3e106f2859514bb4ca95d9fe5d70b370`.
Its qualified shared-host DSO is
`d427667a0f2f55d0bd96778c1259ffcf52e3925341890b81c38f5f94e1eb7052`
(16 native and 67 integration tests). A local rebuild has its own recorded
artifact identity. The installer still pins signed TIRust
`10ce34ed8ad5b6f5f989a0e220eb3a244a71d1a1`; OEM activation is also published
on upstream main `de60dc480fec6475d2373a533c64d1c40ce23b2e`. No automatic
backend defaults change. The overall integration goal remains incomplete.

Start with a fresh local checkout and check that its source matches this
registration. Run the remaining commands from that checkout's root:

```bash
git clone --branch madloop_tirust https://github.com/MadGraphTeam/MadGraph7.git madgraph7_madloop_tirust
cd madgraph7_madloop_tirust
git rev-parse HEAD
nix develop
source tools/tirust_env.sh
export FFLAGS='-w -fPIC -O3'
```

Export these flags before installation, local preparation and either generated
build, and keep them set for every later command below. The workflow forwards
`FFLAGS` to both baseline and candidate, records the local deployment and
effective compiler recipes, and checks their equality. Verify `-O3` for
MODEL, DHELAS, process code and drivers in both arms. Use fresh local roots;
changing flags does not optimize old objects or relabel earlier measurements.

A later documentation-only commit may preserve the registered exporter hash;
changed generator, bridge or measurement protocol bytes require a new reviewed
binding. The local wrapper verifies exporter/native identity and freezes its
local protocol files; also retain the recorded source commit and those seals.
Reproduce all comparison libraries with the pinned
[comparison-backend instructions](qualification/comparison-backends/README.md#reproduce-the-installations).
Keep NinjaRS's independent dependency pins. A normal interactive upgrade is
`./bin/madgraph`, followed by `install tirust --update` and `quit`; it persists
the `tirust` library path. For the acceptance run, the receipt-producing command
below invokes that same CLI in a **fresh prefix**, binding actual source and
native bytes. Use fresh installation, measurement and render IDs each time.

```bash
python3 -B tools/tirust_benchmark_local.py install \
  --campaign tirust-native-tiers-1 --installation hs-native-tiers-install-1 \
  --tirust-prefix HEPTools/local-native-tiers-final-1/tirust --jobs 4
TIRUST_LOCAL_CPU=$(python3 -B -c 'import os; print(min(os.sched_getaffinity(0)))')
python3 -B tools/tirust_benchmark_local.py prepare \
  --campaign tirust-native-tiers-1 --measurement hs-native-tiers-final-1 \
  --tirust-prefix HEPTools/local-native-tiers-final-1/tirust \
  --install-receipt docs/benchmarkfolder/local-installations/hs-native-tiers-install-1/receipt.json \
  --cpu "$TIRUST_LOCAL_CPU"
python3 -B tools/tirust_benchmark_local.py verify --measurement hs-native-tiers-final-1
python3 -B tools/tirust_benchmark_local.py forward --measurement hs-native-tiers-final-1 init
python3 -B tools/tirust_benchmark_local.py forward --measurement hs-native-tiers-final-1 model
python3 -B tools/tirust_benchmark_local.py forward --measurement hs-native-tiers-final-1 build --jobs 4
python3 -B tools/tirust_benchmark_local.py forward --measurement hs-native-tiers-final-1 numerical
python3 -B tools/tirust_benchmark_local.py forward --measurement hs-native-tiers-final-1 campaign
python3 -B tools/tirust_benchmark_local.py forward --measurement hs-native-tiers-final-1 summarize
python3 -B docs/benchmarkfolder/build_local_report.py \
  --measurement hs-native-tiers-final-1 --render-label render-1
```

`prepare` creates isolated local roots and preserves the frozen baseline record.
Keep the delivered input, model, predecessor and physical-card ancestry files.
The earlier signed handoff and its old installed DSO do not qualify native tiers.

Complete all eleven registered processes: 100 deterministic physical points per
case, baseline CutTools plus candidate backends **1, 6, 7, 8 and 9**, giving
**66 fresh numerical lanes**. Compare Born, finite, single and double poles at
`3e-6`, retaining separate `1e-8` target misses. The eight main runtime gates are:

| Final particles | QCD (`loop_sm`) | EW (`loop_qcd_qed_sm_Gmu`) |
|---|---|---|
| 1 | `u u~ > z` | `u d~ > w+` |
| 2 | `g g > t t~` | `e+ e- > mu+ mu-` |
| 3 | `u u~ > z g g` | `e+ e- > t t~ a` |
| 4 | `u u~ > z z g g` | `e+ e- > mu+ mu- a a` |

The frozen cards select the corresponding virtual corrections. Mixed QCD/QED
`u u~ > g a` and loop-induced `g g > h`, `g g > h h` remain controls. Preserve
models, points, flags, helicity/color sums, full finite terms, cache settings,
COLLIER UV/IR pole overrides, and natural stability/rescue policy.

For **each of all eight main processes**, compare the TIRust-first chain with
the materialized default chain of baseline
`9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f`. Require at least seven alternating
warmed pairs with approximately one-second-or-longer timed batches and upper
95% paired candidate/baseline ratio **≤1.05**. Record initialization separately.
Explicit reducer timings and separate measured existing-backend regression
controls are also required. The normal campaign proceeds to them after the
main chain gates pass. If a main gate fails, the deferred controls remain open
requirements; that failure does not waive their measurement. Inspect every
backend regression independently and retain failures, incomplete runs and every
missing or refused sample.

Separate native-DP and rescued-cohort timing remains an **adapter TODO**. Final
status `219` now identifies accepted DP input/output transport; internal tensor
rescues may use 106, 256 or 512 bits. It cannot classify a point as native DP.
Report primary/check route evidence, attempts, refusals, censoring and unknown
coverage separately; batch counters do not assign individual points or observe
Fortran coefficient-cache reuse. Status `323` establishes QP transport, while a
QP numerator claim also requires the verified effective card
`UseQPIntegrandForTIRust=True`; `False` selects copied DP numerator coefficients.
Freeze cohort point IDs/counts with a qualified per-call route adapter, keep
natural rescues and checks, and include unsuccessful DP work. Forcing QP is a
separate measurement. Do not infer whole-amplitude arithmetic from transport.

These cohort diagnostics do not replace full-stream acceptance. Shared-host
pilot gains, earlier campaigns, and installation tests do not close these new
runtime gates or unresolved EW/soft-collinear/complex-mass stress accuracy.
Publish the local numerical summaries, full usage evidence, paired intervals,
failures and reproduced report before marking any corresponding gate complete.
