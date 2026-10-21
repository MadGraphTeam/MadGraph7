# Final pinned core correctness

The actual CLI exports generated at `856e76e669618ec8d0f799b9c8e6f86819dcd986`
passed all eight fixed 100-point cases. Each case has one calibration and two
actual repeated traversals, with 200 Born/Laurent split-order rows per traversal.
All repeated rows are exactly equal. The 800 case-points have 768 native-DP
results (`218`) and 32 QP-rescued results (`322`), identically in every traversal.

The publication authority is `dataset-correctness-only-public-v2.json`. Verify
it from the repository root with the retained development environment:

```sh
source .ninjars-build/dev-env.sh
python3 -B tools/ninjars_benchmark_policy_v3.py verify \
  --dataset docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/core-correctness/dataset-correctness-only-public-v2.json
```

The unchanged reader validates source, full runtime, cards, all fixed momenta,
all answer orders and the two repeated traversals against the retained numerical
references. `independent-ninja-audit-v1.json` independently checks only the
requested Ninja6/NinjaRS8 comparison: 19,200 coefficients, no failures. The root
orchestrator separately reported checking six top-level links and all 24 raw
record hashes; that checksum check is not an independent numerical calculation.

`run-v1/` and `terminal-correctness-v1.json` preserve the original measured
records. `public-source-view-v2.json` records the exact publication-only delta:
eight copied source proofs replace 16 private generation/compilation paths with
byte-identical public copies. Original proofs, binding and dataset remain intact;
all scientific case conclusions are identical. The public view requires no
private generated-process tree.

Retained verification still requires the exact installed native archive
`b0d69deb8c16deb13293176296afc5e18dbeeebe778a20fa39b61804fba89d4c`
and its bound configuration/Cargo metadata. Small `runtime-assets/` snapshots
preserve the full linked-runtime ABI evidence, including compressed GFortran
module metadata. Large native libraries, executables, objects and build caches
are not publication inputs to stage here. A fresh rebuild with different archive
bytes must establish a new explicit measured cohort.

This is correctness evidence only. Initialization includes unreported warm calls;
the 2,400 retained calibration/repeat point evaluations are not the total number
of calls made during initialization. Correctness timings, including the disclosed
external diagnostic sample during QCD4, do not qualify as performance evidence.
Every seven-pair timing gate, loop-induced qualification, focused/stress controls,
actual MG7 CLI profile and final report remains separately required. No full-goal
or runtime-performance passage is claimed by this capsule.
