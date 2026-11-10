# Final implementation readiness, with registration held

This new snapshot binds the implementation at
`059bb9c979194bfda783cad83443223def8df280`, the observed checkout revision,
and the actual corrected package at
`.tirust-build/qp-transport-install/mg/HEPTools/tirust`. The later observed
checkout commit contains documentation/evidence changes; the inventory requires
no changed exporter files relative to the production commit. Earlier plan1 and
plan2 remain unchanged, including their earlier package identities and scope.

The implementation combines split primary/check DP workspaces, guarded terminal
rank pairs `(1,1)`, `(2,1)` and `(3,1)`, and cache invalidation after a failed
Fortran QP transport. The package's DSO SHA256 is
`eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9`,
its Fortran archive SHA256 is
`0c86349a4fd0dadd918476731e899529ce1c7b17f5892a4af2c22dd762250c1a`,
and its configuration SHA256 is
`25182cfb31e68f8ff6246717e6819f0ec86232de88742f6e47e73676e6a41af9`.
The installed interface/module, link metadata and Cargo inputs are also bound.
The private Born fallback, 512-bit bridge and scalar-shadow proposals are
excluded.

**This is not a campaign registration or permission to launch.** The observed
installation record is a frozen copy of the publication check at preparation
time. Its qualification state is retained as observed, even if later independent
qualification succeeds. Final package evidence and explicit coordinator
clearance remain prerequisites to registration. No source compilation, relink,
amplitude call or native reducer call is performed by this inventory.

## Baseline and candidate preparation

All eleven original baseline cases passed private untouched-driver byte-identical
relink preflights. Their independently audited index is bound as
`f839e57f8d66a84c2060d38d4876717f221e2debefacfb434c46443a9dbd561c`.
These readiness proofs do not import a cached approval into a future campaign:
`reuse-baseline` still repeats current source/object/dependency/compiler checks,
the untouched-driver link proof, private copying and worker-only relinking.
Original baseline objects, cards and libraries remain unchanged. Their original
export seed was unrecorded and remains unrecorded.

After clearance, generate all eleven candidate exports through the actual MG
CLI with `PYTHONHASHSEED=0` and `--no_save` configuration commands, then build
their numerical objects from generated sources. Do not reuse historical
candidate objects. The new manifest and generation records bind the seed;
metadata alone cannot reset an earlier failed performance gate.

The proposed registration command, **not executed by this snapshot**, is:

```sh
source tools/tirust_env.sh
python3 tools/tirust_benchmark.py register-campaign \
  --campaign split-dp-terminal-1 --predecessor legacy \
  --tirust-prefix "$PWD/.tirust-build/qp-transport-install/mg/HEPTools/tirust" \
  --reason 'Split primary/check TIRust DP workspaces, guarded DP terminal rank pairs 1x1/2x1/3x1, and invalidate caches after failed Fortran QP transport' \
  --worker-timeout qcd_n4_uux_zzgg=14400 \
  --worker-timeout ew_n4_epem_mupmumaa=14400
```

The other cases retain 3,600-second administrative worker limits. These bounds
do not shorten streams, timed batches or repetitions. The actual manifest
seals the approved source, package, points, policy and deadlines before launch.

## Numerical qualification and timing order

Complete baseline CutTools and explicit candidate backends 1, 6, 7, 8 and 9 on
all 100 frozen points of every one of the eleven cases. Untimed warmup-only
preparation may use the final complete steady traversal; partial streams remain
unqualified. Preserve coefficient mismatches, failures and actual accepted
backend/precision counts. The numerical prerequisite is separate from timing
and must not be inferred from passing rescue-chain measurements.

Keep the frozen policy: `MLStabThres=1e-6`, Init1/Run−1, forced host stability
checks, finite and UV/IR coefficients, physical cards, QP-integrand and cache
settings, and both UFO models. The recorded materialized baseline chains are
`8|6|7|1` for main/mixed cases and `7|6|1` for loop-induced controls. The candidate
prepends backend 9. Earlier accepted results and failed gates remain retained
under their original identities.

After numerical validation, owned compilation/diagnostics end and host
conditions permit a quiet allowed CPU, run all eight main rescue-chain gates
before expensive existing-backend regressions and controls. Every main gate
uses seven alternating pairs and the complete 100-point stream, with sustained
calibration toward two-second batches and an upper 95% paired ratio bound of
1.05. Any failed, incomplete or ineligible main gate remains reported, returns
nonzero and defers the later phase. All selected main gates are still attempted.
Main existing backends 1/6/7/8 retain their regression gates; standalone backend9
and control speed ratios are descriptive, with numerical/duration requirements.

The retained scheduling estimate has about 17 hours of known main timing work
but omits unobserved heavy lanes, candidate-chain costs, preparation and queues.
It is neither a completion forecast nor an upper bound. Plan for hours of
preparation and potentially multiple days if the complete timing campaign can
proceed. No kernel or diagnostic speedup qualifies a full-amplitude gate.

## Read-only reproduction

From the repository root, `python3 .../inventory.py` prints a fresh inventory
without writing a campaign. It verifies the actual package and retained
baseline preflight ledger, and checks current baseline source/executable bytes.
It does not repeat every live object/compiler/dependency proof; those were
audited in the linked preflight and are repeated by final reuse. A future HEAD
or timestamp may differ from this snapshot even when implementation bytes
remain identical. The local `observed-installation.json` is intentionally frozen.
