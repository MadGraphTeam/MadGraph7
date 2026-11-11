# Successor cache lifecycle qualification proposal

Source-only derivative of the sealed preceding-source two-subprocess protocol.
Production registration: entry-latch-cache-guard-1, revision
0fcd2cd873d61d134837e1d9e87afb13a47824fd; manifest
4a27dda7a0977cd1b0ed0d779b30c403da50968bf494aa96d31cf18fc9fe2076;
exporter inventory afb6e0bea64ea0932751ff4869c206bcea9477d40a39a54bb99d0f06d85b75a6.
The original installed DSO eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9 is unchanged.

Nothing here runs compilation/native physics without an explicit --execute
stage. Root will admit and serialize later actual execution. Fresh private
REPORT .tirust-build/subprocess-cache-successor-results-1 and WORK
.tirust-build/subprocess-cache-successor-work-1 are append-only. No production
source changes, public seal mutations, reference writes or Git mutations.

All stages run from the repository root after source tools/tirust_env.sh:

    python3 -B .tirust-build/subprocess-cache-successor-proposal-1/run.py
    python3 -B .tirust-build/subprocess-cache-successor-proposal-1/run.py freeze --execute
    python3 -B .tirust-build/subprocess-cache-successor-proposal-1/run.py materialize --execute
    python3 -B .tirust-build/subprocess-cache-successor-proposal-1/run.py generate --execute
    python3 -B .tirust-build/subprocess-cache-successor-proposal-1/run.py prepare-build --execute
    python3 -B .tirust-build/subprocess-cache-successor-proposal-1/run.py build --execute
    python3 -B .tirust-build/subprocess-cache-successor-proposal-1/run.py smoke --execute
    python3 -B .tirust-build/subprocess-cache-successor-proposal-1/run.py run --execute

Each stage has an inclusive600s bound, except card-only smoke30s and physical
run750s including four individual180s workers. Ten seconds are reserved for
owned cleanup. Physics acquires the common numerical-tests.lock, uses CPU12
without reservation and one native thread. The actual ELF interpreter, linker
inputs and runtime maps are pinned; all existing external reducers coexist.
Prebuilt vendor reducer artifacts are copied; fresh generated MODEL/DHELAS/
process objects are compiled using --always-make.

Two actual CLI subprocesses: loop_sm, u u~ > z [virt=QCD] @101 and
 d d~ > z [virt=QCD] @202. Four fresh workers: plain/observed x cache0/cache-1.
Each has the unchanged first-checked stopping warmup per prefix (max32), then
19 full physical A/B/A calls covering changed scale, boosts, model mass,
restoration, a real cached seed and context finalization/recreation. Action301
is a clean measured reference;302..319 deliberately set both prefixes' two
software failure latches before each independent call. The called prefix must
reset its own latches, preserve the other prefix, and reproduce clean/restored
results. These latches are software COMMON variables, not hardware IEEE flags.

The observer must see two successful native DP requests per measured physical
call, with primary/check flags[0,1]. Live contexts retain entry and final clears;
after finalize,317 must skip the absent-context entry clear, then create and
register before evaluation and perform the final clear. Seed316 must be seen
and removed at its real entry boundary. All4 workers require exact coefficient,
status, accuracy, physical-input and observed IEEE parity.

Each worker additionally makes28 generated tensor diagnostic requests (14per
precision), with mass/scale/momentum key changes. Keys13 independently recompute
the warmed primary key through CT2/5; cache statistics must show no primary
store hit/insertion (QP bypass+1 only). Keys14 must still hit the primary cache
when enabled. Both numerical values and raw request flags are checked. QP uses
genuine113-to256-bit transport but diagnostic inputs are exact DP promotions.

This bounded scope is real-SM N<=3/rank<=2 native-DP physical amplitudes and
explicit DP/QP tensor cache keys. No physical-QP rescue, rank-six, EW,
loop-induced, poles-only, independent analytic amplitude truth, performance
acceptance, global compiler/platform or whole-program hardware-FPE claim.
The observer is the unchanged before/after IEEE-preserving wrapper from the
preceding packet. Cleanup covers sampled owned process identities, with failure
if any tracked process survives; it is not a proof about rapidly orphaned
unobserved descendants. The original failed/successful packets remain immutable.
