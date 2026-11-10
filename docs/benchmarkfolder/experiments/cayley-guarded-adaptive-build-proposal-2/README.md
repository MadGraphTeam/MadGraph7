# Source-only second guarded adaptive build proposal

The first private build failed while compiling the bridge: LLVM reported
`out of memory / Allocation failed`, then `SIGABRT`; Cargo returned 101.
Its child was reaped after 193.177 seconds, with total controller duration
201.192 seconds. The guarded core had compiled, but no bridge DSO was
produced. This is a compiler allocation failure under its 4 GiB virtual
address-space cap; it does not establish a host OOM kill.

The original failed receipt remains unchanged, including its incomplete
source-closure field. Its hash is
`75841160a744f3de466fa4ef219e0a431b24c694a7fdf25278d09d49d80f2079`.
A separate read/hash-only closure subsequently verified all 1,163 original
bindings without drift. Both records and raw logs are retained under
`failed-attempt1`; the completed closure does not convert the failed build
into a successful attempt. Compiled core artifacts are recorded by hash
only and are not reused by the proposed second build.

**No compiler, linker, loader, reducer, or amplitude has been executed for
this second proposal.** Root review and admission are required before its
one attempt. The first source proposal, failure, and earlier tiny-fixture
proposal remain unchanged.

This uses the same 499 source files and guard as
[proposal 1](../cayley-guarded-adaptive-build-proposal-1/README.md), copied
byte-for-byte into the new private working folder
`.tirust-build/cayley-guarded-adaptive-build-proposal-2/sources`.
All Cargo manifests, locks, dependency pins, compiler binaries, release
flags and job/thread settings remain identical. The base is still the
adaptive release DSO `c5de24e6…`; only the previously reviewed Gram guard
differs in its executable source. NinjaRS and production remain untouched.

The only build-resource change is **8 GiB virtual address space per Cargo
build process**, inherited by its compiler/linker children. Cache copying
keeps its 4 GiB cap. The attempt retains CPU11 nonexclusive affinity, one
Cargo job, offline/locked mode, release opt-level 3, thin LTO and
codegen-units 1. This is not an aggregate RSS limit or a CPU reservation.

Both cache and output folders are new and refuse overwrite. The cache
seed is the **original** `.tirust-build/cargo-build/release`, copied using
private `cp --archive --reflink=auto`; the failed guarded build cache is
not reused. Actual fresh Cargo records from the new private guarded core
and bridge are required. The one absolute 600-second budget covers
preflight, copying, compilation, source closure and successful receipt
publication. Failure-only metadata retention may finish after expiry
and reaping; it cannot qualify a success. There is no automatic retry.

After separate root admission, from the repository root:

```sh
source tools/tirust_env.sh
python3 -I -B docs/benchmarkfolder/experiments/cayley-guarded-adaptive-build-proposal-2/build.py --root-admitted
```

The controller reuses the existing reviewed process-group supervisor.
Before invoking it, proposal 2 adds `SIGALRM` to the imported helper's
in-memory deferred-signal tuple. This covers the absolute alarm during
the helper's atomic child creation/ownership and cleanup/reaping regions;
the helper's on-disk source remains unchanged. The original outer alarm
was missing from that tuple. Three synthetic tests cover an alarm between
spawn and ownership, an alarm during reaping without repeated stale-PID
cleanup, and a negative control that reproduces the original gap. Every
child and kill operation in these tests is a mock; no real child is spawned
by the tests. The pre-correction source index and changed text are retained
under `ownership-correction`.

`plan.json` contains exact executable paths, hashes, environment and
commands. Successful output would be hashed and copied to
`attempt1/runtime` inside this new proposal's private working folder,
without loading or installing it. The local attempt name `attempt1`
belongs to proposal 2 and does not overwrite proposal 1's attempt.

Read-only preflight:

```sh
python3 -I -B docs/benchmarkfolder/experiments/cayley-guarded-adaptive-build-proposal-2/verify_source.py
python3 -I -B docs/benchmarkfolder/experiments/cayley-guarded-adaptive-build-proposal-2/test_alarm_ownership.py
```

The verifier also checks the exact resource/output-only command delta
from proposal 1 and reparses its retained failure/closure distinction.
A later checked/trapped and forced-QP physical replay must bind an actual
successful new DSO receipt and needs separate admission. No presumed DSO
hash, physical correctness, DP/QP qualification, performance acceptance,
installation, or source promotion is claimed here.
