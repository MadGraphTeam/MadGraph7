# Source-only guarded adaptive build with a reviewed longer deadline

Proposal 2 exhausted its absolute 600-second deadline during bridge
compilation. The controller retained its timeout, killed/reaped its owned
child group, and recorded Cargo return −9. Total recorded duration was
600.022 seconds. The private guarded core had compiled, but there was no
successful bridge DSO. This is a deadline failure; it is not a host OOM
diagnosis or a completed build.

Its original failed receipt remains unchanged, including the incomplete
source-closure field. A separate read/hash-only post-failure receipt checks
all 1,186 original protected bindings without drift. Both records and
raw logs are retained under `failed-attempt2`. The earlier 4 GiB allocation
failure and its separate 1,163-binding closure remain under
`failed-attempt1`. Neither failed attempt is rewritten as a success.

**This proposal has not compiled, linked, loaded or evaluated anything.**
Root deliberately requested a new reviewed attempt with a **1,800-second
total deadline** after observing the timeout. It is not an automatic retry.
No build begins before separate root review and admission.

All 499 private source files are byte-identical to proposal 2. Numerical
source, Cargo manifests, locks, dependency revisions, compiler binaries,
environment, flags, release opt-level 3, thin LTO, codegen-units 1 and
one-job/thread settings remain unchanged. The only numerical-source delta
from the original adaptive release `c5de24e6…` is still the reviewed Gram
signed-zero guard. No NinjaRS pin, reference, vendor, production file,
existing generated process or numerical campaign is changed.

The new namespace is
`.tirust-build/cayley-guarded-adaptive-build-proposal-3`. Its source,
build-cache, target and attempt folders are separate and refuse overwrite.
The original `.tirust-build/cargo-build/release` cache is privately copied
using `cp --archive --reflink=auto`; neither failed guarded cache is reused.
Actual fresh guarded core and bridge artifacts from the new source
manifests remain mandatory.

Resource limits stay **8 GiB virtual address space per Cargo/compiler
process** and **4 GiB per cache-copy process**, with nonexclusive CPU11
affinity, one Cargo job, offline/locked mode and zero core dumps. These are
per-process address-space caps, not aggregate RSS measurements or a CPU
reservation. The 1,800-second aggregate alarm covers preflight, copying,
compilation, hashes, final source closure and successful receipt publication.
The imported existing supervisor includes SIGALRM in its in-memory atomic
spawn/reap deferral; its source file is unchanged. Failure-only metadata
can be retained after expiry and cleanup and cannot qualify success.

After root admission, from the repository root:

```sh
source tools/tirust_env.sh
python3 -I -B docs/benchmarkfolder/experiments/cayley-guarded-adaptive-build-proposal-3/build.py --root-admitted
```

`plan.json` binds the exact commands, tool files, environments and both
failed ancestors. The controller would copy and hash its successful DSO
inside the new `attempt1/runtime` folder without loading or installing it.
The local `attempt1` name belongs only to proposal 3.

Read-only source/provenance verification and synthetic ownership controls:

```sh
python3 -I -B docs/benchmarkfolder/experiments/cayley-guarded-adaptive-build-proposal-3/verify_source.py
python3 -I -B docs/benchmarkfolder/experiments/cayley-guarded-adaptive-build-proposal-3/test_alarm_ownership.py
```

The physical replay proposal targeting failed proposal 2 remains unexecuted
and cannot bind that receipt. A separate binding version may target proposal
3 only after this source is reviewed and an actual successful DSO receipt
exists. No loader, physical replay, installation or source promotion is
authorized by this build preparation. Physical accuracy, checked/trapped
DP/QP behavior, per-call precision and performance remain unqualified.
