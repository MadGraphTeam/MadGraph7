# Retained failed component qualification

Attempt 1 completed its bounded compiled fixture, but **parity failed**. The
controller returned 1 because its completed record has `passed=false`. All 124
owned commands were reaped; 60 returned zero and 64 terminated with SIGFPE.
All 137 protected input identities matched before and after the attempt. The
frozen 15-file source proposal and its index remain unchanged; their statements
that execution had not occurred describe the earlier proposal stage.

Each of the four combinations (historical flags or supplemental O3, binary64 or
binary128) contained 1,344 original/candidate pairs. Each had 120 failures,
solely in IEEE state, for direct matrix route 2 with an upper or lower signaling
NaN (cases 7 and 8). All matrix-policy, caller-input-word, halting-mode and
rounding checks passed in that bulk corpus. The observation boundary and its
limitations are exactly those recorded in the frozen fixture README.

Of 48 supported trap pairs, 32 differed: the original returned zero while the
candidate terminated with SIGFPE under invalid trapping, despite matching input
words and entry state. The other 16 agreed by **both terminating with SIGFPE**
under inexact trapping. Matching termination supplies no post-call value or
state evidence. Both mutation controls discriminated: removing the finite guard
caused 360 matrix-policy failures; removing the equality guard caused 72.

These are synthetic fixtures of extracted local components, including a direct
matrix insertion wrapper. They do not compile or exercise the full generated
adapter, model initialization, reducer, physical amplitude, QP transport/cache
keys or real primary/reversed calls. The failed local qualification is preserved
without being generalized to a full-adapter result. No performance conclusion
is made.

This package retains the exact final execution record, progress record and all
124 command stdout files, including compilation, linking, loader listings, bulk
frames, mutation frames and trap diagnostics. Deterministic gzip payloads record
both compressed and original byte hashes/sizes. The manifest retains actual
commands, environment, runtime dependency listings, process outcomes and
executable hashes. `artifact-hashes.json` additionally records the ten object
hashes observed during retention. No executable, object, library or core dump is
copied. Runtime loader listings are dependency-resolution evidence; they are
not a live full-amplitude runtime mapping check.

Default verification checks the portable byte inventory and its joins to the
execution metadata. `--reparse` additionally loads the unchanged, hash-bound
`../analyze.py`, replays its integer/text functions against retained gzip data,
and requires exact agreement with all four bulk, 48 trap and two mutation
results. The analyzer hash is checked before and after replay. Neither mode
invokes the fixture runner, compiler or any native workload. Absolute historical
paths are record identifiers; portable verification and replay require only
this package and its repository-relative frozen source bindings.

```sh
source tools/tirust_env.sh
python3 -I -B docs/benchmarkfolder/experiments/ew-invariant-symmetry-prototype/fortran-fixtures/qualification-1/package.py
python3 -I -B docs/benchmarkfolder/experiments/ew-invariant-symmetry-prototype/fortran-fixtures/qualification-1/package.py --reparse
```

A successful verifier exit means the failed evidence was retained and reproduced;
it does not mean fixture parity passed. `--package` is an append-once retention
operation for the completed attempt. Any packaging failure remains on disk,
without automatic retries. No alternative guard or source repair is included.

Execution SHA-256:
`5046c42550f4f968768844145518d75c5360ca8a7d22d318b5a3267e5b1233b1`.
Frozen analyzer SHA-256:
`602fec89e78ccd22105d21fb323e93608ab35ce0e86b01c751dfeb2933c329ed`.
