# Private observer build qualification

The private observer DSO and all four original/observed QCD/EW executables built successfully. Local MPFR binding, actual interpreter runtime closure, exact numerical-object provenance and linker inputs passed. This is compilation and linkage evidence only: no six-arm replay, observation-neutrality, cause-attribution, accuracy or performance claim is made.

The history remains explicit:

| Attempt | Outcome |
| --- | --- |
| cause-1 | Cache preparation rejected an unrelated GammaLoop checkout symlink before Cargo/compiler execution. |
| cause-2 | The narrowed locked-source cache passed; the 1,800-second build timed out during OneLOop compilation. |
| continuation-1 | Reused dependency artifacts; the observer DSO and first QCD original executable built, then a provenance guard misclassified the pinned compiler module as an archive. |
| continuation-2 | Accepted only the exact frozen `lib/mpmodule.mod` identity, reused the QCD original arm, and completed the remaining Fortran builds and linkage checks. |

All failure manifests and original command logs are preserved byte-for-byte. The continuation changes are recorded in the adjacent controller revisions and diffs, bound by `source-bindings.json`. Cached compilation outputs legitimately changed during reuse; earlier manifest artifact hashes describe their historical attempt-end state, not the final contents of those cache paths. Protected scientific inputs and earlier manifests/logs remained unchanged. Cargo's ordinary dependency build included GMP/MPFR build-time self-checks; these were not MadLoop/TIRust reduction evaluations.

Every retained raw manifest, command log and link map is deterministic gzip text, with both compressed and original SHA256/byte counts in `index.json`. No executable, object, archive, DSO or Cargo dependency cache is copied here. Binary identities and actual runtime/link-input provenance remain in the retained manifests and compact `summary.json`. Run `python3 -I -B package.py` from any directory to verify portable payloads, decompressed raw identities and the bound repository sources; it does not invoke a compiler or numerical library.

The unchanged execution controller's `verify_build` passed 33,940 current bindings. A byte-identical final manifest is also placed under the required `cause-2-qualified/build/build.json` admission layout. `future-command.json` contains the exact **unlaunched** command and tool pins. Independent source review confirmed that this metadata copy satisfies the existing layout check.

All six numerical arms must wait for root's admission and the common numerical lock. The frozen controller runs QCD original/disabled/enabled, then EW original/disabled/enabled, serially under one lock, with 600 seconds per arm and 3,900 seconds overall. Each process uses the unchanged bootstrap-plus-100-point warmup stream; disabled/enabled use the same observed executable and DSO. There is no individual-arm or partial-resume option. Lock contention returns 75 before creating an attempt; a started failure remains consumed. Do not add an outer lock around the controller. Read-only hashes, source inspection and portable evidence verification may run independently; no numerical arm may be run ahead of the six-arm admission.
