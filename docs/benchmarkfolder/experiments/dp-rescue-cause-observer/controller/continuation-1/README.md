# Incremental compile continuation

The cause-2 attempt reached its 1,800-second bound while compiling OneLOop. Its failed manifest is retained byte-for-byte (`7749f7d9…`), together with its original command logs. No observer DSO or host executable qualified, and no MadLoop/TIRust numerical evaluation occurred. Cargo's dependency build did run its normal GMP/MPFR build-time self-checks.

The separately authorized continuation uses CPU 11, one Cargo job, and one additional 1,800-second limit. It first verifies every artifact in that failed manifest, the original protected inputs, and all private package source trees. It then resumes the exact recorded Cargo command in the same staged paths and environment, retaining successful dependency artifacts. It neither recopies the Cargo cache nor reapplies patches. Cached target/build products may change; their historical hashes remain in the failed manifest, and the continuation records the new final hashes separately.

The script reuses the frozen builder's native function tail verbatim as Python AST, from `map_path` through its return, then calls the unchanged frozen Fortran build/link routine. The same owned-child cleanup and source, local-MPFR, linkage and runtime checks apply. Every new command log and result goes to `.tirust-build/dp-rescue-cause-observer/cause-2-continuation-1`; the prior failed manifest, commands and logs cannot be overwritten. The numerical policy, observer sources, original installed library, and six-arm execution controller are unchanged. No numerical replay is part of this command.

Run once, after the separately assigned CPU11 build is idle:

```sh
source tools/tirust_env.sh
python3 -I -B docs/benchmarkfolder/experiments/dp-rescue-cause-observer/controller/continuation-1/resume.py
```

A continuation success never changes the status of cause-2's failed attempt. A new failure or timeout is retained without automatic repetition.
