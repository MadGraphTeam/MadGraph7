# Private adaptive QP bridge: compiled, not numerically qualified

The private bridge and its test binary compiled successfully in attempt 3.
No test, reducer evaluation or amplitude ran. Production sources, installed
libraries and the frozen historical 256/512 libraries were unchanged. This
package retains all three source/controller snapshots and both failed builds.

| Attempt | Result | Change before this attempt |
|---|---|---|
| 1 | Rust parser rejected an unclosed delimiter | Initial frozen prototype |
| 2 | Test compilation lacked the SelfRing trait | One closing brace added to the new telemetry getter |
| 3 | Compile/link succeeded; test binary not run | SelfRing imported in the new test module |

The implementation after the brace correction is byte-identical in revisions
2 and 3. No numerical policy, eligibility rule, precision, transport, cache or
test assertion changed during the compiler-fix iterations. Source snapshots'
statements that compilation has not yet occurred describe their freeze time;
the later execution records here provide the build outcomes.

The successful source index is
`f5550b8719e76e177ad6b73d456cc655815770eb7092a8fabdc776c415c03213`.
The test executable hash is
`d22953ed3b955adf7dcce3831259060cec9e974816d432a4a86fd000b1afe8b7`.
The successful compile took 102.84046 seconds on CPU 11 with one Cargo job.
It used locked/offline dependencies, Rust 1.97.1, Cargo 1.97.0 and a recorded
positive environment allowlist. The two further attempts shared one fixed
900-second monotonic budget. The reviewed runner2 owned-process supervisor
enforced the remaining deadline and reaped every compiler process group.
Cargo's --no-run flag prevented the 11 unchanged and 9 new tests from running.

The private behavior under test is 256-bit arithmetic first, followed by at
most one genuine 512-bit attempt only for a recognized current Gram refusal
with a positive finite recorded condition. The 256-bit gate remains 2^-136;
the 512-bit gate 2^-392 retains the same 120-bit arithmetic-headroom heuristic,
not a proved error bound. Exact 113-bit inputs, independent MadLoop checks,
separate full/poles/check/tier workspaces and bounded tier-specific result keys
remain required. No independent check cache is implemented. The additive
telemetry reports actual working-tier attempts and successes; return code 323
alone cannot identify an internal tier. A promoted DP numerator remains DP.

The new tests compile but have not established their assertions. Later gates
include the genuine conditioning fixture, failure/cache/tier isolation, actual
Fortran transport and IEEE/trap behavior, mixed-tier amplitude comparison, and
working-tier metadata in diagnostic records. An existing all-512 experiment
does not qualify a mixed 256/512 result. No production or performance promotion
is supported by this compile-only package.

`copy-bindings.json` maps every historical copied file to its original path and
hash. Native binaries are recorded by hash only. Cargo-reported dependency
artifacts were hashed after the successful build; this is explicitly not a
claim that all cached binaries were pinned before compilation. The controllers
do pin their frozen source inputs and historical DSOs before and after each
attempt. Copied manifests/controllers are historical evidence and retain their
original staging paths; do not execute them from this report directory.

Run `python -B verify.py` for portable inventory, exact compiler-fix and build
record checks. Add --live to additionally verify available original source,
test binary and dependency-artifact hashes. Verification never loads or runs a
native library or test executable.
