# Frozen adaptive-QP test controller

This is an exact portable copy of the reviewed private test controller,
source index `74058446fda67d455c78bf1d3404a10d9bf9a5cdb4a66f6a9ca47d16c4ac9603`.
No test or native library was executed while packaging it. The binary is
recorded only by path, size and SHA256; no executable is included.

The controller permits one serial 20-test process, CPU11, under the common
nonblocking numerical lock and the reviewed inclusive180-second process
supervisor. Root's conditional authorization requires the priority runner2
diagnostic and root stability check first, followed by explicit harness/root
slot admission. That slot has not been admitted in this frozen package.
There is no automatic retry or busy-lock loop.

The copied run.py retains its original staging paths and authenticates that
original controller tree. It is historical source evidence, not a separately
authorized invocation from this directory. The original prepared command is:

```sh
python -B .tirust-build/adaptive-qp-rescue-test-controller/run.py
```

The exact expected binary comes from the successful private compile-only
package `1a4af7f8…`: 11 unchanged Rust tests plus 9 new adaptive tests compiled,
none yet run. A later successful unit process would still not qualify Fortran
transport, trap behavior, installed runtime or physical mixed-tier amplitudes.
The controller has no actual-runtime mapping observer; this limitation is
explicit in its copied README.

Run `python -B verify.py` for portable inventory, source-copy and build-binding
checks. Add --live to check the original controller and test binary hashes.
Verification is read-only and never executes a native artifact.
