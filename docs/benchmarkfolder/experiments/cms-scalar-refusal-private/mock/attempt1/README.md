# Bounded pure-Fortran mock qualification proposal

The successful source/build milestone is immutable. This append-only proposal
uses its two exact mock executables; there is no recompile, archive replacement,
real AVH analytic call, Rust library, or backend 8/9 evaluation.

After explicit root admission, `control.py run` performs two exact-runtime loader
checks (at most five seconds each), then the classifier mock (15 seconds) and
caller mock (10 seconds), sequentially on CPU 8 under a shared 35-second deadline.
The main reducer lock is unnecessary: these binaries contain recording scalar
and coefficient stubs. All loader objects must exactly match the frozen GNU
runtime closure. Loader injections are rejected and library search directories
are limited to that closure. No retry follows a failure or interruption.

The controller reuses the already reviewed owned-process supervisor, which kills
and reaps the process group before returning. Each process uses an empty private
directory. It retains the exact command, exit/status, merged stdout/stderr,
output-file hashes, host context, and protected-input verification before and
after execution. A nonzero exit, timeout, missing/truncated record, extra output,
unsupported subnormal fixture, or failed assertion stops qualification.

The classifier must emit exactly both eight-overload precision completion records
and its final sentinel. Its hard assertions cover erasure metrics, strict cutoff
boundaries, complex operands, underflow/overflow, finite and nonfinite callback
results, original input bits and unchanged output buffers on refusal.

The caller has 91 logical cases in two synthetic mass variants, for 182 primary
cases. Each runs directly, through the rendered private MadLoop adapter, and
through a subsequent successful reset: 546 future reduction calls. The complete
frozen journal specifies each status, stability/result-valid flag, precision,
direction, refusal history, scalar/coefficient/amplitude count, normalization
count, and active MadLoop reduction-FPE latch. The strict parser requires every
row in order, including the final sentinel; exit zero alone is insufficient.

IEEE coverage is limited to finite/NaN predicates, exact storage bits and the
MadLoop refusal latches. These latches do not prove a hardware exception. Sticky
flag, trap and rounding transparency remain unqualified. Physical accuracy,
accepted analytic-value equality, complete MadLoop host rescue, and performance
also remain unqualified even if both mocks pass.

`control.py verify` checks the proposal without launching a process. The retained
source/build milestone has a separate portable verifier under
`../source-build-evidence/verify.py`; native artifacts there are hash-only.
