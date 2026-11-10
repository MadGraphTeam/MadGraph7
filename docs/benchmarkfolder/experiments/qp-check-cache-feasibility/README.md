# Independent check-tensor reuse: retained call evidence

The saved QCD n3 stress trace contains repeated exact tensor inputs within its
CTMODE 5 check pass. Of 116 calls, 89 input keys are distinct: 27 calls repeat
an earlier key. In the original-geometry control, 26 repeated pairs succeeded
and one pair (groups 41/55) refused twice. In the private topology-C variant,
all 27 repeated pairs succeeded. Every repeated successful output has exactly
the same binary128 tensor-moment words. The current bridge bypasses its QP
cache for CTMODE 5, so these check results were evaluated separately.

This is evidence for investigating a separate check cache. It is not a cache
implementation, cache-hit measurement or speedup result. Both saved stress
amplitudes still return 420; their physical acceptance remains unresolved.
The two ordinary controls use DP and have no recorded QP calls.

The key comparison retains exact shifts, supplied distances, complex squared
masses, squared scale, N, rank, pole selection, finite selection and precision.
It preserves signed zero and partitions CTMODE 4 and 5 before counting. These
are the exact binary128 inputs before deterministic ES64.39E4 transport; no
geometric tolerance, denominator permutation, numerator or group identity is
used. Raw binary128 equality guarantees identical packed input records. All
input/output and duplicate-group details are retained in counts.json.

Every recorded QP call has squared-order index 1 and each group is visited once
per direction. Thus the observed repeats are **between groups**, not evidence
of repeated helicities or split orders. In this direct-interference export,
MP_COMPUTE_LOOP_COEFS first aggregates helicities and the host subsequently
reduces the group coefficients. Other split-order and loop-induced flows need
their own call census; their repetition must not be inferred from this sample.

The pinned host source clears caches at event entry and exit when automatic
clearing is enabled, and unconditionally at label 200 before a rotated point.
The primary-to-reversed check transition instead jumps to label 300 without a
clear. A future check cache therefore needs a distinct direction namespace;
it must never satisfy the first check evaluation from the primary result.
Exact effective inputs, working tier, mode and policy must remain in its key.
No cross-event or cross-rotation reuse is counted here. Event and explicit
clear calls must clear both direction namespaces, and aggregate resident
payloads must obey the existing overall budget rather than multiplying it.

The source-only adaptive prototype continues to bypass all check-cache reads
and writes. Any future optimization should be a separate change after the
adaptive path is qualified. Failed/invalid/nonfinite results must never be
retained; transport failures, context poisoning and failure invalidation need
the same conservative treatment in every affected direction/tier. These
records do not justify negative caching of conditioning refusals.

A cache hit also skips arithmetic side effects. The trace records sticky
wrapper flags but not each kernel's MPFR or complete IEEE flag delta. Exact
returned moments alone therefore do not qualify host flag/trap behavior. A
separate cache experiment would need actual first-check evaluation evidence,
bounded memory accounting, changed-input/mode/tier/direction/event tests,
failure recovery, exact output and IEEE/MPFR comparisons, followed by whole
amplitude measurements. No implementation or new native run is authorized or
performed by this package.

Run `python -B verify.py` from this folder (or pass the script path from the
repository root) for portable inventory/source binding checks and a pure
recount of the two frozen traces. It imports only the authenticated Python
decoder, never loads or executes a numerical library. Source anchors and
hashes are recorded separately from native observation in source-notes.json.
