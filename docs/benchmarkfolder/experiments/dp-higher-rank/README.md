# Private higher-rank terminal profile

This bounded experiment extends the guarded diagonal technique privately to
rank 2×1 and 3×1, using the fresh QCD n3 output that already contains the
committed 26 rank-1×1 sites. Production support remains unchanged. All native
work finished at 23:31:19 UTC, within the ten-minute budget beginning at
23:23:29 UTC. CPU 10 was used; these shared-host kernel timings are diagnostic.

The complete private source proof passes for 147 roots: 26 existing 1×1,
85 new 2×1 and 36 new 3×1. It checks 139 Fortran units and 171 source/include
paths, including the exact canonical coefficient-index table and its read-only
closure. Reachable table entries are bounded by the actual output storage.
The original compact M,K,I,L accumulation order, zero-B skips and full output
initialization are retained. The source closure is copied in `process-source/`.

Compiled qualification passes 1,536,000 synthetic comparisons of consumed
outputs, admission and IEEE state, plus actual generated caller→CREATE/MERGE
checks for both ranks. Four real array contexts were captured from the retained
QCD n3 generated process using explicit CutTools selection and a deliberate
stop before tensor reduction. All four corrected caller-scope records have
pre-existing INEXACT, disabled traps and bounded active operands. Their B arrays
contain respectively 32, 40, 40 and 40 nonzero complex entries out of 80.
These four samples do not establish an admission rate across a process stream.
The recorder did not retain the rounding mode; replay explicitly sets nearest
rounding. Thus complete admission is established for replay, not inferred for
every original call. Sparsity counts above are computed from exact captured
array bits, separately from the timed executable.

All four real-input compiled replays preserve consumed bits, IEEE flags and
input bits. Seven alternating pairs of 500 calls use the actual 70-coefficient
module shape and existing polynomial `-O3 -fPIC` flags without bounds checking.
The private timings include the caller IEEE query and every runtime guard;
the unchanged dense control has no added query. Median private/dense ratios:

| Captured context | Ratio |
| --- | ---: |
| Rank 2×1, first | 0.51093 |
| Rank 2×1, second | 0.46889 |
| Rank 3×1, first | 0.40706 |
| Rank 3×1, second | 0.39433 |

These are warm, repeated-input kernel measurements. They establish neither
whole-amplitude speed gains nor an acceptance-campaign result. A subsequent
full-amplitude comparison would need frozen sources, unchanged libraries and
flags, stream-level parity and a separately approved timing window.

Every attempt is retained. The initial capture build failed because the
makefile's `/bin/bash` was absent; retry used the supported shell override.
Attempt 2 queried IEEE state inside the recorder and cannot establish caller
admission. Attempt 3 captures flags inline in the actual generated caller;
all array payloads remain identical. The first synthetic oracle used a different
loop traversal and changed exceptional NaN payload bits; subsequent checks use
the unchanged public compact routine. The first captured-input build had a
TRANSFER array-rank error; the corrected version uses RESHAPE.

`summary.json`, `kernels/results.json`, `kernels/captured-results.json`, and
`capture/admission-sparsity.json` contain the result scope and raw-record links.
`proof/README.md` gives source-only proof reproduction. No native executables
or object files belong in the portable evidence index.
