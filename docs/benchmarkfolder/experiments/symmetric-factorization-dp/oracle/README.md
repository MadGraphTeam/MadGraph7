# Experimental binary64 stored-input oracle

This directory is separate from the frozen QP oracle. It supports an experimental
DP derivative-row factorization, with no production change or performance claim.
After the coordinated quiet window ended, the actual-source binding and all
seven oracle self-tests passed; see `source-parity.json`, `bind-sources.log` and
`self-tests-v1.log`. Compiled helper validation and exception controls are
separate pending gates at this stage.

The private interface agreed with the prototype owner is
`TEST_UPDATE_WL_<rank>_1_FACTOR` or `TEST_TERMINAL_WL_<rank>_1_FACTOR`, with
`(A,LCUT_SIZE,B,IN_SIZE,OUT_SIZE,OUT,WL_ENTRY_INEXACT)`. All complex arrays are
`COMPLEX*16`; dimensions are MAXLWFSIZE=4, LOOPMAXCOEFS=210 and
VERTEXMAXCOEFS=15. Instrumentation uses `COMMON/FACTOR_TRACE/FAST,FALLBACK`.
Initial executable prototype coverage targets full compact ranks 2–4 and both
FFV1 orientations. Rank-1 expanded helpers are outside this scope. Terminal
oracle generation is prepared for a separately proved experimental helper;
there is no existing public compact DP terminal routine, and no terminal
process support is claimed by this preparation.

`proof-inputs/` retains the actual generated DP `FFV1L1_2.f` and `FFV1L2_1.f`
from the b44 QCD3 export. `bind_sources.py` will compare all 80 coefficient
expressions per vertex with the sealed exact source proof, prove all 64
derivative-row identities independently, and bind DP source hashes in
`source-parity.json`. The driver evaluates the actual DP expressions at runtime.
It does not cast previously computed QP coefficients to DP.

`oracle.py` decodes each stored binary64 real/imaginary component from one
16-digit hexadecimal word using only integers and `Fraction`. Signed-zero bits
are retained separately. The mathematical reference is the direct monomial
contraction over the stored A/B arrays, with independently constructed symmetric
monomial indices; it does not reuse the prototype's intermediate U or summation
grouping. It checks the complete ten-case stream, full array extents, input
immutability, inactive output tails, cumulative IEEE flags, path counters,
fallback bit identity and numerical row relations.

The original reference call always uses the unchanged full public DP updater.
All original output bytes remain in the dump. If a future terminal prototype
takes its accepted path, the oracle checks its diagonal against the full direct
contraction and requires numeric zero in every off-diagonal output. If that
prototype falls back, the oracle instead requires the complete original output
bit for bit. Thus no invented public terminal implementation supplies the
reference, and no projection overwrites the retained original result.

The forward bound is the same conservative operation-graph bound specialized
to binary64, not a copied QP numerical tolerance. With N nonzero complex terms
and S the sum of their four absolute real products, both original and candidate
must satisfy `max(component errors) <= gamma_(2N+8)(epsilon)*S`, where
`epsilon=2^-52` and `gamma_k(epsilon)=k*epsilon/(1-k*epsilon)`.
The exact result and worst errors/scales are retained as rational strings.
Cancellation is reported at `|Re(reference)|+|Im(reference)| < 2^-35*S`;
no relative-to-near-zero-output gate is used. This is not a correctly-rounded
polynomial guarantee or a substitute for physical amplitude/stability controls.

Fixtures use perturbations at `2^-48`/`2^-49`, which are distinguishable in
binary64, and verify that the generic A perturbation survives storage. They
cover dense/sparse inputs, severe cancellation, common scaling by `2^180` and
`2^-180`, coefficient-dependent mixed scales, actual independently rounded DP
ALOHA rows, false-entry fallback, and both signs of zero components in nonzero
real/imaginary derivative rows. Unlike the QP fixtures, no sub-binary64
perturbation is silently lost. The signed-zero cases require all four
real/imaginary positive/negative-zero patterns to be present in the raw dump.

The accepted path compares component values numerically, ignoring zero signs.
Its outputs may change rounding and zero signs. An actual stored row mismatch
must fall back; the actual-ALOHA fixture allows this outcome. False entry-inexact
must fall back bit-identically. Admission and exceptional-input controls are
specified separately in `safe-bound.md`; their execution belongs to the
prototype owner. No such control is reported as passed before it runs.

The source-binding and validation sequence is:

```sh
python -B docs/benchmarkfolder/experiments/symmetric-factorization-dp/oracle/bind_sources.py \
  --output docs/benchmarkfolder/experiments/symmetric-factorization-dp/oracle/source-parity.json
python -B docs/benchmarkfolder/experiments/symmetric-factorization-dp/oracle/test_oracle.py -v
# The prototype owner then compiles the generated driver and retains its dump.
python -B docs/benchmarkfolder/experiments/symmetric-factorization-dp/oracle/oracle.py \
  path/to/dump.txt.gz --vertex FFV1L2_1 --result path/to/exact-oracle.json
```

These scripts suppress Python bytecode caches. Result files refuse overwrite;
failed builds, guard controls and numerical validations must remain in their
original experiment namespace rather than being replaced by a successful retry.
