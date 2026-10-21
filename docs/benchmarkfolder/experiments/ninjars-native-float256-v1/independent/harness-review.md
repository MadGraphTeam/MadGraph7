# Independent read-only native-policy audit

Auditor: `/root/native_final/native_audit`. No build, numerical engine, process
signal, timing, production change, commit or push was performed by this review.

The private test-only policy is assigned only by construction of an empty
`MpState`, remains unchanged during evaluations, and is applied on every late
`(n,rank)` entry creation. Ordinary constructors and public ABI evaluation
remain interpreted. The source delta enables the pinned native-kernels feature,
adds this private policy/backend observation, and adds ignored study tests.
The tensor, binary128 transport, precision, scalar provider and absolute scalar
threshold remain the corrected production baseline. Prepared source hashes
were independently checked against the private source tree.

The upstream result order is finite, single pole, double pole, cut-constructible
finite, explicit rational. Flattened real/imaginary indices therefore are
finite0/1, single2/3, double4/5, CC6/7, rational8/9. Existing analyze.py uses this
order correctly and compares all six Laurent components against the common
interpreted norm; rational and finite-split diagnostics are separate.

The fixed focused run contains30 analytic/rebind records,30 explicit invalid
width/recover records, and30 malformed odd-length coefficient/recover records.
The companion focused_audit.py independently checks the raw run/result binding,
fixed coverage, input recovery at113bits, exact analytic poles, all full Laurent
256/384 convergence checks, all384-bit cross-policy comparisons, finite split,
sign/precision classifications and recovery categories. It imports the existing
hash-bound exact-rational comparison code rather than silently changing the
preregistered value screen. This closes an offline-analysis omission in the
original focused test, which checked poles and same-bit reused/fresh values but
did not itself compare its256/384 finite values.

One focused coverage gap remains. The original short_shape_then_recover control
truncates to one64-byte scalar record. `pairs()` panics while unwrapping its
missing imaginary component, before entry creation or set_coefficients. The
harness catches this and replaces the entire private state. Its retained
success is legitimate panic recovery evidence but does not establish the
comment's claimed post-entry shape-error recovery.

If native policies survive the numerical gates and are considered for promotion,
add a separately disclosed test-only supplement, preserving all original runs:

- For each policy and each256/384 precision, first evaluate a valid rank0 input
  and retain its complete observed result and entry/cache identity. Supply four
  scalar coefficient records, representing two well-formed complex coefficients
  where rank0 requires one. Assert evaluate returns explicit Err, no panic,
  no success/raw/output, no input mutation, and no state replacement. Confirm
  the same preexisting amplitude entry and policy remain, then evaluate the
  original valid input and compare its full raw/output/backend observation to a
  fresh same-policy context exactly.
- In another retained context with at least one different valid shape cached,
  submit that validly paired wrong tensor count for a not-yet-created shape.
  Assert a new entry is created with exactly the requested policy/backends,
  the coefficient-count error is returned, the prior cached entry is unchanged,
  and both new-shape recovery and the prior-shape reevaluation match fresh
  same-policy contexts exactly.
- Keep the public malformed-buffer tests on the unchanged default ABI. They
  establish public transactional behavior independently of these private
  per-policy controls.

The second source review covered the allocation process candidate at another
agent's request. Independently verified all123 sealed combined-correctness files,
the39-file corrected production/installation source chain, and the process copy's
sole tensor change. Minimal production tensor is the exact tested tensor body
with trailing cfg(test) declarations removed. Its borrowed scalar products match
the pinned Complex::mul expression in operand and round order. The only set_from
reset sources are constructor finite zero/one values; prior arbitrary-NaN-source
sign counterexamples do not lie on this reset path. Complete-process controller
and final archive packaging require separate pre-timing review.
