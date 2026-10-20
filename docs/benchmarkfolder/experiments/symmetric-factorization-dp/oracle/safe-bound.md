# Proposed binary64 admission bound and environment controls

For each active real and imaginary input component, admit exact signed zero or
a finite normal binary64 number with unbiased exponent `e` in `[-200,200]`.
In raw IEEE binary64 bits this means magnitude zero or biased exponent in
`[823,1223]`. NaNs, infinities, subnormals and values outside this interval must
reach the unchanged original fallback. Classify integer bits before floating
comparisons or conversions that could quiet a signaling NaN or alter flags.

This interval leaves a conservative arithmetic margin for the specified graph:

- Every admitted component is a multiple of `2^-252` and has magnitude less
  than `2^201`.
- Every exact real product is a multiple of `2^-504` and has magnitude less
  than `2^402`.
- An output contains at most 20 complex input products, or 40 real-product
  contributions per component. Its accumulation and the four-term U sums stay
  below `2^408` in magnitude.
- Under round-to-nearest, rounded products and sums remain on the `2^-504`
  lattice. Any nonzero cancellation result is therefore at least `2^-504`.

These extrema lie well inside binary64's normal finite range
`[2^-1022,2^1024)`. Thus neither intermediate overflow nor gradual underflow is
needed by an admitted evaluation. Zero results remain allowed. This argument is
for the bounded multiply/add/sign-permutation graph; it does not justify a
different algorithm that introduces divisions, rescalings or transcendental
operations. The QP `±1000` input bound is not appropriate for binary64 and must
not be reused.

Additional admission requirements are nearest rounding, all queried halting
modes supported and disabled, the required dimensions, a true captured
entry-inexact flag, and exact numerical derivative-row relations. Zero-component
signs may differ. The original function must handle every rejected case. The
prototype must not clear flags, mask exceptions, or change rounding mode to make
an input admissible. Accepted operands/intermediates are normal, so flush-to-zero
and denormals-are-zero cannot change their arithmetic; unsupported or separately
modified hardware rounding controls still require rejection or an explicitly
validated fixed-environment contract.

Planned safety controls must distinguish the following groups:

1. False-entry fallback identity with complete output/input bit comparisons and
   cumulative flags. These cases do not establish the later guards.
2. True sticky entry-inexact, otherwise compatible B rows, and one violation:
   nonnearest rounding; exponent `±201`; subnormal input; infinity; quiet/signaling
   NaN; valid smaller shape; or a broken finite derivative relation. Check one
   fallback and original output/flags, with complete case-count assertions.
3. Exact endpoints `e=-200` and `e=200`, maximum mantissas at those endpoints,
   signed zeros, cancellation and mixed exponents. These must exercise the
   admitted graph and its exact stored-input bound.
4. Each supported preexisting sticky flag, with traps disabled, across original
   DP and guarded accepted/fallback calls. Reset both paths to the same initial
   flag set, keep actual entry-inexact consistent, and check each flag survives.
5. Each supported enabled trap, first with an exact no-new-exception input and
   then with operations deliberately raising invalid/overflow/underflow/inexact
   where the original updater can produce them. Run original and candidate in
   separate subprocesses and retain both exit/signal outcomes and trap masks.
   This multiplication/addition updater has no division, so division-by-zero
   coverage is a mask-preservation control rather than an invented division.

The last two groups are DP-specific requirements: the original compact DP
updater lacks IEEE imports, while the private guarded helper introduces an IEEE
procedure boundary. Existing QP evidence does not establish that this new GNU
boundary preserves preexisting sticky flags or enabled-trap behavior. Trap
controls must not call a shared IEEE-heavy wrapper around both implementations
that hides the boundary difference being tested. Expected trapping attempts are
retained as controlled outcomes; unexpected differences fail the experiment.

Use actual standalone/NLO compiler flags and record compiler identity. Do not
add unsafe reassociation or fast-math flags to obtain a favorable result. Exact
algebra tests, safety/environment tests, physical amplitude accuracy/stability,
and process timing remain separate gates.
