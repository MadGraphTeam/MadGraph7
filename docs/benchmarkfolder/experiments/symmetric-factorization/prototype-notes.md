# Guarded derivative-row factorization prototype

This isolated Fortran experiment changes the association of binary128 sums.
It has not changed production MadLoop, the final generated outputs, or the
benchmark inputs. Kernel measurements and complete process validation are
separate requirements; successful compilation or a symbolic identity does not
establish either.

For the two initial fermion vertices, the retained exact source proof supplies
signed row permutations \(B_\mu=S_\mu B_1\). The prototype retains every constant
coefficient contraction with its original multiplication and accumulation order,
forms \(U=B_1 A\) once for each input polynomial coefficient and cut component,
then applies the four signed/permuted derivative shifts. The symmetric
coefficient index is the original `COMB_COEF_POS`. Full routines form all four
rows of U. A terminal routine forms only the two rows needed by each consumed
diagonal; its public fallback remains available.

The changed association is deliberate: it replaces repeated sums of individual
products by sums containing a previously accumulated U. Outputs therefore need
not be bit-identical. The independent oracle decodes the actual stored A and B
binary128 values into exact dyadic fractions and evaluates the complete
contraction. Its forward-error scale is the sum of absolute real products,
multiplied by binary128 epsilon, so a near-zero answer is not judged by relative
error alone.

The runtime row guard uses numerical complex equality. Unlike the prospective
bit-equality guard discussed in the sealed source-research note, it ignores the
sign of a zero **component**, including `(nonzero,+0)` versus `(nonzero,-0)`.
All nonzero components must match the signed/permuted row exactly. This is a
numerical equality and changed-rounding contract, not an output-zero-sign
promise. The oracle includes real and imaginary H entries with independently
signed zero components and reports resulting output-bit/sign changes.

The fast path requires the recognized IEEE binary128 layout, nearest rounding,
four-dimensional vertex shapes, a valid cut size, supported and disabled
halting modes, bounded finite operands, and an already-set entry INEXACT flag.
The private call receives that flag explicitly because the GNU Fortran IEEE
procedure boundary can save and clear the caller's sticky flags. A false entry
flag uses the unchanged public updater. Unsupported layouts, exceptional or
extreme operands, mismatching rows, and enabled traps also fall back. The
existing integer classifiers avoid raising invalid while inspecting signaling
NaNs. Bounds of unbiased exponent -1000 through +1000 keep the small fixed
products and sums far from overflow and underflow; cancellation remains on a
dyadic grid far above the binary128 subnormal range. INEXACT is already sticky,
so regrouping cannot change its cumulative Boolean state.

`check_fallback.py` runs the existing independent shape/rounding/exception
fixtures with an explicitly false entry flag, comparing output bits, complete
input bits, and IEEE flags. It additionally enables each supported trap and
checks that the instrumented prototype uses its fallback. It uses the actual
standalone and NLO optimization flags, including NLO `-fno-automatic`.
`run_oracle.py` retains the separately authored oracle driver's bit dumps and
marks them pending numerical validation until its exact validator has run.

The first fallback attempt, `fallback-safety-v1`, passed the full rank-two tests
but failed to compile the terminal reference because the experimental test
writer concatenated two routines without a newline. Its generated source,
compiler failure, partial results and original script are retained. The writer
was corrected for `fallback-safety-v2`; no kernel arithmetic changed.

The completed false-entry campaign has 48,000 exact comparisons and 40
enabled-trap controls. The trap calls explicitly pass a true entry flag, so they
reach the halting checks. `true-entry-guards-v1` separately exercises 152
single-violation controls with the actual sticky INEXACT flag set: nonnearest
rounding, extreme/subnormal/nonfinite inputs, invalid shape for this
specialization, and a finite nonzero derivative-row mismatch. Each case takes
one fallback and reproduces public output/input bits and cumulative flags.

The first FFV oracle campaign consists of 20 compiled streams, 200 fixtures and
250,880 complex output coefficients. All pass the independent exact stored-input
oracle, including 2,582 severely cancelled outputs. The maximum observed
factored error is below 0.72 epsilon times the absolute-product scale, within the
declared forward bound. The separately retained FFS selector extension admits
only two further proved scalar vertices. Its 12-source comparison establishes
that every previously generated FFV rank/full-or-terminal body is byte-identical.
The separate four-stream FFS oracle campaign is not process-performance evidence.

`kernel-timing-v1` is retained as exploratory evidence: its timing executable
checked each repeated output against a locally warmed result, but the script did
not mechanically bind those warm values back to the independent oracle dump.
`kernel-timing-v2` adds exact source-hash checks and untimed binary128 dumps of all
trial input and output values, which must match the validated oracle's stored
bits before any ratios are accepted. The timed kernels omit trace counters and
include all guards. These are concurrent-host CPU-time kernel diagnostics,
without a complete-process speed claim.

`qcd3-factor-trace-v1` compares copied final generated QCD3 outputs using all 100
unchanged points and original Ninja backend 6. The user coefficient tolerance
and the unchanged strict stability settings were written before execution.
All 800 printed binary64 coefficients, accuracy estimates and statuses match in
this sample; each variant uses 98 DP evaluations and two QP rescues. The measured
trace accepts 6,528 calls and falls back on none across six private helpers.
This establishes reachable guards and sampled external agreement, not universal
binary128 identity or a full-process speed improvement. The subsequent
uninstrumented seven-pair campaign is a separate retained experiment.

The completed seven-pair series, `qcd3-factor-quiet-v1`, is **inconclusive for a
speed improvement**. Its ratio of medians is 0.9677909505, with paired bootstrap
95% interval [0.9462018443, 1.0085839876] at the predeclared retention threshold
1.0. All seven pairs retain identical printed coefficients, accuracy estimates,
and 98-DP/two-QP status counts. No samples were removed and no production change
was retained. This result does not establish the roughly 3.2% improvement
suggested by its point estimate.

The trial used CPU211 after preflight rejected busy CPU206 and transient kernel
activity on CPU223. All three preflights are retained and occurred before any
timing sample. CPU211 has one online hardware thread on socket1/NUMA node6.
Five-second activity samples cover the entire 304.6-second series, including
builds and initialization. Other task-owned compute was paused; the previous
reference process had ended before this trial. CPU207 nevertheless recorded
37.3% host activity, so the evidence describes a coordinated local window on a
shared host, not an exclusively reserved machine.
