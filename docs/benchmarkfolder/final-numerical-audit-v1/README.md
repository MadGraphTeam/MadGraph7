# Independent numerical interface audit

This bounded source and retained-evidence review covers the production bridge,
MadLoop Fortran adapters and guarded QP wavefunction changes at the hashes in
`verification.json`. It starts from branch revision
`81a4adff34b00188c844f5eada268c6acf33d317`. It found no new actionable
user-visible numerical correctness defect in that source. It does **not** pass
the unfinished eight-process correctness or runtime gates, certify an unpromoted
native/allocation proposal, or certify subsequent source/dependency changes.

No compiler, reduction engine, MP worker, amplitude or timing run was launched
for this review. `verify_evidence.py` rechecked five existing sealed evidence
packages on CPU 5, including the preserved live relocated zero-sign installation.
All five verifiers passed; exact source hashes were unchanged during the checks.
The older f4b generated controls keep their original archive identity and remain
historical controls, even where the current Fortran source is unchanged.

## Numerical contract review

| Contract | Source assessment and limits |
| --- | --- |
| C layout and explicit dimensions | ABI 3 declares every DP element count and MP byte count. Checked dimensions, exact lengths and writable output ranges precede numeric slices and reduction. `C64` has `repr(C)` with two doubles, matching the interoperable Fortran complex argument. Input/output disjointness and live handle ownership remain explicit caller preconditions; arbitrary invalid addresses cannot be made safe by length checks. |
| Failure and panic behavior | C outputs commit only on success. Ordinary and compensated DP entries overwrite all coefficients/shifts before the engine prepares fresh status/state; generic MP overwrites coefficients and calls `reset_batch(1)` before evaluation. MP parsing, scale/mass rejection and failed/nonfinite evaluations return errors or instability without committing buffers. Fortran clears public results before every call and propagates failed/overflowed results into its existing FPE rescue flags. Rust unwinds are caught at the C boundary and again in the MP worker, where a panicked context state is removed. Upstream process aborts, allocation failure and Symbolica permit/license termination cannot be caught as Rust unwinds; the header documents the latter limitation. |
| Retained contexts and concurrency | Rust mutexes isolate ordinary DP, compensated DP and MP states. A single process-wide MP worker serializes contexts; destroying the final MP context joins that worker before another creation can acquire its license permit. Destruction concurrent with a live evaluation is explicitly forbidden. The Fortran module has a saved context and generated MadLoop uses COMMON state, so this is the existing one-thread-per-evaluation/process workflow, not certification for concurrent Fortran calls. |
| Binary128 transport | Fortran requires radix 2 and 113 significand bits and packs ES64.39E4 records. Rust parses at 113 bits, rejects nonfinite values and directly widens the represented value to 256 bits. Direct widening preserves signed zero, unlike the removed rational round trip. Formatting and Fortran unpacking return 40 significant decimal digits; the wrapper rejects unrepresentable/nonfinite final results. The sealed dyadic extremes, last-significand-bit, signed-zero and complex-branch tests cover this contract. Internal Symbolica Float operations track precision; this is not a promise that each intermediate has exactly 256 significant bits or that the full reduction is correctly rounded. |
| Kinematics, tensor order and thresholds | The adapter follows original Ninja's momentum sums, residual-conservation test, propagator reversal and complex squared-mass ordering. Odd tensor ranks change sign once for the denominator loop-momentum convention. It uses the existing DP/MP kinematic-matrix helpers and reconstructs the real invariant matrix after their dimensionless OSThres tests. Native masters keep their separate absolute 1e-9 on-shell threshold; generic masters explicitly use original Ninja's QP 1e15 * 2^-112 threshold at the selected working precision. The matrix is symmetric, so Fortran column-major versus Rust flat matrix indexing does not transpose a physical asymmetric input. |
| Scale, Laurent order and normalization | Both adapters pass the same MU_R**2 operation as original Ninja. Results are finite, single pole and double pole, with the complete rational contribution already included in the finite component. Born interference takes twice the real part with 1/(16*pi**2), while loop-induced reduction keeps the complex amplitude with the same normalization. The separate generic rational diagnostic is deliberately not added again; its basis split can differ from original Ninja. |
| Born, loop-induced, split orders and UV/R2 | Separate dispatch branches pass the original split-order-indexed loop interference or loop amplitude arrays to backend 8. The MP template change only selects actual QP numerator construction via UseQPIntegrandForNinjaRS; UV/R2/Born/color/split-order accumulation stays in the existing MadLoop flow. Baseline-to-current template diffs support that scope. Real four-backend loop-induced/mixed-order controls supplement the recording-wrapper tests; final source/runtime controls still have to be refreshed if a new proposal is promoted. |
| Retry, statuses and statistics | Compensated DP is a single explicit retry after failed finite native-DP stability checks, using the same checks; it cannot silently become the ordinary DP engine or genuine QP rescue. The public wrapper resets retry state on entry/exit. Legacy U values stay unchanged; H=2,U=8 identifies stable NinjaRS DP and H=3,U=2 identifies successful NinjaRS QP rescue. Dedicated counters remove these new uses from old PJFry/Ninja-QP histograms, and old XML/logs remain readable. |
| QP Fortran optimization | Private FFV helpers only apply to generated disjoint wavefunctions at ranks 2–4, with actual stored derivative-row identities checked at runtime. Integer-only bounded finite classification, nearest rounding, supported disabled traps and an already-sticky entry INEXACT flag guard changed association. Unsupported inputs fall back to the public updater. Terminal selection additionally proves diagonal-only consumers in all rendered Fortran files, otherwise the exporter discards the private selection. Finite fast-path outputs have a tested accuracy contract; they do not promise identical rounding or signed-zero bits. The compiled regression exercises 204 cases under each of ordinary and FKS -fno-automatic flags, including fallbacks, NaNs, rounding, sticky flags and enabled traps. |

The seven bundled physical tensor fixtures bind fourteen input/reference files
and 21 lanes; they are separate from the **18 ordinary production Rust tests**
and ignored **1,392-request/11,136-field** transport replay. The private allocation
candidate's 22-test suite includes four extra oracle tests and is not the current
production suite. The zero-sign install is a distinct archive,
`832fc0d1198814b9cc80a65e47bc7aa5dc204f7d25701f206c910cd188fbe80b`,
with fresh compiled smoke, relocated standalone DP/QP and 20/20 NLO pole checks.
Historical controls using archive `1199bc...` cannot certify a substituted archive.

## Native study history diagnostic

The separately owned native study retained a full-history versus selected-history
comparison in `experiments/ninjars-native-float256-v1/history-diagnostic.json`.
Its input/runtime/historical identities are preserved there; this review only
read that diagnostic and the associated source. Twenty n=4,rank=4 requests
(including ten repeated inputs) change raw finite and cut-constructible dyadics
under the different histories in all three policies. Their public eight ASCII
records, statuses, output precision metadata, signs, poles and rational diagnostic
remain exact across histories. The largest raw change relative to a common
Laurent norm is 5.665930685366729e-65. All retained 256/384 Laurent screens pass.
The separate physical-fixture sign change is a nonzero cancellation residual,
not a negative-zero transport branch change.

This is **unexplained internal history dependence**, common to interpreted and
native policies. No stale public output, wrong physical coefficient or lost input
precision is demonstrated. It must not be described as exact raw history
equivalence, exact full-reduction rounding, or a proven native-specific defect.
The bridge replaces all incoming values and resets amplitude accumulation;
the tensor recurrence clears/replaces scratch; pinned Float::set_from and
Complex::set_from copy represented values and precision. The scalar and batched
Symbolica interpreters copy parameters and initialize each Add/Mul temporary
from its first operand. These source observations rule out the simple claim
that an unchanged destination precision survives those assignments. They do
not establish the cause in retained geometry, masters, evaluator construction
or nested function workspaces.

The newer Symbolica 9cdc294 source fixes a real mixed-precision multiplication
and division rounding defect and changes multiplication by exact +/-1. That
global arithmetic change is independent of the history observation: sqrt can
produce a 257-bit value from a 256-bit argument even in a correct fresh
evaluation, so mixed precision itself does not imply a stale cache. A controlled
trace or regression is needed before attributing this observation to that
upstream change. No dependency was updated and no numerical source was changed
in this audit.

A separate plausible source is symbolic construction order. NinjaRS reduction
kernels use `ninjars-positional` specifically to remove symbol-registration
history from their instruction ordering. The generic master's oneloop expression
builder keeps the original symbol ordering and does not use that positional
renaming. The full stream starts with n=2,rank=1, whereas the selected stream
starts with n=3,rank=2; master arguments are registered before lazy shared
oneloop definitions. These histories can therefore differ before numerical
workspace reuse. This is a source-supported hypothesis, not an observed
instruction-DAG difference or established explanation of the measured drift.

## Proposed isolated panic regression

Root requested one cheap, engine-free regression for a deliberate panic through
the private C-boundary guard. `panic-contract-tests.patch` and `proposed/` retain
that test and its cfg(test)-only module declaration for root to apply after the
private timing freeze. The test checks ERROR, the actual C last-error API's
required length/NUL/truncation contract, replacement of an older error, and
clearing of the thread-local error after a successful call. It has not been
compiled or executed yet and is not counted among the 18 tested production
cases. The briefly applied live append/new test were restored immediately on
root's timing-freeze correction; the live lib.rs returned to its exact a4f7215e
source identity and no build/evaluation ran during that interval.

## Remaining release checks

1. Freeze the final promoted numerical source/dependencies and bind the installed
   archive and all regenerated exports to it. Refresh relevant controls and MG7
   unit/compiled tests after any numerical change.
2. Finish the eight-process 100-point four-backend correctness, legacy regression
   and per-process 5% full-evaluation performance gates with actual rescue counts.
3. Include loop-induced timing/stability, mixed/complex/near-singular/forced-QP
   controls, convergence and available analytic pole checks in the final report.
4. Keep this microscopic raw-history observation visible until classified or
   resolved; sampled ASCII identity and convergence are the currently demonstrated
   limits. Do not turn a tolerance check into a reset/bitwise-equivalence claim.
5. Preserve Symbolica process/thread licensing limits and actual pending/failed
   gates in the report. This audit alone cannot justify marking the goal complete.

Recheck this sealed audit without writing files with:

```sh
source .ninjars-build/dev-env.sh
PYTHONDONTWRITEBYTECODE=1 taskset -c 5 python -B docs/benchmarkfolder/final-numerical-audit-v1/verify.py
```

Add `--sources` to require that every audited production source still has its
recorded hash. The separate `verify_evidence.py` command regenerates this audit's
verifier logs and summary and invalidates its existing seal; it needs preserved
runtime artifacts for the historical live installation check. Neither script
rebuilds missing artifacts or rewrites historical expectations.
