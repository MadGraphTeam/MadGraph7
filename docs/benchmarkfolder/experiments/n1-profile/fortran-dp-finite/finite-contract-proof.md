# DP raw-output finiteness contract

This is an isolated optimization experiment, not a production change or an
acceptance result. `transform.py` removes exactly the DP raw-array
`IEEE_IS_FINITE` loop after `STATUS == TIRUST_SUCCESS`. It retains the QP loop,
all input/context/ABI/status checks, the tensor permutation and pole addition,
independent MadLoop stability tests, and the final contraction finite check.

**Disposition: reject this removal for production and retain the public
diagnostic's defensive behavior.** Source analysis and an isolated build were
completed. The prepared native adapter fixture and whole-amplitude timing
were not run. No measured benefit is claimed.

The experiment uses the previously frozen generated `qcd_n1_uux_z` process and
the split-workspace diagnostic DSO with SHA-256
`1c3688011fb0b423142ab4ac4c832f5b71d8e4409ef323087dfac5ceb2075cd7`.
Its bridge source is recorded in the existing
`../bridge/split-build-metadata.json` and `split-validation-workers/split.json`.
The source hash is
`acc21eb665e523a99906d5e071781c966c5f06d57de8f8683cd713147bc73252`.
Current repository code was independently audited for the same property; the
experiment's identity is the frozen DSO, not whichever library is installed
later.

## Why the DP scan is redundant for this bridge

The frozen bridge's `ml_tirust_evaluate_dp` calls `State::dp_checked` before
writing output (`.tirust-build/n1-profile/split-bridge/src/lib.rs:795–857`).
`dp_checked` examines every real and imaginary component of every packed
returned `C64`. It changes a nominal success into `UNSTABLE` if any component
is nonfinite (lines 146–178). On an ordinary unsuccessful return it clears the
affected numerical store and discards the relevant workspaces. Full DP discards both direction
workspaces. The C entrypoint writes the caller's output only on success.

This path is common to full and poles-only calls. It is also common to cache
hits, misses, disabled caching, and independent check passes: there is no
whole-DP-output return that bypasses `dp_checked`. Engine cache hits are packed
and scanned again. `pack_dp` initializes all four returned slots to zero and
copies only selected slots (lines 503–516); unselected poles and the
unevaluated finite slot are therefore finite zero placeholders. This is a
guarantee about the returned slots, not unselected internal engine values.

`mtirust.f90` declares `KI_TIR = c_double`, binds the output as
`complex(c_double_complex)`, and copies it into a same-kind complex array.
`tensor_dp` validates shapes, obtains an ABI-v1 context, and directly calls the
C entrypoint (lines 245–263). It performs no floating-point arithmetic or kind
conversion on a successful raw output. An array packing temporary, when
needed at lower rank, copies identical binary64 components. On any nonzero
status, the module clears the output and the generated adapter returns before
using it.

The ABI context-version check remains unchanged. The experiment does not
change installation metadata checks, library hashes, source pinning, symbol
isolation, or compiler flags. The repository-owned bridge validates results
even when an allowed installer override supplies a different upstream engine
source. A replacement DSO that falsely reports success with nonfinite output
would violate this bridge contract; this experiment does not claim protection
against an arbitrarily incompatible implementation bearing the same ABI
number. That distinction is the reason the removal was rejected for
production. The existing successful-DP-output guarantee can still be
documented explicitly in the bridge's public numerical contract without
removing the adapter's independent defensive check.

The recording test in `tests/unit_tests/loop/test_tirust_fortran.py:253–254`
can inject an explicit failure or success with a NaN for either precision.
Its existing driver checks explicit DP failure at lines 314–316 and
success-with-NaN QP failure at lines 317–319. The scoped removal leaves those
assertions intact and changes no tests. Nevertheless, source inspection
shows that a DP success-with-NaN provider would change from an unstable
diagnostic return to a nominally stable return containing NaN. The final
amplitude finiteness check is a separate safeguard and cannot preserve the
public tensor diagnostic's own result contract. Passing the narrower
existing test would not justify that semantic change.

For valid ABI buffer addresses and sizes, ordinary input errors, unsupported
requests, reduction failures, and contained panics before publication return a
nonzero status. Output is published only after successful finite validation.
An unwind while holding the state mutex poisons that mutex; later calls reject
the poisoned context instead of reusing its intermediate results. The finite
scan omission does not change that existing failure behavior.
The unsafe C ABI still requires valid nonaliasing caller storage; source-level
finite validation does not make invalid pointers safe.

## Checks that must remain

Finite raw operands may overflow during the addition of UV and IR simple
poles or subsequent numerator contraction. The deleted loop never checked
those later results. The existing common `TIRLOOP` check on the three final
complex Laurent coefficients remains unchanged, marks a failed result
unstable, sets `FPE_IN_DP_REDUCTION`, and clears its result. Independent
direction and precision stability tests also remain unchanged.

QP has a different transport boundary. `pack_mp` rejects nonfinite MPFR
components, but a finite MPFR exponent can exceed binary128's range.
`mtirust` parses decimal records with `IOSTAT` and explicitly rejects parsed
nonfinite binary128 values. These checks and the QP adapter scan are all
retained. This experiment makes no QP scan-removal claim.

Poles-only diagnostics deliberately insert a quiet NaN in the unevaluated
finite MadLoop coefficient after successful raw evaluation. That existing
behavior is unchanged. Ordinary amplitude entrypoints still reject partial
requests; the raw zero placeholder must not be interpreted as a computed
finite term.

## Evidence and its limits

`existing-native-contract-evidence.json` is a reproducible read-only reanalysis
of previously retained native results. Across 90 raw calls, all 80 successful
calls returned finite components and all 10 failed calls left every sentinel
unchanged. It includes full pole selectors 0–3, partial selectors 0–2,
cache reuse, disabled caching, check passes, overflow failures, invalid input,
and post-failure recovery. The exact raw payloads and source/library hashes
remain in the indexed bridge evidence. These examples support the universal
source argument; they do not replace it.

`static-audit.json` confirms that the QP routine's disassembly and relocations
are exactly identical. DP retains its original GNU Fortran IEEE procedure
entry and exit calls. DP code size falls from 4565 to 4286 bytes under the same
unoptimized adapter compiler recipe. The prepared native adapter fixture
would record output/input bits, stability, and five IEEE flags for each DP/QP
call through N=1–6, rank=0–2, and both directions. It was not executed. Neither
native adapter parity nor whole-amplitude timing is claimed for this variant.
