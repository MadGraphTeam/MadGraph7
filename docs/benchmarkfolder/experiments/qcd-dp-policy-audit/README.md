# QCD DP policy and reuse audit

The 454 recorded QCD n3 rows are **method-choice attempts**, with 320 successes
and 134 refusals. They are not 454 higher-precision attempts. The existing
counter corrigendum `dp-rescue-cause-observer/precision-counter-correction-1`
preserves that correction without rewriting historical evidence. This additive
audit joins the retained native journal to source eligibility rules. It changes
no production source or policy and runs no compiler, reducer or timing job.

The installed bridge `eb3d67fd…` uses the bound `455529b9…` Rust source,
TIRust revision `854b79dc…`, its pinned poles patch and oneloopmaster
`bd703d09…`. `source-bindings.json` includes exact hashes and line anchors;
original dependency files were also compared byte-for-byte to their pinned
Git blobs. The local patched counterparts are bound separately.

## Actual algorithm and precision

Full DP uses `TensorWorkspace<f64>::new_embedded` with default options:
method choice on; partial DD and whole-lane precision tiers off; direct gate
`1e-12`, method-choice gate `1e-15`, expansion boost 12. After a conditioning
refusal, method choice creates/reuses another workspace of the **same generic
scalar type**. Its triangle/box expansion executors also retain that type;
they use small-Gram/small-Cayley expansions and alternative PV, with bounded
boosts and refusal if their checks fail. No internal rotation certificate is
supplied. MadLoop's independent checks remain essential.

The default master policy in this path selects `DirectMasters::Native`:
`NativeEvaluator<f64>` at 53 bits, including the existing handwritten C0
choice. This is different from inferring a backend from the general
`MasterEvaluator` defaults. OneLOop already has same-type analytic alternatives
and a finite-result retry that snaps sufficiently small C0/D0 invariants after
nonfinite output; it is not a precision escalation. This audit does not change
or newly qualify that existing scalar policy. The journal records no whole-lane
tier event. `Direct` alone is not a precision certificate for a future option
set: enabling partial rescue can put DD work inside that route. Master-row
counters are especially unsuitable for this inference: they aggregate routes
and count rows even when served from the integral cache.

Pole-only DP uses its separate `DirectPoleWorkspace<f64>`, without these
embedded fallback options. It must stay unchanged in a finite-DP experiment.
The genuine QP entrypoint still parses 113-bit inputs and computes at 256 bits;
lifting already-rounded DP inputs into DD or MPFR is a different operation.

## What the actual refused visits permit

`classify.py` verifies and streams the frozen compressed journal, excluding
bootstrap, and retains all 454 joined rows in `classification.json`. The first
direct diagnostic is a positive, finite ill-conditioned Gram measure in every
row. Classifying its member set gives:

| Final result | Integral N/rank | First refused subset | Visits |
| --- | --- | --- | ---: |
| success | 4/3 | top box | 20 |
| success | 4/4 | top box | 118 |
| success | 5/4 | non-top triangle | 30 |
| success | 5/4 | non-top box | 152 |
| refusal | 4/3 | top box | 6 |
| refusal | 4/4 | top box | 26 |
| refusal | 5/4 | non-top box | 102 |

Partial DD currently requires binary64, no jet/dimension shift, a non-top PV
triangle or box, and a nonzero Gram solve before the conditioning test. It
does not rescue top boxes, exact singular pivots, bubble gates, a pentagon or
hexagon-specific gate, input validation or scalar-master failures. In the
method-choice workspace the Gram-expansion branch takes precedence over
partial DD. Expansion auxiliary failures can themselves involve high-rank
bubbles, which partial DD also excludes.

Thus 102 of the 134 refused visits have the static shape of the proposed
partial-DD target; the 32 top-box visits do not. Those 102 visits occur in all
13 rescued physical points, but eight points also have excluded top-box
refusals. Only points **21, 55, 65, 94 and 97** lack a recorded top-box refusal.
This is an eligibility filter, not proof DD would pass, the remaining reduction
would succeed, or the host would accept. Only the first failed subset is
reported; a later failure may appear after it is rescued. Visits are neither
distinct integrals nor time. The n4 public record has 14 DP/86 QP results but
no matching detailed cause journal here; this n3 classification must not be
extrapolated to those 86 points.

## Narrow candidates

1. **Partial DD only**, if separately reviewed: keep default method choice,
   gates, scalar kernels, exact 53-bit inputs and host checks; turn on the
   existing partial option in a private finite-DP context. Its threshold is
   `2^-66`, and rounded DD subset coefficients feed the remaining DP recurrence.
   Count actual subset attempts/successes/failures and numerical reuse, including
   calls later refused by a parent. A public `219` after DD is mixed 53/106-bit
   execution, not pure DP and not genuine QP. Check passes detach all numerical
   caches; no result may cross primary/check or finite/pole domains. Retain
   both-direction invalidation and failed-call clearing. Higher-rank and
   noneligible controls are necessary before any physical qualification.
2. **Avoid repeated unchanged store attachment**, as an independent allocation
   candidate: every cached primary call currently reattaches the same store.
   That re-registers nested gathers, allocates a `Vec`, and locks/searches the
   shape list, although registration survives event clearing and the bridge
   preregisters plans. Remembering the attachment identity/mode per workspace
   could skip this work without changing numerical keys. It must still attach
   on first use, detach when disabled, never attach to checks, and reset its
   bookkeeping when failure destroys the workspace/context. No implementation
   or n3 timing benefit has been demonstrated here.

Immutable full-DP plans are already shared by N/rank; primary and check mutable
workspaces are already separate. Method-choice/expansion executors and structural
variants are already retained. Recommending another plan cache without measuring
a specific missing reuse would duplicate existing work. Native failures destroy
both directional workspaces and clear their numerical store by design; simply
retaining failed state is not a safe optimization. Small per-call input/output
allocations remain a lower-priority profiling possibility, not an explanation
of the observed QP rate. Host early-exit research is owned separately.

Whole-lane embedded tiers are a broader unimplemented alternative, not part of
the narrow proposal: gates `2^-66`, `2^-216`, `2^-472` at 106/256/512 bits,
with exact lifting of the supplied 53-bit inputs. They need independent actual
tier telemetry and can add work before genuine-QP rescue; invalid inputs and
master failures are not conditioning retries. No threshold is relaxed here.

Reproduce the pure join with `python -B .../qcd-dp-policy-audit/classify.py`.
`verify.py --live` also checks source bindings. Final precise performance
measurement belongs to Hua-Sheng's local run; shared-host observations remain
provisional. No source count predicts a speedup or scientific acceptance.
