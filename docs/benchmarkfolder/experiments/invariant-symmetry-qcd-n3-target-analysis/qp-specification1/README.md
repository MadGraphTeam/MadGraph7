# Proposed QCD n3 higher-precision diagnostic

**Specification only.** No diagnostic source, compiler command, loader or native
worker is implemented or launched here. The closed offline analysis in
`../attempt1` remains immutable. Root review is required before implementation
and a later runtime admission is required before execution.

Investigate exactly three existing points, in order **1, 93, 39**: point 1 is
the quiet anchor, point 93 has the largest CutTools/non-CutTools finite
discrepancy, and point 39 has the largest non-CutTools discrepancy. Copy their
original decimal records without reformatting; `selected-inputs.json` retains
those records and their binary64 interpretations. No momentum, mass, width,
coupling, scale or scientific threshold changes are proposed.

## Numerical matrix and bounds

Use the current closed candidate's generated `loop_sm` subprocess and bound
reduction libraries for all four explicitly selected backends **1, 6, 8, 9**.
Run **CTModeRun=4** and **CTModeRun=5** in separate fresh processes. Both modes
use genuine QP numerator construction, with all four existing
`UseQPIntegrandFor{CutTools,Ninja,NinjaRS,TIRust}` settings true. Do not substitute
DD, a DP numerator with a QP reduction, an old all-512 library, or the adaptive
private bridge. TIRust uses the existing fixed-256 bridge recorded by the
closed case; it is not an independent truth reference.

The proposed matrix has **eight scientific arms**. Run each arm twice with
identical initialization and call order: one private driver with the passive
observer disabled, then enabled. This gives **16 serial fresh workers**,
**48 diagnostic physical calls**, and **96 answer-order rows**. The observed
half supplies 24 fully traced diagnostic calls; the plain half tests whether
observation changes the returned coefficients, status, actual inputs or IEEE
state. It is not a timing repetition. If an observer can safely remain enabled
for the existing plain control without querying a mutating API, its disabled
mode must only suppress record insertion and must forward each call once.

Each worker performs one recorded point-1 bootstrap followed by at most eight
recorded point-1 initialization calls, under the existing CTModeInit=1 and
normal run policy. Stop if initialization has not completed; do not fake a DP
failure. Only after initialization, set the requested fixed mode for the three
diagnostic calls. Turn off the public `FORCE_STABILITY_CHECK` double-call option
for these fixed-mode diagnostics and retain that deliberate diagnostic policy
in the record. The maximum is **192 public calls including initialization**.
Every actual public, repair, numerator, reducer and return call is counted;
the number of tensor calls is observed, never inferred from 48 public calls.
Unexpected, missing or truncated records invalidate the corresponding arm.

Proposed limits, subject to the later concrete source review:

- Serial private driver/observer compilation and relink: one job, at most
  120 seconds total and 4 GiB; retain original numerical objects unchanged.
- One loader check per unique linked executable, at most 30 seconds each;
  verify actual mapped libraries for every worker.
- Each worker: at most 180 seconds including bootstrap, 4 GiB, one thread and
  nonexclusive CPU affinity. Across the native stage: at most 3,000 seconds.
- Use the existing reviewed owned-process supervisor and common numerical
  lock. A busy lock is a scheduling hold before attempt creation. No automatic
  retries, adaptive expansion of the point set, or unrecorded continuation.
- Preserve all stdout/stderr, raw records, `param.log`, failures and command
  metadata. Compute trace capacities from the actual generated loop/helicity
  counts before freezing implementation; overflow must fail visibly.

## Input and precision evidence before any comparison

The current closed points file and previous points file are byte-identical
(`c1dc86e4…`). The closed candidate input record binds the physical parameter
card `5a535b70…`, original settings, generated source hashes, executable hash,
archives and runtime libraries. These are retained metadata facts, not a fresh
live-binary check. Before implementation, verify the original private donors
against those records and snapshot exact object hashes. Do not claim historical
object-byte identity where the retained record only binds source and final
executable. An unexplained mismatch blocks reuse and goes back to root.

Keep the copied parameter card, model and every numerical setting unchanged
except the explicit backend selection and the fixed diagnostic mode after
bootstrap. Preserve `MLStabThres=1e-6`, the existing helicity/color sums,
`ImprovePSPoint=2`, full finite/pole evaluation, all QP integrand flags, and the
original scale construction and `UPDATE_AS_PARAM` call. A fixed-mode diagnostic
does not acquire a new tolerance or become a replacement campaign policy.

Record integer representations of the actual Fortran inputs at these existing
boundaries, using passive forwarding observers without floating arithmetic or
exception-flag repair:

1. Public physical momenta, model masses/widths, scale and policy before/after
   each call; real Born entry inputs and returned Born value.
2. DP and MP momentum-repair inputs/outputs, actual MP masses used by repair,
   coupling-update inputs/outputs, and the canonical repaired MP momenta.
3. Actual QP numerator/Born inputs, helicity/color metadata and MP numerator
   coefficients. Demonstrate 113-bit Fortran values and actual QP call paths;
   a requested mode or status 120 alone is insufficient evidence.
4. Per-group native reduction inputs, including the actual precision of
   shifts, invariants, squared masses and scale. Preserve CutTools' DP-routed
   denominator inputs if that is what the unmodified generated interface uses;
   do not promote them on paper into identical native QP inputs.
5. Mode-4/5 native order and mass permutations, return status and Laurent values,
   host contraction and final returned values. Retain the QP values before any
   final conversion to the public binary64 answer where an existing boundary
   exposes them. Missing paths stay explicit coverage gaps.

CTModeRun=5 reverses the denominator order (for more than two propagators) in
the existing TIR routing. Capture both sides and use its exact documented
permutation/sign map to compare canonical physical inputs. Do not demand
literal equality of differently ordered native arrays, silently symmetrize or
round them, or call algebraically different effective invariants identical.
Compare coefficients as an adjudication only after the recorded physical,
Born/numerator and effective native inputs satisfy their respective equality
gates. Raw outputs with differing effective inputs remain retained descriptive
observations, not a same-input numerical agreement claim.

## External adjudication and failure handling

Require finite returned values and complete traces; retain **status 120 and
accuracy -1 as unchecked**, never as MadLoop-checked acceptance. Scan every raw
output line case-insensitively for `##W03\b` and retain line/text witnesses.
Any unstable/omitted-contribution warning, native failure, missing contribution
or incomplete execution excludes that result from coefficient comparisons.
The absence of a warning does not turn status 120 into a completeness or
stability proof. Preserve all IEEE flags, trapping and rounding state rather
than clearing original evidence.

Externally compare each backend's mode 4 versus mode 5, then all backend pairs
for each direction, separately for Born, finite, single and double poles and
each answer order. Retain absolute, raw relative and registered symmetric
differences. These diagnostics may test the `1e-8` target but cannot edit the
registered `3e-6` gate. Compare observed/plain records first and reject any
unexplained numerical/input/IEEE difference before using the trace as evidence.

Independent CutTools and Ninja QP with controlled input and direction agreement
would be stronger evidence than voting among ordinary outputs. NinjaRS/TIRust
agreement and a successful internal 512-bit evaluation alone are not truth.
If the modes or independent reducers disagree, leave the discrepancy open;
do not force fake DP failures, choose a favored direction, change thresholds,
patch physical inputs, or mark this case/the full campaign qualified.

## Can the current frozen worker run this unchanged?

**No.** The current and previous closed worker templates are byte-identical,
but the benchmark worker explicitly rejects status 120 in its warmup predicate
(`tirust_benchmark_worker.f:205` rejects `U=0`), and the Python parser accepts
only the checked aliases. It also redirects Fortran stdout to `/dev/null` after
initialization (`:117`), which would hide diagnostic `##W03` warnings. It has
no runtime command to switch modes after normal initialization and lacks the
required effective-input observer.

The generated numerical code itself supports fixed modes: the retained
`loop_matrix.f:913` selects CTModeRun, `:977` constructs MP phase space,
`:1012` takes the direct MP numerator path for modes at least 4, and
`TIR_interface.f:341` contains the MP mode-5 permutation. Their hashes match
the current closed candidate compilation record. This supports a minimal
**private driver/observer relink**, reusing the original numerical objects and
libraries, rather than regeneration or modification of generated physics.

The historical unchecked worker is also insufficient unchanged: although it
disables the public force-check flag, it suppresses stdout and lacks actual
input/precision traces. Neither historical native outputs nor the recorded
current executable (`c18c9dbd…`) authorize running an amended diagnostic now.
This specification reads only closed text sources/metadata and selected point
records; no donor executable was opened or loaded.
