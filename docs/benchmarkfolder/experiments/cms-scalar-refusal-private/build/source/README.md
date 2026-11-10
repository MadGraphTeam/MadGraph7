# Private CutTools scalar-classification refusal prototype

This source-only experiment addresses the nonzero-to-zero classifications
identified in the pinned CutTools scalar wrappers. It does not change the global
OneLOop classification setting, prescribe a replacement cutoff, or repair scalar inputs. The
production repository, installed archives, reference repositories, and frozen
point-20 evidence remain unchanged.

The intended bounded behavior is conservative: if the existing absolute
classifier would erase a nonzero input or a nonzero invariant-minus-mass
quantity, a new checked entrypoint reports the classification and withholds a
valid result. A near-on-shell rounding residue can therefore be refused even
when a more complete policy could justify treating it as zero. This is limited
support, not a universal numerical error bound or a new on-shell identity test.

## Source and numerical scope

`classifier/` contains additive preflight and checked entrypoints in isolated
copies of the pinned AVH modules. The target overloads are `a0_c`, `a0_r`,
`b11rc`, `b11rr`, `c0rc`, `c0rr`, `d0rc`, and `d0rr`, in both the binary64 and
binary128 modules. These cover the actual CutTools scalar paths and their real
mass counterparts; they do not assert coverage of every public AVH overload.
The actual CutTools wrappers pass complex squared masses even when their
imaginary parts vanish. The stand-alone real-mass overloads need separate tests.

The observer must follow each overload's actual metric and branch order:
complex modulus and sum-of-component magnitudes are not interchangeable. Box
soft-kinematic permutations and selected invariant-minus-mass comparisons are
part of the classification. The six box invariants are not all clamped at the
same source site. Original scalar inputs remain untouched; classification
copies and records are per call. No process-global observer store or result
cache is introduced.

The initial compiler proposal uses the already pinned GNU Fortran environment
and its existing `REAL(8)`/`REAL(16)` representation. The arbitrary-precision
AVH `mp` overloads, the QCDLoop scalar selection, and other compiler ABIs are
outside this prototype. A checked caller must refuse unsupported selections
explicitly instead of falling through to an unobserved scalar kernel.

`caller/` contains isolated checked scalar callers, the corresponding full
CutTools reduction control flow, and private MadLoop adapter templates.
Refusal must stop the scalar-to-coefficient-to-amplitude path before an invalid
scalar value is consumed. Automatic CutTools mode may attempt its existing QP
rescue after a DP refusal; a terminal refusal must remain unstable and reach
MadLoop's failure machinery. A refusal is never evidence for a stable zero.
The original symbols, call signatures, and ordinary entrypoints remain
available unchanged. Exact caller contracts and validity handling are recorded
in the generated component documentation and tests.

The proposed implementation reads the actual initialized threshold state; it
does not call `olo_onshell` to switch policy around an evaluation. Unsupported
initialization or policy states must be explicit. A full event buffer cannot
turn a refusal into success. Extra preflight arithmetic and its IEEE effects
remain an item for native qualification, even when input arrays are unchanged.

## Retained physical inputs

`retained-inputs/extract.py` verifies and reads the original point-20 scalar
log. `retained-inputs/point20-inputs.json` binds its source files and retains 18
selection labels covering 10 distinct scalar calls, including complex masses, a physical transfer near
`-2.5e-7`, small external residuals, and exact-zero masses. The full source log
contains 840 scalar calls and 2,708 input fields, of which 1,076 are invariants.

Selection labels describe raw values; they are not claims about an unexecuted
preflight result. Tokens are copied exactly as printed. The old log did not
include group or direction identifiers, so these are not reconstructed. The
trace was an unchecked fixed-QP diagnostic with promoted DP denominator
inputs. It cannot establish checked host acceptance, repaired transport, or a
complete amplitude result.
`retained-inputs/selection-scope.json` makes the repeated labels explicit; they
are not independent numerical samples.

The frozen causal evidence is
`docs/benchmarkfolder/experiments/complex-mass/direct-qp/cuttools-adjudication/`.
It separates scalar classification from legacy denominator precision: a lower
private cutoff removes the large discrepancy without changing scalar inputs,
while a separate QP denominator transport intervention removes the remaining
disagreement. Neither intervention is applied here.

## Qualification stages

1. Source-generation tests bind every transformed routine to the pinned source,
   prove that original entrypoints remain unchanged, enumerate intercepted
   nonzero-to-zero assignments, and check all scalar-to-coefficient barriers.
2. A bounded, separately authorized Fortran build checks the generated DP/QP
   modules, checked caller control flow, and standalone mock fixtures. It does
   not rebuild Rust or load backend 8 or 9.
3. Separately authorized mock executions cover real and complex branches,
   exact zeros, strict cutoff boundaries, every difference-classification site,
   event completeness, immutable inputs, unchanged legacy controls, global
   state invariance, and DP-refusal/QP-rescue/terminal-refusal routing. Native
   IEEE behavior must be observed, not inferred from source alone.
4. Any later real scalar or amplitude run requires its own explicit scope and
   admission. Failed and unsupported cases remain in the evidence. No current
   source-only result qualifies the required complex-mass stress campaign.

`prepare_plan.py` freezes the final sources and the two component recipes into
one proposal. `control.py verify` checks its hashes and compiler environment
without launching a process. The sole execution action is `control.py build`:
17 sequential compiler/linker commands on CPU 8, one job, with a shared
180-second deadline and no retry. It reuses the previously reviewed owned-child
termination and reap logic. The actual AVH and checked CutTools objects are
typechecked; only recording mocks are linked into test executables.

The controller has no mock, scalar, amplitude, loader or Rust execution action.
The component plans describe separately bounded future mock runs. All failures
and partial outputs from an authorized build must be retained. Source inventory
and command availability do not themselves authorize a compiler or native
execution.
