# Resolved recurrence export

`pyamplicol export-recurrence LC_DIR -o capture.json` exports the portable
batched f64 program of an LC recurrence artifact. It does not evaluate a
phase-space point. By default the command exports every stored process,
alias, and process-group member, with every public colour flow. Select subsets
with repeated `--process ID` or `--color-flow ID` options.

Use `--model-parameters parameters.json` to supply the same JSON parameter
mapping accepted by `Runtime.load`. Resolution happens after these values are
applied. In Python, `runtime.export_recurrence(color_flows=None)` returns a
dictionary for the selected process. Rust callers use
`NativeRuntime::export_recurrence(colours)` or `export_recurrence_json(colours)`.
The API is available in ordinary builds; no capture feature, environment
variable, or recording session is needed.

The supported executor set is the portable JIT backend and Rusticol's built-in
intrinsics. A native C++/assembly executor has no stored portable operation
stream and is rejected with a request to regenerate using JIT. Contracted
colour, loop, and non-recurrence execution modes are outside this format.

## Identity and numbers

Consumers must check `schema_version` (currently `1`) and
`execution_semantics` before interpreting instructions. `rusticol_version`
identifies the producer. `artifact_id` is the original artifact identity.
`parameters` records the model values used to resolve the programs. Complex
numbers are objects with `re` and `im` f64 values. A reader must preserve f64
round trips, including signed zero.

`kernels` contains each referenced saved `.plane.symjit` application, deduplicated
by its `sha256`. `application` is a JSON array of byte values, containing the
original saved application without recompilation or transformation. It can be
written directly as a binary `.plane.symjit` file. The saved application's
operations, including its FMA operations, determine kernel arithmetic.

## Programs and storage

Each element of `programs` identifies a `process_key` and `colour_id`. It also
records the process expression, its representative, `external_pdgs` in program
leg order, and `caller_external_pdgs` in caller leg order. `external_count`
gives the leg count. All array indices and leg indices are zero based.

`crossing`, when present, maps every `program_leg` to a `caller_leg` and a
momentum `sign`. Apply it before constructing momentum forms. The forms
already incorporate the selected flow's replay permutation. Every form is an
ordered list of `(leg, coefficient)` terms. Evaluate each Lorentz component
from positive zero, preserving term order.

`currents` describes semantic currents, their physical component range,
component count, initial state, and lifetime. Physical ranges can overlap
because the scheduler reuses storage after a current's last use. Instructions
address physical complex component offsets, not semantic current IDs. Allocate
`current_component_count` complex components and `amplitude_count` complex
amplitudes for a run. The body explicitly clears live destinations before
accumulating into them. `physical_type` is null when the stored binding does
not identify it; the component count and instruction operands remain
authoritative.

`scales` and `factors` contain resolved complex values. Instruction operands
also carry their resolved values, so a consumer never evaluates a model
expression. `alpha_s` identifies the strong-coupling parameter and its value
when known. `squared_amplitude_power` is present for a homogeneous QCD order;
null means no single exponent was established.

## Runs and instructions

Initialize every helicity output to positive zero. Execute `runs` in order,
using a fresh arena lifetime for each run. Execute its `body`, then its
`reduction`. Outputs are in `helicity_outputs` order; each descriptor includes
the stable helicity ID and every external leg's helicity. Structural zeros
remain in this axis even when they require no instructions.

| Instruction | Operation |
| --- | --- |
| `clear` | Write positive complex zero to the specified component range. |
| `clear_amplitude` | Write positive complex zero to the amplitude. |
| `source` | Evaluate the named wavefunction family with the resolved orientation, helicity, chirality, mass, and momentum form, then store with the given factor and embedding. |
| `vertex` | Evaluate the named versioned intrinsic with its input currents and momenta, coupling, and factor. `mode` specifies set or add. |
| `propagator` | Apply the named versioned propagator in place. |
| `scale` | Multiply each component in place by the given complex factor. |
| `copy` | Multiply the source range by its factor and store in the destination range. |
| `scatter` | Scale a fanout representative and set or add its result with the runtime's specialized factor arithmetic. |
| `closure` | Accumulate the ordered component contraction into an amplitude. |
| `kernel` | Evaluate the saved application using the explicit ordered input and output bindings, then apply its storage adapter. |

Source families are `scalar`, `weyl`, `dirac`, `vector`, and `spin2`.
Orientations are `particle`, `antiparticle`, and `self_conjugate`. Source
helicity and chirality have already been crossed. Source formulas use
`mass.re`; `mass.im` records the parameter's imaginary component but is not
used by these wavefunctions. A nonempty `embedding`
scatters the source formula's components into the arena; a null
`source_component` writes positive zero. Its factors already include the
crossing phase. Use the versioned Rusticol wavefunction formulas in
`rust/crates/rusticol-core/src/wavefunctions.rs` and the source adapter
`rust/crates/rusticol-core/src/evaluator/recurrence_source_direct.rs`.

Kernel `binding.inputs` is in saved-application input order. Each input reads
the real or imaginary part of a physical current or amplitude, a Lorentz
component of a momentum form, or a resolved real `literal`. Outputs specify
their exact current/amplitude component and real/imaginary plane. Snapshot
inputs before writing aliased outputs. `direct` records whether Rusticol
bypasses the complex output-scaling adapter; it is significant even for a
unit factor because arithmetic on signed zeros can differ.

## Arithmetic order

Do not reassociate sums or introduce fast-math contractions. Ordinary complex
multiplication means `(ar*br - ai*bi, ar*bi + ai*br)` with separate operations.
An explicit FMA is one fused operation. Intrinsic names include their version
and refer to the formulas in
`rust/crates/rusticol-core/src/evaluator/recurrence_intrinsic_direct.rs`.
Changing the operation order of those formulas requires a new execution
semantics version.

A vertex first computes `scale = complex_multiply(coupling, factor)`, then
evaluates the named intrinsic's base result, multiplies that result by
`scale`, and sets or adds it. The exported coupling already includes the
runtime's coupling-resolution arithmetic. Do not apply another unit scale.
In-place `scale` uses factor-first multiplication
`(fr*vr - fi*vi, fr*vi + fi*vr)`. Certified `copy` and kernel adapters use
value-first multiplication. Sources use factor-first multiplication after
the wavefunction formula (or the precombined embedding factor for union
sources). Operand order also matters for signed zeros.

For a closure, process `component_factors` in order. First multiply the
component factor by the row factor. Multiply the left component by the right
component without conjugation. Multiply that result by the combined factor,
then add it to the amplitude. Each multiplication is the ordinary complex
multiplication above. Accumulate into the amplitude separately for every
component; summing the contraction first changes rounding.

Fanout is explicit: an intrinsic or kernel computes its representative into
scratch components, followed by ordered `scatter` instructions. A scatter
with a real factor uses componentwise multiplication, with unit and minus-unit
factors handled by copy and negate. A purely imaginary factor maps `(re, im)`
to `(-im*k, re*k)`; otherwise use ordinary complex multiplication. Preserve
these shortcuts and the scatter's set/add mode. Certified `copy` instructions
use ordinary complex multiplication, except a zero factor overwrites with
positive complex zero.

Reduction terms preserve the runtime's amplitude and helicity-orbit order.
When `apply_amplitude_scale` is true, apply the term's replay amplitude scale
even if it is `(1,0)`. Otherwise leave the amplitude untouched. Calculate
`fma(real, real, imaginary*imaginary)`, multiply by its weight, and add to its
helicity output. After all runs, apply `group_factor` once: if numerator and
denominator differ, multiply each output by the numerator converted to f64,
then divide by the denominator converted to f64. Equal values require no
arithmetic. Precomputing a ratio would change rounding.

This export describes batched execution. The one-point path may use different
native interaction rewrites. Use batched `evaluate_resolved_f64` as the
reference when comparing consumers. Exporting before or after evaluation does
not change the program; point counts and tile sizes are not export inputs.
