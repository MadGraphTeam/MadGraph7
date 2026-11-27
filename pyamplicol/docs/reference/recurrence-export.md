---
title: "Recurrence Export"
nav_order: 6
parent: "Python API"
---
<!-- SPDX-License-Identifier: 0BSD -->
# Resolved recurrence export

`pyamplicol export-recurrence LC_DIR -o capture.json` exports the portable
batched f64 program of an LC recurrence artifact. It does not evaluate a
phase-space point. By default the command exports every stored process,
alias, and process-group member, with every public colour flow. Select subsets
with repeated `--process ID` or `--color-flow ID` options. Colour-flow IDs are
per process (they name leg permutations of that process), and `--color-flow`
applies to every selected process: on a multi-process artifact, combine it with
`--process`. An ID that a selected process lacks is refused with an error that
names the process and lists its flow IDs.

The CLI writes compact JSON by default. `--indent 2` enables readable output
for inspection. Schema 2 replaces schema 1 directly: consumers should update
their reader instead of treating the two layouts as interchangeable.

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

Consumers must check `schema_version` (currently `2`) and
`execution_semantics` before interpreting instructions. `rusticol_version`
identifies the producer. `artifact_id` is the original artifact identity.
`parameters` records the model values used to resolve the programs. Complex
numbers are objects with `re` and `im` f64 values. A reader must preserve f64
round trips, including signed zero. Every number is finite: JSON has no NaN or
infinity, so an export whose resolved values include one (for example from a
model parameter) is refused with an error that names the quantity, such as a
parameter or a program's `body[12].factor.re`.

`kernels` contains each referenced saved `.plane.symjit` application, deduplicated
by its `sha256`. `application_base64` contains the original saved application
encoded with standard padded base64, without recompilation or transformation.
Decode it to obtain a binary `.plane.symjit` file. The saved application's
operations, including its FMA operations, determine kernel arithmetic.

## Programs and storage

Each element of `programs` identifies a `process_key` and `colour_id`. It also
records the process expression, its representative, `legs` in program leg
order, and `caller_legs` in caller leg order. Each leg has `pdg`, `name`, and
`role` (`initial` or `final`). `external_count` gives the leg count. All array
indices and leg indices are zero based.

The CLI emits a full program for each needed representative. An alias or
process-group member instead has `program_reference`, an object with the
representative program's `process_key` and `colour_id`. Its own identity,
caller legs, crossing, group factor, and helicity output descriptors remain
on that entry. Resolve the reference to obtain the body, momentum forms,
storage layout, kernel bindings, and reductions; use the member entry's
crossing and group factor and label outputs with its helicity descriptors.
These replace the representative's corresponding fields. Referenced
representatives are included even when `--process` selects only a member.
The Python and Rust per-runtime APIs return a full selected-process program.

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
amplitudes for a run. `current_component_count` includes the fanout scratch
suffix. Those temporary components are absent from `currents`; vertices or
kernels write them and subsequent scatters read them. The body explicitly
clears live destinations before accumulating into them. `physical_type` is
null when the stored binding does
not identify it; the component count and instruction operands remain
authoritative.

`first_use` and `last_use` are inclusive dependency-stage events. Sources
start at zero; a current supported on `k` external legs starts at `k - 1`.
Its last use is the latest child stage that reads it, or the final closure
event, one past the largest current stage. These values describe the stored
plan's conservative liveness; they do not index plan rows or exported body
instructions, and fanout expansion does not renumber them.

`scales` and `factors` contain resolved complex values. Instruction operands
also carry their resolved values, so a consumer never evaluates a model
expression. `alpha_s` identifies the strong-coupling parameter and its value
when known. The name comes from the model's explicit alpha_s declaration,
its external SLHA `SMINPUTS[3]` parameter, or provider metadata, when that
names a parameter of the runtime. Otherwise, a runtime that defines
`normalization.alpha_s_me_check` names it: the built-in models, compiled or
not, apply `(4 pi alpha_s)^p` through that parameter's normalization, which is
part of every reduction `weight`. `squared_amplitude_power` is
present for a homogeneous QCD order; null means no single exponent was
established. The parameter's name and value remain available in that case.

## Runs and instructions

Initialize every helicity output to positive zero. `body` is stored once per
program. For each entry of `runs`, execute that body, using the run's `sources`
table, then execute its `reduction`. A `source` instruction's `source` index
selects a complete source descriptor from that run's table. Only the source
variants and reductions vary between union runs.

Runs may reuse one arena and execute in sequence: the exported clears and
initializing writes establish every component that a run reads. A consumer
may concatenate the runs into one execution stream while preserving their
order and source-table selections. Outputs are in `helicity_outputs` order;
each descriptor includes the stable helicity ID and every external leg's
helicity. Structural zeros
remain in this axis even when they require no instructions.

| Instruction | Operation |
| --- | --- |
| `clear` | Write positive complex zero to the specified component range. |
| `clear_amplitude` | Write positive complex zero to the amplitude. |
| `source` | Read the indexed descriptor from the run's `sources`, evaluate its wavefunction, then store with its factor and embedding. |
| `vertex` | Evaluate the named versioned intrinsic with its input currents and momenta, coupling, and factor. `mode` specifies set or add. |
| `propagator` | Apply the named versioned propagator in place. |
| `scale` | Multiply each component in place by the given complex factor. |
| `copy` | Multiply the source range by its factor and store in the destination range. |
| `scatter` | Scale a fanout representative and set or add its result with the runtime's specialized factor arithmetic. |
| `closure` | Accumulate the ordered component contraction into an amplitude. |
| `kernel` | Read the indexed shared binding, resolve its relative inputs and outputs against the row's operands, evaluate the saved application, then apply its storage adapter. |

Source families are `scalar`, `weyl`, `dirac`, `vector`, and `spin2`.
Orientations are `particle`, `antiparticle`, and `self_conjugate`. Source
helicity and chirality have already been crossed. Source formulas use
`mass.re`; `mass.im` records the parameter's imaginary component but is not
used by these wavefunctions. A nonempty `embedding` scatters the source
formula's components into the arena; a null
`source_component` writes positive zero. Its factors already include the
crossing phase. Use the versioned Rusticol wavefunction formulas in
`rust/crates/rusticol-core/src/wavefunctions.rs` and the source adapter
`rust/crates/rusticol-core/src/evaluator/recurrence_source_direct.rs`.

Each program's `kernel_bindings` stores every distinct binding once, including
the saved application's SHA-256, relative `inputs` and `outputs`, and `direct`.
A kernel instruction names its `binding` ID and explicitly supplies
`destination`, `amplitude`, `left`, `right`, `left_momentum`,
`right_momentum`, `mode`, and `factor`. `amplitude` distinguishes an amplitude
destination from a current component base; null parents and momentum forms
are absent operands.

Binding inputs are in saved-application input order. A `parent` projection
selects operand 0 (`left`) or 1 (`right`), plus a component offset and a real
or imaginary plane. A `destination` projection selects a component offset
from the row's destination. `momentum` selects operand 0 or 1's momentum
form and a Lorentz component. `literal` holds a resolved real value; `factor`
selects the real or imaginary part of the row factor. Outputs use relative
current/amplitude projections in application output order. Snapshot inputs
before writing aliased outputs.

With `direct: true`, store the kernel's outputs unscaled, whatever the row's
factor; `mode` still applies, setting or adding those outputs. Otherwise
multiply each complex output by the factor before setting or adding it.
This distinction matters even for unit factors because arithmetic on signed
zeros can differ. Rusticol sets `direct` only on finalization rows whose
factor is exactly `(1, +0)` and whose outputs overlap neither the inputs nor
each other; such rows always use `mode: "set"`.

## Arithmetic order

Do not reassociate sums or introduce fast-math contractions. Ordinary complex
multiplication means `(ar*br - ai*bi, ar*bi + ai*br)` with separate operations.
An explicit FMA is one fused operation. Intrinsic names include their version
and refer to the formulas in
`rust/crates/rusticol-core/src/evaluator/recurrence_intrinsic_direct.rs`.
Changing the operation order of those formulas requires a new execution
semantics version; the golden vectors below enforce this.

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
not change the static program; point counts and tile sizes do not specialize
its body.

## Golden vectors

`execution_semantics` is a bit-exact contract, and committed golden vectors
enforce it. `rust/crates/rusticol-core/tests/fixtures/recurrence_export_golden.json`
records, for the current `execution_semantics`, the inputs and the expected
output bits of every exported instruction kind that carries arithmetic: each
vertex and propagator template, every source family and orientation (massless
and massive, positive and negative energy, the axis and rest branches, and an
embedded source), `scale`, `copy`, every `scatter` factor shortcut in both
modes, `closure`, the kernel input projections and output adapter (direct and
scaled, set and add, current and amplitude destinations), momentum forms with
and without crossing, reductions, and group factors. f64 values are IEEE-754
bit patterns written as 16 hexadecimal digits. Instruction cases execute one
instruction on shared initial arenas and momentum forms over five points, so
both the paired and the single-point paths are pinned, and list the arena
slots whose bits change. Kernel cases use identity kernels (`identity-N`
copies N complex inputs to N complex outputs; `identity-N-to-M` copies the
first M), so they pin the projections and the adapter rather than any saved
application.

The Rust test
`recurrence_export_golden_vectors_pin_the_execution_semantics` replays every
case from the fixture's own inputs through the export interpreter. A differing
bit fails with "the execution semantics changed: bump the version string and
regenerate". A new template, source family, orientation, or instruction kind
cannot be added without new cases: the generator enumerates them exhaustively.
After bumping `EXECUTION_SEMANTICS` in
`rust/crates/rusticol-core/src/engine/recurrence_export.rs`, regenerate the
vectors with:

```console
cargo test -p rusticol-core --lib regenerate_recurrence_export_golden_vectors -- --ignored
```

The generator refuses to change an existing case while the version string is
unchanged; it may add cases. External runners can replay the same fixture to
check their implementation.

The pinned arithmetic is binary64 addition, subtraction, multiplication,
division, square root, sign operations, and the reduction's explicit FMA. These
are correctly rounded on every IEEE-754 platform; Rust never contracts
expressions into FMAs, and no transcendental library function is involved.
Every pinned input and output is finite, so platform-specific NaN payloads do
not occur. The bits are therefore platform independent, and the test runs
unrestricted. The fixture was generated on x86_64 Linux, which CI also uses.
Golden vectors detect formula changes that alter the bits of their inputs. The
inputs have full mantissas and cover every branch to make an undetected change
unlikely, but they are samples rather than a proof.

## Optional reference values

`--reference-points FILE` adds reference outputs using caller-order momenta.
The JSON input is either a shared batch shaped `[point][leg][E,px,py,pz]`, or
an object mapping exported `process_key` values to such batches, even when
`--process` selects an expression. Each selected batch must be a nonempty
JSON array. In Python, pass
`reference_momenta=batch` to `runtime.export_recurrence()`. Evaluation occurs
only when these values are requested. A single input point is internally
duplicated to select batched arithmetic, and only the requested point is
retained in the export.

Each selected program or member entry gets `reference_values` with `momenta`,
`helicity_ids`, and `values` shaped `[point][helicity]`. The helicity axis
matches that entry's `helicity_outputs`. These f64 values let a consumer test
its implementation on a machine without Rusticol. The field is absent from
ordinary static exports.

## Consumer parity tests

The Rust export interpreter test covers every stored process, alias, and
process-group member when `RUSTICOL_RECURRENCE_ARTIFACT` names a generated
artifact, or a list of artifacts separated like `PATH` (`:` on Unix). It also
reconstructs member programs from their representative references before
comparing every colour and helicity bit for bit. Set
`RUSTICOL_RECURRENCE_PROCESSES` to a JSON array of process IDs to restrict that
traversal. Without an artifact the test reports that it skipped. CI runs it
against the artifacts that the export integration tests generate: `d d~ > z g`
in the topology-replay and all-flow-union layouts, and a grouped
`q q~ > t t~ g` set with an explicit alias, flavour-blind process-group members
(including crossed initial states), and massive top sources.

```console
RUSTICOL_RECURRENCE_ARTIFACT=/path/to/artifact \
  cargo test -p rusticol-core --lib recurrence_export_generated_artifact_matches_every_helicity_bit -- --test-threads=1
```

The ordinary focused export tests exercise zero-factor certified copies,
in-place scaling, spin-2 sources, direct and scaled kernels with both write
modes, nonunit replay multiplicities, group factors, and imaginary/general
fanout scatter factors without requiring an external artifact. Their expected
values come from Rusticol's own source, certified-reuse, identity-finalization,
closure and fanout-scale executors, its kernel output adapter, and its replay,
reduction and group-factor arithmetic.
