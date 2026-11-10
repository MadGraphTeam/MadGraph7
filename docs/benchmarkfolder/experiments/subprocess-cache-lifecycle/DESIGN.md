# Generated subprocess cache lifecycle qualification

Status: source design for independent review; no generation, compilation, or
native evaluation has been performed. This is correctness qualification, not a
performance measurement. Production sources and reference repositories stay
unchanged. Work products are confined to this package and the ignored matching
`.tirust-build/subprocess-cache-lifecycle/` tree.

## Frozen implementation and actual installation

Materialize Git revision `059bb9c979194bfda783cad83443223def8df280` using `git archive`
inside the private work tree. Run its actual MG CLI from the repository working
directory with `PYTHONHASHSEED=0`, isolated XDG configuration and `--no_save` on all
`set` commands. The materialized model cache is private. Use the qualified
installation `.tirust-build/qp-transport-install/mg/HEPTools/tirust/lib`:

- DSO: `eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9`.
- Fortran archive: `0c86349a4fd0dadd918476731e899529ce1c7b17f5892a4af2c22dd762250c1a`.
- Configuration: `25182cfb31e68f8ff6246717e6819f0ec86232de88742f6e47e73676e6a41af9`.
- Compiler module: `d7838e75335c908a31e644d909bdd6dc71d98467ef8bdca3ca06f8bfad20eaf5`, also checked against the pinned configuration's artifact inventory.

Only prebuilt CutTools/IREGI archive/module artifacts are copied from the
existing development installation into previously absent private paths; they
cannot replace tracked snapshot files. CLI generation must preserve the main
configuration and loop_sm model byte inventories. Loader override variables
are recorded and removed for every worker and provenance query; the resolved
TIRust shared library must be the qualified installation above.

Generate `u u~ > z [virt=QCD] @101` and `d d~ > z [virt=QCD] @202` together,
with subprocess grouping disabled, in `loop_sm`. Require exactly two different
`proc_prefix.txt` values and verify the generated process identities. Build
model/DHELAS and the complete numerical objects using the generated makefiles.
Link one new driver with the existing `OLP_PROCESS` list and one shared copy of
MadLoopCommons/parameter reader, model, DHELAS and reduction libraries. No
concatenation of per-subprocess PROCESS lists, no duplicate parameter reader,
and no generated check_sa program are allowed.

## Physical amplitude protocol

Four fresh processes execute an identical prescribed call stream: plain/cache
0, plain/cache -1, observed/cache 0, observed/cache -1. Each starts from its own
fresh copied runtime-resource directory and the same physical parameter card.
Generated objects/executable inputs are shared read-only. Only backend 9 is
selected, full finite and both pole slots are requested, `MLStabThres=1e-6`,
automatic precision and forced stability checks remain enabled. Use the
ordinary generated amplitude and its real automatic entry/exit cache clearing.
Do not alter initialization COMMONs or bypass the generated lifecycle.

Warm A and B through their ordinary public amplitude entry points, with at
most 32 calls per prefix; retain every call and require identical warmup paths
between executable/cache variants. Each warmed prefix must reach a checked
steady result. Do not silently extend this cap or reuse warmed state from a
previous process.

Run A/B/A for each of: rest-frame on-shell momenta at nominal MZ; the same event
with doubled MU_R; a longitudinal boost beta=0.25; an alternate physical card
MZ=92 GeV with consistently regenerated 2-to-1 momenta; restored nominal card
and scale. Parameter changes use SETPARA, then MU_R assignment and
UPDATE_AS_PARAM. MZ variation changes external mass/kinematics, not internal
propagator masses; no stronger claim is made. Record exact binary64 input,
Born/finite/simple/double coefficients for every split-order column, returned
status and accuracy, all IEEE status flags, and context generation.

Call A_FINALIZE_TIRUST (not EXIT_MADLOOP, which also affects NinjaRS); require
generation=-1 without querying a statistics API that would recreate a context.
Then run B/A/B with the original event. B must recreate the shared context and
register its plans despite its retained per-prefix LAST_GENERATION. A must
also register its plans in the new generation. These steps test the actual
compiled per-prefix initialization routines.

A separately tagged real cached rank-zero tadpole seed before one automatic
amplitude call establishes nonempty incoming native cache state; the actual
entry boundary must remove it. Seed calls are not physical amplitude calls.

## Observation without generated-source changes

Use GNU linker --wrap for the actual C exports context_create/context_destroy,
clear_cache/register_plan/evaluate_dp/evaluate_mp. Every wrapper forwards its
original arguments exactly once to __real_*; no reduction is substituted.
The same generated numerical objects link both executables. The observer uses
a fixed-capacity integer-only journal with no allocation or file I/O inside a
wrapper. The driver flushes the journal only outside amplitude calls.

Only the clear-cache wrapper queries native cache counters immediately before
clear and after a successful clear. These queries cannot create a context.
They clear Rust's last-error string, but the wrapped clear operation itself
unconditionally clears that string. Never query counters after an evaluation
failure or another error-consuming path. Errors from statistics queries,
journal overflow, absent expected calls, observer-induced IEEE flags or
rounding changes fail qualification. No flags are silently repaired.

For every measured physical call, the trace must contain a real clear before
its first reduction and a real clear after its last reduction; the last clear
must leave both native cache counters empty. Enabled execution must retain
nonempty entries before at least one final automatic clear. Disabled physical execution
must never retain entries, except for the explicitly tagged forced-cache seed
which runs identically in every worker and must be evicted at the next event entry. Each evaluated tensor must have the requested
cache flag (the separate seed explicitly requests caching even in cache0), and repeated calls must not accidentally reuse another process's
Fortran tensor-index state. Cache-enabled versus disabled amplitudes require
exact binary64 outputs and status/accuracy parity; observed versus plain also
requires exact IEEE parity. Retain all failures without choosing a later run.

## Separate exact-key subsection

After physical calls, a bounded generated tensor-diagnostic sequence tests
DP and genuine QP repeated values while varying one of complex squared mass,
MU_R or supplied momentum increments, returning to earlier inputs. It uses
actual A/B generated diagnostic entry points and native cache counters.
Generated MP M2L and MU_R remain DP inputs promoted into QP; the test does
not claim 113-bit mass/scale transport through that generated boundary. Each
key action has its own uncached reference and changed-input coefficient witness.
Only the separate host tensor-index cache may be cleared between these calls;
no hidden native clear may occur. For these Born exports the generated host
clear routine is a dummy; no host-cache population claim is made. The automatic physical protocol never turns
automatic clearing off. This subsection proves native key behavior and does
not stand in for the physical event-boundary test. Its exact action table and
source must pass peer review before execution.

No rank-six extension is included: these inexpensive exports have smaller
rank capacity. Existing native core rank-six and compiled mapping coverage is
preserved, without claiming a generated native rank-six success matrix.

## Resource limits and provenance

One CLI generation, deadline 600 seconds. One serial source build, deadline
600 seconds, one assigned CPU. No Rust/dependency rebuild. Two driver/observer
links from identical numerical objects. Four native worker processes, at most
180 seconds each, serial under `.tirust-build/numerical-tests.lock`; no timeout
retry. At most 64 warmup + 19 measured physical calls per worker, plus the
bounded frozen key/seed actions. Journal capacity 32768 records per flush,
under 16 MiB; overflow is fatal. Expected output under 20 MiB. No performance
claim, so shared-host timing is recorded only as context.

The shared build deadline includes compiler identification, link-input queries,
runtime resolution, and symbol/relocation reports. Every subprocess starts in
an owned session/process group; on deadline expiration only that group receives
TERM and then KILL, with at most five seconds allowed for cleanup. No matching
or signalling of unrelated processes is permitted. The assigned CPU is 10.

Before compiling, freeze original generated-source inventory, complete driver,
observer and parser sources, protocol and cards. Retain exact commands, compiler
policies, make dry-run/object lists, numerical source/object/link-input hashes,
archive/module/DSO metadata and resolved runtime hashes. Recheck all protected
inputs after builds/runs. Verify the materialized source snapshot and main
configuration remain unchanged except private model caches/generated outputs.
Package portable text evidence and binary hashes; do not commit executables.
Root alone commits. Native execution remains held pending independent review
and a root-approved CPU/time slot.

Stages are explicit: `qualify.py freeze`, `materialize`, `generate`,
`prepare-build`, `build`, and `run`. No stage runs its successor. The initial
freeze binds this protocol, CLI template, driver, observer, parser, runner and
synthetic parser tests. The second freeze binds the actual CLI-generated and
rendered sources before compilation. `run` starts four fresh workers serially
under the shared numerical lock, retains every log including failures, verifies
all pre-existing runtime inputs afterward, rejects new runtime files other than
the two declared output logs, and invokes only the retained-text analyzer.
