# Private current-source trace executor — source review only

This is the concrete successor to sealed proposal
`31ea5e9a14851d436f22bb76dd620a799ba2c3c034bae28faf43c31da26f053f`.
That proposal and historical packet `1b57d926…` remain unchanged. No CLI
generation, compiler, library loader, amplitude evaluation or Git command has
been executed while preparing this packet. `work/` and `results/` are absent.

The active amendment is `amendment.json`: modes **1 and 4 only**, five saved
process/point pairs and three variants, hence **30 fresh workers and at most
60 outer amplitude entries**. Each worker makes one ordinary-point bootstrap
with tracing off and one measurement. The mode restriction, subsequently
superseded five-bootstrap amendment and five-bootstrap driver are retained as
unexecuted history. Modes 2/5 retain symbolic source coverage and remain native
unqualified.

`LoopPoleMode=0` is deliberately preserved. In the current generated code it
temporarily bypasses helicity/group filters and restores the saved INIT flags
after the call. These restored flags are observations, not a readiness gate.
`bypass_contract.py` checks that source branch, full-helicity `USERHEL=-1`,
positive `CTModeRun` routing, restoration and unchecked return codes before
admitting a generated driver. The private driver sets forced stability checking
false so the threshold wrapper does not add its forced second call. Expected
fixed-mode codes are DP **110** and QP **120**; they do not establish numerical
stability or an independent accuracy result.

The source anchor remains `0fcd2cd873d61d134837e1d9e87afb13a47824fd`, inventory
`afb6e0be…`. The reused EW export was generated at `b6302b3e…` with exactly that
numerical inventory. `ew-source-bindings.json` hashes all 427 current generated
source/include entries before the symbolic consumer parses them. The current
review artifacts are **`ew-derived-current/`**. `ew-derived/` is the earlier
unexecuted emitter draft. The current artifacts contain newly derived insertion
patches, exact stripping proofs, group/amplitude identities and the signed
momentum mapping. The active group filter is EW 5; fresh CMS must rederive and
check groups 3/4/8/12/24 and their signed corners before any build.

`inputs/` contains literal cards and byte-exact one-point slices, their donor
pins and the complete ordered 30-request list. It preserves physical cards,
mass policy and all unrequested numerical settings, including CMS
`MLStabThres=1e-3` and `OSThres=1e-8`. No floating point expression, threshold or
reducer guard is changed. Absent model/offshell/onshell metadata stays unknown.
The earlier `initial-one-bootstrap-inputs/` directory is retained history, not
the active request list.

`trace_current.py` reuses the sealed narrow symbolic consumer and insertion
primitives against current bytes. Added blocks must strip to the original
files byte for byte. Trace schema V2 records point, per-loop serial, actual
LOOPNUM/group, mode, rank, propagator count, pair/branch and split order; it then
records integer bit words for complex W/MP_W, real corners, the unsimplified
conservation residual, shifts, original DIFFSQ/default invariants, actual snap
branches and final TIRust inputs. The returned TIRust integer status is recorded
after evaluation. It is not the final host STABLE result: early exits or later
IEEE rejection must remain incomplete/failing observations. No failed tensor
buffer is read.

The controller reuses pinned process ownership, deadlines, cleanup, compiler
inspection and linker-closure helpers. It accepts only reviewed source hashes,
uses the shared numerical lock for every native stage, and stops between stages:

1. `generate` materializes exact `0fcd` into a new private tree, copies pinned
   prerequisites/model and performs one CMS CLI export. Generation may compile
   dependencies and therefore is a native-admitted stage. Shared installed
   dependencies and configuration are checked before/after.
2. `derive` is source-only: hash the fresh CMS export, validate source semantics,
   rederive both trace patches, render both drivers and seal a second index.
   **Root must review that derived index before `build`.**
3. `build` reuses verified current EW MODEL/DHELAS archives and binds its module
   inputs; it builds fresh CMS common objects and private process/driver
   objects. The instrumented variant changes only CT/TIRUST/driver objects.
   Every link has prebound inputs and a checked link map. Loader admission is
   explicit and belongs to this admitted stage.
4. `ordinary` runs CMS1/EW1 parity (12 workers). Only successful exact
   unmodified/disabled/enabled observation and actual-mode admission permits
   `stress`, which runs CMS20/21/EW2 (18 workers).

Inspect the dormant plan from the repository root:

```sh
source tools/tirust_env.sh
python3 -B .tirust-build/kinematic-provenance-current-executor-1/source/run.py
```

Future admitted commands add the stage, `--execute` and
`--reviewed-source-index SHA256`; build/run also require
`--reviewed-derived-index SHA256`. The default command launches no child, and
there is no implicit continuation or in-place retry. No reviewed-index value
has yet been supplied for execution.

Worker output is bounded: 180 s bootstrap/protocol phase, 300 s measurement,
600 s outer worker limit, 8 GiB address space, zero core dumps, 64 MiB per trace,
32 MiB stdout and 1 GiB total retained trace allowance. A full 64 MiB allowance
is reserved before each launch. Stage caps include a 60 s cleanup reserve.
Runtime providers are observed at READY and DONE; BYE is not a valid mapping
boundary because the child may have exited. Variant executables are separately
bound. Overflows, incomplete phases, unknown/failing outputs and unlaunched
requests are retained explicitly. Timing fields are incidental protocol data;
this experiment makes no performance claim.

Exact parity includes every split-order answer, raw answer bits, original FPE
flags, restored INIT flags, actual mode, bootstrap evidence and warning evidence.
Enabled traces need the complete target-group stage sequence. Equal failed
outputs remain failures; unchecked fixed-mode results are never promoted to
independent accuracy qualification.

Pure Python tests check byte preservation, actual EW group identity, complex
word counts, residual retention, card/point derivation, bypass semantics,
driver limits, mode/split-order admission and the late stdout-overflow race.
They do not qualify native compilation, storage layout, runtime closure,
observation parity, CMS20/21 or EW group 5 numerically. Those remain pending.
