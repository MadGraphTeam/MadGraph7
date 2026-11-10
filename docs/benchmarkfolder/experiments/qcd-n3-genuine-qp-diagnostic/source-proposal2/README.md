# Closed QCD n3 genuine-QP diagnostic — source proposal 2

This is a **source-only proposal**, implementing the frozen specification
`invariant-symmetry-qcd-n3-target-analysis/qp-specification1`, index
`741813a73a19891091f5597d8032033e2e1723866c45b10f7f8ebacbe6c1c9a9`.
No compiler, linker, loader, generated amplitude or reducer has been run for
this proposal. Python tests use synthetic byte traces, temporary files and
fake supervised children. A passing test is not scientific qualification.
The prior offline analysis, portable analysis, specification, production code,
main campaign, stress pipeline and CMS evidence remain unchanged.


## Mechanical environment amendment

Proposal1's source index `bf2cd097…` and its actual failed build1 remain
unchanged. `prior-build1/` retains exactly its four files: plan, first-command
metadata, empty diagnostic log and failed build result. Command1 returned
SIGSEGV (-11) after 0.247 s, before any compiler object or diagnostic appeared.
No numerical worker ran. The precise crash location was not observed.

Read-only ELF inspection shows host taskset/prlimit use the glibc2.42
interpreter, whereas the closed donor executable uses glibc2.40. The failed
command exposed the target's old-libc LD_LIBRARY_PATH to the host utilities.
`elf-environment-evidence.json` binds that static evidence; it supports the
loader-mismatch diagnosis without claiming an observed crash-site map.

Only launch mechanics and output namespaces change. Driver, both observers,
analyzer, scientific fixtures and owned-process supervisor are byte-identical
to proposal1. The new one-shot directories are **build2** and **run2**; they
never reuse or rewrite build1. The four compiler/link commands keep their
numerical objects and scientific flags. A fresh binary must have exactly the
donor's PT_INTERP, checked before a build can pass and again before execution.
No dynamic-linker flag is added speculatively.

Compilers, taskset, prlimit and the hash-bound isolated Python start with all
LD_* variables removed. `exec_target.py` verifies target bytes and PT_INTERP,
then sets only the bound native LD_LIBRARY_PATH immediately in **execve of
the target**. It does not fork, invoke a shell or launch ld directly. PID,
process-group ownership, target argv and final /proc executable identity stay
compatible with the existing supervisor. The loader check uses the same target
execve with LD_TRACE_LOADED_OBJECTS=1; ordinary numerical execution removes that
flag. This trace stage does not enter the Fortran driver.

Historical ldd metadata contains both the host glibc2.42 loader and the donor's
actual glibc2.40 interpreter. All historical paths/hashes remain bound. The
new target closure excludes **only** the exact known host ld-linux entry;
all other fifteen files remain mandatory. The target interpreter and derived
native library path are explicit. A loader-trace disagreement, missing library,
wrong interpreter or unexpected actual ELF map fails; there is no blanket
allowance for missing dependencies.

`prepare.py` is retained as proposal1 provenance and was not rerun here.
`mechanical-change.json` and `mechanical.patch` expose the source delta.
The new mock tests exercise environment separation, delayed execve, exact
argv, trace-only flags, closure mismatch and wrong-interpreter/hash rejection.
They execute no ELF, loader, compiler or reducer. Source review/seal is separate
from any later root admission to build2 or run2.

The future experiment has exactly 16 fresh, serial workers: backends 1, 6,
8 and 9, each in fixed mode 4 and mode 5, each with observation disabled and
enabled. One privately linked executable uses the same numerical objects and
pass-through wrappers in both observations; the flag controls insertion into
the journal. Each worker uses one original point-1 bootstrap and at most eight
point-1 initialization calls, then points **1, 93, 39** in that order. The
maximum is 192 public calls, including 48 diagnostic calls and 96 returned
answer-order columns. Initialization uses the ordinary checked policy; only
afterward does the driver disable FORCE_STABILITY_CHECK and select mode4/5.
There is no fake failure, policy threshold change, precision substitution or
adaptive bridge. All four QP numerator flags stay true; MLStabThres remains
1e-6, ImprovePSPoint remains 2, and complete finite/pole evaluation remains on.
The physical parameter card, decimal point records and three color/helicity
resources are exact closed-input copies. The only card edit is backend ID.

`prepare.py` performed read-only binding against the closed candidate input
record `2cfa4187…` and the specification. `bindings.json` identifies all source,
compiler, archive, runtime and donor object dependencies. The existing closed
executable matches its recorded hash. The **20 individual numerical object
hashes are current donor snapshots**, because the historical record did not
hash each object. We do not claim retrospective per-object provenance. The
proposal reuses those objects unchanged and makes no generated-source edits.
`shapes.py` checks COMMON order, scalar kinds and extents against generated
sources. The actual QCDn3 model has **34 complex couplings**, and the observed
DP/MP model snapshot has **254 int64 words**. `SETPARA` in `rw_para.f` forces
loop initialization before `COUP`; `couplings.f` initializes MP parameters and
MP_COUP1/2/3. Card policy fields are not read before the first public call.

## Observations and admissible comparisons

The C observer performs integer/byte copying only and forwards each wrapped
call once. The Fortran observer uses TRANSFER and read-only IEEE queries. It
neither resets flags, trapping or rounding nor queries error/cache APIs.
In particular it does not clear a reducer's error message by querying a
statistics getter. Failure output buffers are not read. The mode-order
observer passes dummy PDEN through without reading it.

The actual boundaries are public entry/return, real Born entry/return, DP and
MP repair, MP coupling update, MP Born/helicity/numerator construction,
canonical MP coefficients, per-group native reductions, QP CutTools numerator
callbacks, MP mode routing, and the host's contracted group results/stability
array. The parser checks nonzero canonical coefficient groups against actual
native group IDs. The array extent 116 alone is not evidence of 116 evaluations;
zero-coefficient groups without a native call remain explicitly listed.
It records all 70 coefficient slots but labels only binomial(rank+4,4) as
rank-valid. The actual observed call counts, helicities and group coverage are
reported, never inferred from the number of amplitude calls.

Exact binary128/decimal113 decoding preserves distinctions below binary64.
The native shifts/masses are checked against each interface's observed routing
and ordered prefix construction. Mode5 reversal and sign conventions are
checked without changing the captured input. CutTools still receives the
actual **DP denominator ABI** while executing QP callbacks. Ninja's native
binary128 and the bridges' ASCII113 records are distinguished. Ninja/NinjaRS
promote a DP-computed squared scale; TIRust squares the widened scale. Squared
masses can be formed in DP and then widened. These are observed adapter facts,
not claims that all reducers receive identical native tensor bases or inputs.

Before comparing coefficients, the parser requires exact plain/observed
public input/output/status/IEEE parity and equality of the common physical,
real Born, repaired DP/MP, model, coupling-update and canonical QP numerator
inputs across the relevant arms. The native encodings and bases are reported
separately; no all-backend raw-CFFI identity is required or claimed. Shared
input mismatches retain their witnesses and suppress coefficient comparison.
For admissible upstream comparisons the report includes absolute differences,
raw relative differences and abs(a-b)/abs(a+b), with explicit zero/opposite
handling. Both directions and all within-direction reducer pairs are retained.
Agreement is not truth by reducer voting.

**Status120 / accuracy-1 remains unchecked**, including when every protocol
gate succeeds. All stdout/stderr is retained; every case-insensitive ##W03
warning is recorded and excludes the worker's coefficients from comparison.
Native refusal, absent nonzero groups, unstable host groups, incomplete traces,
IEEE observation changes and nonfinite outputs also fail the protocol.
`result.passed=true` would mean the protocol, observation parity and shared
input gates completed. It would **not** mean the coefficient discrepancies
passed either numerical tolerance, that MadLoop checked a fixed-mode result,
or that this case/the campaign/performance was accepted. Host-local QP
counterterm and contraction temporaries have no separate exposed ABI and are
explicitly unobserved before conversion to the public DP return. Native QP
results and the returned host-contracted DP group values are retained.

## Capacity, execution and limits

The byte journal has 200,000 records and 128 MiB of payload per worker. A
conservative non-callback allowance from 3 points ×116 groups, 48 helicities,
70 coefficient slots and the fixed public envelopes is 6,562 records; the
remaining space permits at most 96,719 callback pairs. This is a finite
instrumentation limit, not a proven bound on CutTools' internal work. Any
record or byte overflow sets health and fails visibly; no silent truncation,
capacity expansion or native retry is allowed. The observer is not a timing
instrument, and no overhead claim is made before later measurement.

The future build consists of four serial commands: private Fortran observer,
C observer, driver compilation, then one relink. Private diagnostic flags are
explicit in `run.py` and differ from production recipes by design; numerical
objects keep their existing bytes. Total build budget 120 s, address space 4 GiB.
The existing reviewed owned-process supervisor is copied unchanged except its
two literal CPU11→CPU12 affinity strings. SIGINT/TERM/HUP handlers feed its
owned-group cleanup/reap path. CPU12 affinity is nonexclusive; no quiet-window
or final timing claim is made.

The run stage acquires the common nonblocking numerical lock before creating
its one-shot run directory. Busy75 is a scheduling hold. It then performs one
30s target-loader trace check and 16 workers with 180 s caps, one thread and 4 GiB each, under a
shared 3000 s stage budget. It verifies actual mapped ELF closure per worker
against the bound libraries, including the current fixed-256 TIRust DSO
`eb3d67fd…`. It retains source/object/runtime hashes, full output, param.log,
trace, warnings and failure metadata. It stops at the first failed worker;
there is no automatic retry. Compiler/runtime receipts contain selected
nonsecret environment values, not arbitrary inherited secret values.

Future commands, **requiring separate root admission before build or run**:

```sh
source tools/tirust_env.sh
PYTHONDONTWRITEBYTECODE=1 python3 -B docs/benchmarkfolder/experiments/qcd-n3-genuine-qp-diagnostic/source-proposal2/run.py verify
PYTHONDONTWRITEBYTECODE=1 python3 -B docs/benchmarkfolder/experiments/qcd-n3-genuine-qp-diagnostic/source-proposal2/run.py plan
# Build only after its explicit admission:
PYTHONDONTWRITEBYTECODE=1 python3 -B docs/benchmarkfolder/experiments/qcd-n3-genuine-qp-diagnostic/source-proposal2/run.py build
# Run only after resulting build/relocations/link/runtime review and admission:
PYTHONDONTWRITEBYTECODE=1 python3 -B docs/benchmarkfolder/experiments/qcd-n3-genuine-qp-diagnostic/source-proposal2/run.py run
```

Actual wrapped edges must still be demonstrated by the resulting link map,
external relocation/symbol evidence and runtime traversal. Source review does
not qualify those edges. The plan deliberately depends on the named closed
workspace donors; it is not a portable scientific replay capsule or an
instruction to reconstruct missing binaries silently. This proposal is not
part of the frozen report4/report5 sources and does not regenerate a report.
