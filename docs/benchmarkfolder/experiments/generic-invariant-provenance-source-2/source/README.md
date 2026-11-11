# Generic invariant provenance — source proposal 2

This private source-only proposal closes the **implicit expression-function
blind spot** in the first proposal. It does not activate an invariant correction,
compile Fortran, load a reducer, or qualify amplitude accuracy/performance.
The parent source index `72c5f272…` and its public capsule `0d2bbdda…` are unchanged.

The new files are an overlay on that exact parent:

- `src/madgraph/loop/invariant_environment_source.py` resolves direct CALLs and
  every function-like expression reference. Declarations, exact INCLUDE bytes,
  module and internal-procedure scopes distinguish arrays/components from
  functions. Typed scalar function declarations need no EXTERNAL declaration.
  Ordinary intrinsics have explicit arities; locally declared arrays, explicit
  EXTERNALs, bound source definitions and imported procedures cannot silently
  inherit an intrinsic exemption. Intrinsic IEEE getters require explicit
  `USE, INTRINSIC`; setters and unresolved/indirect references fail closed.
- `src/madgraph/loop/invariant_window.py` joins the ordered batch source to the
  existing same-kind W-DAG receipts. It analyzes every statement/callee/function
  through the last proved W momentum producer, including intervening amplitude
  calls. It separately requires a complete later reference graph and rejects any
  possible later momentum-field or whole-derived-object assignment. A finite
  formal-write analysis propagates CALL and expression-function actual/dummy
  aliases, including forwarded numeric momentum components; a read-only callee
  is proved from its body, not inferred from its name. Derived I/O,
  pointer aliases, unknown functions and custom procedure effects remain unknown.
- `src/madgraph/loop/invariant_finalize.py` binds the additional generated includes
  and records whole-producer and ordered-window findings; the closed-source checker
  separately records model-function and initialization findings. The
  parent's geometry, unsigned multiplicities, signs, grouping, kinds, freshness
  invalidation, integer bounds and pass-through scientific dispatch are unchanged.
  The emitted `source_environment_supported` remains **literal `.false.`**.

This is a narrow fixed-form reader, not a complete Fortran syntax/type checker.
Unsupported reachable syntax is a failed proof. It supports the explicit
sequence-of-data ALOHA types and scoped read-only `ASSOCIATE(...=>TRANSFER(...))`
value expressions used by the generated safeguards. It rejects procedure values,
indirect dummy calls, unknown/custom operators, aliases, type-bound behavior and
unresolved includes. Unused model pointer routines are not trusted by name: their
own closures remain unknown, while unrelated producer roots are assessed alone.

`reference_complete` means all direct CALL and expression FUNCTION references in
that bounded source family were resolved. `complete` additionally excludes
unqualified runtime effects. Neither field means native or scientific admission.
Formatted WRITE/STOP remains an explicit unqualified Fortran-runtime effect;
it is never described as pure. A post-momentum WRITE cannot establish full-producer
environment purity. A positive window instead requires a safe construction prefix,
no later momentum write, and a future fresh entry/use-token match. It is conditional
on the bound compiler/ABI/runtime contract, not a proof about arbitrary interposition,
callbacks or concurrent mutation.

## Actual source observations

`check_closed.py` authenticates all 269 saved CMS source bindings and all 195 saved
QCDn3 compilation source bindings before analysis. Parent records and exact input
hashes are in `parent-pins.json`, `input-bindings.json` and `closed-results.json`.
No active main/stress/profile outputs are inputs.

Both closed exports have complete CALL/FUNCTION reference graphs for the producer
batches: 141 procedures for Gmu CMS and 165 for loop_sm QCDn3. Whole-producer
source-environment admission remains false because three split-order index functions
contain WRITE branches (two may return), plus one STOP branch. All are strictly
after the final W.P producer:

| Bound export | DP last producer | QP last producer | Prefix procedures |
|---|---|---|---|
| Gmu CMS | W(8), `helas_calls_ampb_1.f:127` | W(8), `mp_helas_calls_ampb_1.f:113` | 8 each |
| loop_sm QCDn3 | W(35), `helas_calls_ampb_1.f:420` | W(35), `mp_helas_calls_ampb_1.f:409` | 19 each |

All four ordered windows pass the source-only prefix and no-later-write checks.
These closed-source window receipts use every actual same-kind W-output call;
they are not a reconstructed UFO graph proof. The fresh generation tests below
join the same window analysis to actual HELAS graph/DAG receipts.

Model initialization is a separate scope. The checked source families have
18/32 (Gmu) and 16/30 (loop_sm) individually supported model functions, including
simple contained comparison functions. Other closures retain pointer-cache paths,
missing initialization includes and the unqualified `IMAGPART` extension as unknown.
No claim is made that COUP/UPDATE_AS_PARAM is fully qualified. The fresh capture
seam occurs immediately before actual W construction; those initialization roots
are absent from the proved prefix. Their earlier execution may not be used as an
implicit environment token or as proof that an initialized model is physically valid.

## Pure checks and limits

- 25 semantic closure tests include undeclared foreign functions, setter wrappers,
  intrinsic/array shadowing, nested subscripts, components, internal/module functions,
  indirect arguments, scoped association escape, derived I/O and unknown syntax.
- 15 window tests include an interleaved WRITE, late function-mediated momentum
  writes, whole-object overwrites, malformed DAG joins and unknown late effects.
- The parent's 17 geometry/kind/overwrite/freshness tests remain passing, unchanged.
- 7 actual MG source-generation checks now emit real MODEL, DHELAS, polynomial and
  process source, with subprocess/system/exec/CDLL operations blocked and all writes
  constrained to this private proposal. They cover grouped/ungrouped Born processes,
  nonidentity external W slots, loop-induced processes and a grouped process prefix.
  Five Born cases pass both DP and QP window checks. Both loop-induced cases remain
  unknown: their generated CREATE_LOOP_COEFS calls supply seven actuals to a six-formal
  header. This is a strict source-contract limitation, not a demonstrated native ABI
  failure or a numerical defect. No call was modified to make the test pass.

Every source-window result still requires actual same-kind production, fresh entry
and use tokens with no skip/reuse/widened-DP path, qualified GNU fixed-form semantics,
compiled module/object/source identity, no unsafe floating-point flags, and native
plain/observed parity. None is inferred from source inspection. Source checks do not
assert IEEE flags stay clear: ordinary arithmetic may raise sticky flags. No generic
native or normal-checked path, new physics point, rank-six reduction, or performance
has been qualified by this proposal. The separate scientific zero-veto consumer and
kernel remain outside this overlay.

From the repository root, `python3 -B .../verify.py --tests` authenticates this packet
and reruns only the 40 pure source tests. `--live` additionally authenticates the
explicit historical source inputs. Reproducing actual MG generation requires the
pinned parent/current repository and model sources; `test_generation.py` writes only
its private `generation-work` scratch directory and does not overwrite sealed receipts.
The recorded generation log and source inventories are historical source-only checks,
not a portable fresh full-model closure.
