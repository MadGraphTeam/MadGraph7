# Generic invariant provenance: private source proposal

This is an opt-in exporter/caller implementation against the selected production files at `0fcd2cd873d61d134837e1d9e87afb13a47824fd`. It replaces the fixed CMS route/table dimensions with descriptors derived from each actual generated reduction call and its rendered momentum producers. **It does not change an invariant or a numerical result:** the typed `LOOP_*_PROVENANCE_DISPATCH` seam currently calls the original public `LOOP_*` routine. The scientific certificate and direct-invariant adapter replacement remain separate, pending review and correctness qualification.

No production file, old sealed packet, model, card or Git state was modified. No compiler, linker, loader, tensor reducer or amplitude was executed. Pure generation uses actual MG graphs, optimized call writers, Fortran templates and ALOHA output in a private directory; subprocess/exec/ctypes loads and writes outside this proposal are blocked during those tests. This is direct exporter-method testing, not a completed CLI export/bootstrap or native qualification.

## File and call chain

- `01-metadata-caller.patch` is reproduced by `make_proposal.py` from seven pinned originals in `base/`. Existing numerical templates and legacy LOOP helper bodies stay intact. The private `invariant_provenance` exporter option defaults off. Vectorized output leaves it off.
- `invariant_provenance.py` registers descriptors at the actual grouped reference or temporarily relabeled ungrouped/loop-induced call. It never captures a route in the cached call-key closure. Physical leg numbers and actual W slots are separate. Graph metadata is joined to emitted same-kind external sign/copy and ordered internal sums; signed coefficients, unsigned multiplicities and expanded add/sub counts are retained independently.
- `invariant_source.py` reads a narrow fixed-form grammar. Unknown kinds, aliases, overwritten slots, conditional momentum writes, custom calls/form factors, unresolved parents and malformed route shapes stay unsupported. `invariant_finalize.py` runs after ALOHA/model generation, as the current terminal proof does. It emits immutable per-subprocess tables, a namespaced generic factory and a namespaced read-only environment probe.
- The runtime factory reuses the integer formulas from `aba903450f49e0e18ba3fff40dbf6074e1cd67e9af1c67cd83bbd6fd583c1047`. Changes are dimensions, table include/module names, and explicit external-slot lookup; no error-bound constant or numerical threshold is changed. `factory-formula.patch` isolates this delta. Source `environment_token.c` is unchanged apart from the generated symbol name.
- `invariant_render.py` inserts automatic local state before actual producers, passes it through reduction batches and returns it from an internal MP producer with a legacy public wrapper. The private LOOP caller obtains exact complex W/MP_W words with integer TRANSFER, checks route/WE/pairing/environment/kind, builds chord bounds, and passes the whole contract to the separate dispatch seam. No route is inferred from saved COMMON IDs.
- Standalone/OLP makefile additions include generated factory/probe objects and module ordering. These recipes are source proposals only; they have not been compiled. MadEvent-specific makefile integration remains a separate gap even though process-group prefixes and loop-induced caller source are exercised.

## Freshness and source admission

Initial/rotated entry clears both states. Every accepted helicity clears its state before a possible `SKIP_LOOPNUM_COEFS_CONSTRUCTION` jump. DP captures immediately before the Born/UVCT/coefficient producer sequence, including loop-induced output where a Born amplitude is absent. MP captures only inside the actual `COMPUTE_INTEGRAND_IN_QP` branch. Widened-DP MP remains unknown. Calling MP invalidates earlier DP state because MP may narrow W back into DP storage.

A state can serve all split-order batches of one fresh pass; it is consumed afterward. Later mode reuse/`MP_DONE`, unproved skipped producers and filtered split-order construction fall back. No late capture can certify old W. Full checked-rescue behavior and storage reuse still need native tests; fixed-mode private CMS success is not inherited.

The data retain the actual nonzero closure shift and exact N>2 reversal in modes2/5. Complete normalized corner/closure/prefix snippets and exclusive assignment checks join the current TIRust value path. Other adapter prefix implementations require their own join. A symbolic zero/conservation-equivalent or protected external shell is protected. Unknown external-shell metadata is unknown. Only certified composite structure is marked composite. None of these classifications replace supplied momenta or assume exact numerical conservation.

`source_environment_supported` is an unconditionally false gate in this metadata-only proposal. Direct CALL-graph completeness is diagnostic data, not scientific admission. Implicit function calls in expressions are not yet covered; even a complete CALL graph cannot enable the gate. A future separately reviewed complete source contract is required. A missing model source or unresolved/indirect generated CALL path additionally makes that diagnostic incomplete. The finalizer scans generated source for environment setters and rejects custom foreign-source families. The source graph is not a generic Fortran theorem prover: unrecognized indirect/function/plugin paths need additional source review before scientific activation. Compiler reassociation flags, target ABI, actual entry/use tokens and IEEE noninterference require native qualification. A static token or a clean exit token alone never certifies an earlier producer.

The pure-generation fixtures intentionally do not convert/build the model. Their geometry descriptors are admitted, while their **environment-path gate remains false** both because scientific admission is disabled and because model source is absent. Thus these tests cannot silently authorize the scientific correction. This gap is visible in every generation receipt.

## Validation and limits

`test_protocol.py` exercises N=1..6, E=3..6, all four modes, closure/cancellation multiplicities, table ordering, protected/unknown classification, custom/mixed/conditional producers, malformed routes, state placement, and fail-closed environment paths.

`test_generation.py` exercises actual `loop_sm` graphs/output for:

- `u u~ > z`, grouped and ungrouped;
- `u u~ > d d~`, 7 grouped reference calls and 11 ungrouped calls;
- `g g > h`, four loop-induced calls, including process-group prefix `ML5_0_2_3_`;
- a consistent actual HELAS graph permutation exchanging W slots1 and3 while keeping physical external leg numbers unchanged.

The source-only template context explicitly enables DP/QP TIRust branches; this is not a statement that a library was loaded or discovered. The table/shape tests cover E=5/6 synthetically; actual 2→3/4 full export and generic native integration are unqualified. No performance, finite-amplitude agreement, ordinary checked acceptance or source-independent FP-environment proof is claimed.

Reproduce from the repository root, with all output confined to this private proposal:

```sh
source tools/tirust_env.sh
python3 -B .tirust-build/generic-invariant-provenance-proposal-1/make_proposal.py
python3 -B .tirust-build/generic-invariant-provenance-proposal-1/test_protocol.py -v
python3 -B .tirust-build/generic-invariant-provenance-proposal-1/test_generation.py -v
```

The source index excludes `generation-work/`, temporary directories and earlier development logs. It binds the final receipts and authored/generated patch sources. Referenced generated hashes describe local test outputs, not a standalone portable full-model capsule. Re-running generation is the reproduction route.
