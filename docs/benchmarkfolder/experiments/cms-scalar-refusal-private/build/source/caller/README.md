# Private checked CutTools caller prototype

This is an additive **source-only prototype**, not a production patch or a numerical qualification. No compiler, linker, mock executable, AVH analytic routine, or real reducer was run while preparing it. The original CutTools, AVH, MadLoop entrypoints, production callers, historical numerical evidence, and reference repositories are unchanged.

The original `cts_loopfunctions.f90` calls scalar functions immediately before publishing their outputs into saved scalar arrays. `cts_cutroutines.f90` then enters coefficient construction and amplitude contraction without a scalar status. The prototype adds an explicit status boundary at these locations so that a scalar refusal cannot become a zero-valued successful reduction.

## Additive API and propagation

- `cts_checked_loopfunctions` extracts the actual DP and QP `getA/getB/getC/getD` bodies. It retains their momentum differences, contractions, masses, array indexing, Laurent order and successful assignments. Only `scaloop=2` is supported. Calls use the classifier's exact named complex-mass A0, B11, C0 and D0 checked procedures; original `olo` remains untouched. A status is accepted only if `status=CMS_OK`, `result_valid` and `native_called` are all true. Every other outcome returns before publishing that scalar's output. Contradictory nominal success with invalid output becomes `CMS_BAD_INPUT`.
- `cts_checked_reduction` exposes the **module procedure** `ctsxcut_checked`, with the legacy argument list plus a final `type(cts_checked_report)` argument. Callers must use its explicit module interface. It contains the complete real CutTools reduction flow. All eight original amplitude-selection, DP/QP amplitude-construction and stability-test internal routines remain textually unchanged. The four scalar calls gain checked status and scratch-validity gates before coefficient construction.
- `cts_checked_report` contains terminal status, actual result precision/direction, result validity, internal rescue flag, scalar refusal count, the first refused `cms_scalar_report`, and the last scalar report. Reports reset on every call. The shared classifier event payload is copied as a whole, including lossless operands, source indices, native metric/threshold/scale and classification evidence; the caller adds direction (`+1` normal, `−1` inverse) and subtopology index.
- Private `CTLOOP_CHECKED` templates are supplied for both MadLoop output families. The optimized private `LOOP_*_CMSCHECKED` helper routes CT calls to the checked adapter and returns on refusal. These additive names require explicit private selection; no exporter default or generated production call is changed.

CutTools' actual CT interface converts mass squares into complex values before reduction, even when the original mass is real. Consequently the integrated checked scalar calls use complex-mass overloads for both zero-imaginary and complex inputs. The standalone real overloads are covered in the separate classifier package; there is no invented real-overload CT branch here. The QP path preserves original CutTools semantics, including promotion of its DP momentum/mass inputs; this prototype does not claim new model-derived QP mass transport.

## Refusal, rescue and lifetime

| Path | Scalar refusal behavior |
|---|---|
| modes 1, 2, 3 | Stop before coefficients for the refused direction, mark unstable, invalidate output and clean per-call arrays. No internal QP attempt is introduced. |
| mode 0, DP refusal | Skip the unavailable DP comparison and take CutTools' internal QP rescue. A refused normal DP direction does not run an inverse DP direction whose result cannot complete the DP test. A refused inverse DP direction retains only diagnostic history from the valid normal attempt. Both required QP directions and the original QP stability test must finish before result validity is set. |
| modes 4, 5, 6, or mode 0 QP refusal | Stop before the refused QP direction's coefficients, preserve refusal history, return unstable invalid output, and clean per-call arrays. |
| successful scalar calls but failed original numerator/amplitude stability test | Preserve the original unstable numerical result and `STABLE=.FALSE.`. Scalar status remains successful; the existing numerical stability path is not bypassed. |

All amplitude outputs start as quiet NaN markers. Terminal scalar refusal overwrites any earlier partial DP/QP amplitude with those invalid markers, sets `result_valid=.FALSE.`, calls the existing unstable tagging logic once, invalidates DP/QP scalar scratch, and reaches the original `bbn1..bbn4` cleanup. It does not publish zero as a successful amplitude. Invalid mode, unsupported scalar library, or invalid propagator count returns an explicit unsupported status before allocation.

The checked MadLoop adapter sets `STABLE=.FALSE.`, sets the **active MadLoop pass's** existing reduction-FPE latch, and returns before normalization. The latch is a hard-refusal signal here, not a claim that an IEEE exception actually occurred. For `CTMODE=0`, even a terminal internal QP refusal belongs to the active host DP pass; its actual internal QP precision remains recorded in the typed report. Existing `loop_matrix_standalone.inc` turns that latch into rejected accuracy and advances the rescue chain, then resets both latches when moving to a new reduction tool. A successful internal rescue does not set the latch. The adapter never clears a pre-existing host latch.

There is no new success cache. The original CT saved scalar arrays are scratch storage, and CT is outside MadLoop's tensor integral cache insertion path. Earlier accepted scalar entries can remain in scratch after a later refusal, but neither checked coefficient construction nor amplitude contraction can consume that incomplete batch: the per-direction validity flag stays false until the entire batch has succeeded. Entry and terminal-failure resets invalidate both precisions; each direction starts invalid and recomputes all required scalars. No failed scalar output is assigned, no failed batch is marked reusable, and the typed status gates every private CT result consumer. CutTools remains process-local and non-reentrant; this does not add thread safety.

## Mock and compilation proposal

`fixture/` uses the actual extracted checked scalar gathering and complete checked reduction body. Only scalar analytic values, coefficient construction and their dependencies are mocked. One `USE` statement and two integer amplitude-counter calls instrument the retained reduction body; coefficient counters belong to the mock provider. The private optimized adapter is rendered through the repository's real `FortranWriter` and gains one integer normalization counter. The exact shared classifier report module is copied and hashed. The mock does not link the actual AVH analytic object or any reducer archive.

The proposed single-process mock has 91 logical cases, each instantiated with real-as-complex and genuinely complex mock mass squares, for **182 primary cases total**. They cover every mode, both directions, A/B11/C/D, nonfirst subtopologies where available, internal scalar rescue, original numerator-driven rescue, all refusal statuses through nonfinite-result status 8, and a contradictory status-zero invalid result. Each primary case passes directly through the complete reduction body, through the rendered optimized adapter, and then through a successful direct call in the same process to check reset/cleanup: 182 calls of each kind, or 546 reduction calls total. Assertions require exact coefficient/amplitude/scalar call counts, refused output invalidity, preserved first-refusal evidence, no normalization on refusal, correct active-pass latch, and deallocated per-call arrays. Mock scalar/coefficient numbers are synthetic and do not establish physical accuracy.

`compile-plan.json` contains exactly 11 sequential commands:

1. Seven mock source compilations and one mock-only link.
2. Three compile-only checks of the actual checked protocol, scalar gathering and reduction against the classifier's updated **real AVH** modules and a pinned copy of the real CutTools module/header closure.

Mock module lookup uses only `caller/fixture` and `caller/fixture/build`. Actual-interface lookup uses `caller/actual-build`, `classifier/actual-build`, `classifier/observer-build`, then `classifier/includes`; it never searches either mock build directory. Actual compilation depends on the classifier's observer and real-AVH compilation jobs. Actual objects are never linked or executed in this proposal. The plan lists all dependency hashes, expected objects and module outputs. The parent owns admission and the aggregate CPU 8, single-job, 180-second build bound; no retry is proposed. Mock runtime has a separate 10-second bound and remains unexecuted pending separate authorization.

`validate_fixture.py` strictly compares every expected journal row and rejects missing, changed, reordered or extra output. Its optional event report check additionally requires a unique successful process event and the captured journal hash. The parent remains responsible for process timeouts, cleanup, executable provenance and launch permission.

Ten Python/source tests check preservation of the original numerical bodies, checked publication and consumer ordering, mode-0 transitions, exact hashes, formatter output, pure mock link composition, and strict parsing of explicitly synthetic journal strings. These tests passed; they do not establish compiler or native success. No synthetic journal is written into the future runtime directory.

The full MadLoop host rescue chain, real AVH analytic execution, accepted-input numerical equality, real CMS accuracy, actual allocator/error behavior and performance remain unqualified. In particular, a private adapter refusal latch prepared in source is not an executed full-host rescue result.

Source-only reproduction from the repository root:

```text
/nix/store/463ik27ppxhpbvw7c50s7qcfc7mgfa5x-python3-3.12.12-env/bin/python3.12 -B .tirust-build/cms-scalar-refusal-prototype/caller/prepare.py
/nix/store/463ik27ppxhpbvw7c50s7qcfc7mgfa5x-python3-3.12.12-env/bin/python3.12 -B .tirust-build/cms-scalar-refusal-prototype/caller/prepare_fixture.py
/nix/store/463ik27ppxhpbvw7c50s7qcfc7mgfa5x-python3-3.12.12-env/bin/python3.12 -B .tirust-build/cms-scalar-refusal-prototype/caller/test_source.py
```
