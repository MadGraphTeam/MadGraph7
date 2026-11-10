# Guarded caller boundary fixture

This package prepares a bounded source qualification of the caller-level `CTMODE.LT.4` guard. It has **not been compiled or executed**. The Python tests use synthetic journal strings and source inspection; they are not native numerical results.

The generator imports the exact legacy writer from frozen `attempt1/base` and the composed guarded writer and format-corrected exporter from `combined-qualification/src`. It calls their real `get_loop_CT_calls` and `split_HELASCALLS` implementations on actual HELAS objects and renders with the repository's `FortranWriter`. The six generated caller families use the real split template, CT COMMON (`REAL*8 LSCALE`, default `INTEGER CTMODE` in that order), filter and split-order controls, wavefunction COMMONs, and actual ALOHA derived type. Each guarded block is kept in one split item. The fixed three helper families are `LOOP_2`, `LOOP_3`, and combined-denominator `LOOP_2_3`; paired formals preserve the actual interleaved `M1,MP_M1,M2,MP_M2,...` ABI.

The synthetic mass symbols exercise external real (`MDL_REAL`), a real derived expression (`MDL_DERIVED`), and a genuine complex symbol (`CMASS_MDL_CMS`). The derived expression is evaluated independently in binary64 and binary128; the parser computes both expected bit patterns using exact rational round-to-nearest-even arithmetic. This is a controlled model-symbol transport fixture, not a complete physical UFO/CMS export or amplitude test. The recording helper stubs do not link any reducer, Rust library, model archive, or license-sensitive component.

The grouped route retains IDs `4,8,12`; the ungrouped and loop-induced routes retain `7,9,11,13`. All use rank 2 and split order 1, with the actual generated mother order and pairing. The duplicate ungrouped group deliberately verifies that grouping changes neither the surviving argument order nor the generated IDs.

The complete `commands.json` matrix contains 218 separate processes:

| Mode | MP storage | Arms | Trap modes | Routes | Cases |
|---|---|---|---|---|---:|
| −1, 0, 1, 2, 3 | finite poison, qNaN, sNaN | legacy and guarded | invalid trap off/on | grouped, ungrouped, loop-induced | 180 |
| 4, 5, 6 | valid real/derived/complex | legacy and guarded | invalid trap off/on | same three routes | 36 |
| positive invalid control | finite poison, explicit runtime 0/0 | control | off/on | grouped | 2 |

“Missing MP initialization” is represented by deliberately initialized poison storage. Literal uninitialized-memory behavior is undefined and is not claimed as tested. The qNaN and sNaN values have distinct integer payloads for each real and complex component, including sign bits; finite poison includes a complex negative zero. A binary128 and complex layout self-check rejects unsupported layouts. Poison setup and observation use `TRANSFER` to integer words; they do not format, classify, compare, or convert poisoned floating values. Writable model storage and all COMMON control operands are initialized explicitly.

The invalid trap is enabled only after poison setup, model snapshots, clearing flags, and flushing `PRECALL`. Its positive control must terminate with SIGFPE (`expected_exit=-8` on the pinned Linux environment); the nonhalting control must complete with only invalid set. This positive control uses binary64 0/0 and does not independently establish binary128 signaling sensitivity. Every normal case must complete with exit 0, no caller IEEE flags, exact expected argument words and helper order, and zero observer flag changes. The five flags are named in the journal in this explicit order: invalid, divide-by-zero, overflow, underflow, inexact. Valid MP setup may set inexact; the parser compares that observed setup flag between the matched arms. Poison setup may set none.

Each observer saves/restores the IEEE status and separately detects observer-induced flag changes. The legacy stub never accesses MP storage. It records the finite DP arguments and produces three synthetic DP outputs (first mass, last mass, rank+i·group) and a synthetic zero status. The paired stub additionally records actual MP arguments. These outputs/statuses check caller and ABI transport only: they do not qualify tensor construction, squaring, reducer statuses, stability machinery, amplitude values, performance, or physical CMS accuracy. The fixture does not call `UseQPIntegrand` logic; that belongs to the separately prepared real amplitude lane.

`validate.py` requires every enumerated event, its exact command/affinity/cwd and expected exit, no timeout/error or input mutation, verified output closure, and the retained journal hash. It parses the journal strictly in order: missing, extra, reordered, changed-bit, changed-flag, changed-status, and incomplete rows fail. Only the explicit SIGFPE control is allowed to end after `PRECALL`. It checks all MP records and all DP outputs without dropping components, then compares 90 DP arm pairs exactly and 18 QP transport pairs. A failure raises and exits nonzero, so the parent controller can gate the real EW stage. The parser does not launch anything.

`commands.json` supplies 11 compilation commands and one link command, exact source hashes, object dependencies, all cases, and the validation command. Each case declares scientific `caller.hex` and controller-owned `stdout.log` separately; both are required in the final output closure. `module_outputs` lists the three anticipated compiler-generated module files. The parent owns admission, CPU 8 affinity, strict process-group cleanup, build cap (180 s shared with EW), 3 s per-process caps, and 60 s aggregate fixture runtime; no retry is authorized. These caps are bounds, not measured runtimes. The proposal uses pinned GNU Fortran 15.2, `-O0`, `-fno-fast-math`, and `-fsignaling-nans` explicitly. Compiler acceptance and native behavior of these exact sources remain open.

Pure Python/source reproduction from the repository root:

```text
/nix/store/463ik27ppxhpbvw7c50s7qcfc7mgfa5x-python3-3.12.12-env/bin/python3.12 -B .tirust-build/qp-mass-general-implementation/combined-qualification/fixture/prepare.py
/nix/store/463ik27ppxhpbvw7c50s7qcfc7mgfa5x-python3-3.12.12-env/bin/python3.12 -B .tirust-build/qp-mass-general-implementation/combined-qualification/fixture/test_source.py
/nix/store/463ik27ppxhpbvw7c50s7qcfc7mgfa5x-python3-3.12.12-env/bin/python3.12 -B .tirust-build/qp-mass-general-implementation/combined-qualification/fixture/freeze.py --check
```

The first command checks deterministic reproduction of already generated sources and never launches a compiler. The second runs 11 source/parser tests, with subprocess launch blocked. Its complete-gate success test uses explicitly synthetic journals and events held only in Python memory; it writes no future native journals. Runtime journals and build artifacts are absent and excluded from the source index. Independent read-only inspection confirmed the 18 generated routines' COMMON layouts after fixing the preparation mapping and found no remaining source blocker for this scope. The sole current qualification is successful Python/source validation. No production or frozen parent source was edited.
