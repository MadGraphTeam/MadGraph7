# Frozen DP terminal consumer audit

This package validates the **isolated QCD n=1 experiment**. It does not change
production code, compile anything, execute numerical workers, or certify an
arbitrary export. `audit_consumers.py` is standard-library Python. Its public
API is:

```python
audit(baseline_root, candidate_inputs_path, transformed_root=None,
      helper_path=None, helper_sha256=None, build_logs_dir=None,
      trace_admission=False) -> dict
```

An invalid or unrecognized input raises `AuditError` (a `ValueError` subclass).
Unreadable files also fail the call. Success returns `passed: true`, with
`native_execution: false` and `performance_eligible: false`. Helper SHA-256
must match both the supplied value and one of the two reviewed helper
identities embedded in the validator. `expected_terminal_caller(bytes,
trace_admission=False)` returns the permitted caller bytes.

## What is checked

1. The entire `tirust-first-attempt-3/candidate-inputs.json` is pinned to
   `1dbf671783cfb56762b612140693d34c781e0c46cdef88096c9861a944fa41b4`.
   All 111 frozen source/include/build-file hashes must match. Changing that
   manifest to bless a modified source is rejected. Extra source/include/build
   files and unrecognized source aliases are rejected. Materialized copies of
   known symlinks are accepted when their bytes match; preserved source symlinks
   must have the known relative target and remain inside their own export.
2. The three historical compilation logs are checked against the manifest's
   hashes. Their actual command arguments are checked against the independently
   parsed frozen dry-run recipes. There are **49 actual compile roots**:
   17 DHELAS, 13 MODEL, and 19 process units. The 52 dry-run recipes additionally
   list `testprog.f`, `f2py_wrapper.f`, and `check_sa_born_splitOrders.f`; these
   are reported separately, and their source bytes are still pinned.
3. INCLUDE files are expanded physically before fixed-form continuation lines
   are assembled. This matters for the CutTools type fragments in `cts_mpc.h`.
   Current-directory and ordered internal `-I` lookup is preserved, including
   legitimate `../cuts.inc`, `../run.inc`, and `../maxparticles.inc`. Unpinned,
   external, escaping, cyclic, or unrecognized textual includes are rejected.
   The actual closure has **79 logical paths**, **76 resolved physical paths**,
   **30 included files**, **97 distinct dependency edges**, and 169 recorded
   include occurrences. Each result records every occurrence and each logical
   source's resolved internal path.
4. All actual compiled units, including MP code, are scanned. The only five
   permitted DP COMMON holders are the frozen coefficient batch, Born batch,
   loop-CT batch, loop matrix, and MP coefficient routine. Their declaration
   type, dimensions, member order, and alias names are checked explicitly.
   `COMMON/ML5_0_WL/DP_WL,DP_PL` is allowed **only** in
   `mp_compute_loop_coefs.f`, with the exact seven writes to seed indices -1/0.
   The ordinary loop matrix has the corresponding seven `WL` seed writes.
   Included consumers, bare-array escape calls, EQUIVALENCE uses involving the
   wavefunction, unknown aliases, and non-seed uses outside the coefficient
   batch are rejected.
5. The coefficient batch must have precisely the original chain
   `WL0 -> WL1 -> WL2 -> WL3`, followed by the original CREATE call for WL3.
   The DP merger's complete logical body must match the diagonal
   `OUT(J)=OUT(J)+WL(I,J,I)*CONST` implementation. CREATE has only its known
   declaration, argument, and merge-call uses of `LOOP_WF`.
6. A transformed export may differ only in `coef_construction_1.f` and
   `polynomial.f`, and only by the exact expected bytes. The caller gains its
   IEEE import and two logical variables, captures entry INEXACT immediately
   before the final update, and redirects that call with the two extra
   arguments. The polynomial file retains **all original bytes** and appends
   the known helper after name substitution and fixed-form rendering. The
   fixed-form renderer preserves label 900 in columns 1–5. Original CREATE,
   MERGE, QP, intermediate updater and initialization code remain byte-identical.
7. `trace_admission=True` additionally permits exactly the two-line
   `WRITE(*,*) 'DP_TERMINAL_ADMISSION', ...` statement immediately before the
   unchanged CREATE call. Without that explicit option the trace clone is
   rejected. Its result is marked `trace_is_diagnostic_only: true`; this audit
   does not make its timings eligible for comparison.

The numerical guard proof and guard-execution evidence live in the parent
experiment directory. This package checks the identities of the reviewed
helpers and their exact insertion; it does not rerun their numerical tests.
The original and packed helper identities are respectively
`a9644682d62de62dfd3f33c80b51fd916383c443701507cc6fc5c95c81abe95e` and
`0f3af055f42e7fea0334b97fb0765451b9257e5cb0aec5ea48320043fb27a730`.

## Reproduce without native execution

Run from the MadGraph repository directory. `python3` below means a Python
environment with pytest installed for the tests; the audit itself needs only
the standard library.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest -q -p no:cacheprovider \
  docs/benchmarkfolder/experiments/dp-terminal/consumer-audit/test_audit_consumers.py

python3 -B docs/benchmarkfolder/experiments/dp-terminal/consumer-audit/audit_consumers.py \
  --baseline-root .benchmark-work/tirust/outputs/candidate/qcd_n1_uux_z \
  --candidate-inputs docs/benchmarkfolder/runs/qcd_n1_uux_z/tirust-first-attempt-3/candidate-inputs.json \
  --transformed-root .tirust-build/dp-terminal/amplitude-elemental \
  --helper docs/benchmarkfolder/experiments/dp-terminal/terminal.f90 \
  --helper-sha256 a9644682d62de62dfd3f33c80b51fd916383c443701507cc6fc5c95c81abe95e
```

Omit the final three arguments for the baseline audit. For packed, select
`amplitude-packed`, `packed-scan/terminal.f90`, and the packed SHA above. For
the diagnostic clone select `amplitude-elemental-trace` and add
`--trace-admission`. `--output` writes a JSON result. By default the log directory
is derived from the frozen candidate-inputs location; `--build-logs-dir` can
select a relocated copy, still subject to the original log hashes.

## Limits and future production generalization

This is deliberately a source-identity-bound contract. It audits the actual
compiled textual inputs and the permitted source change; it does not inspect
native symbol resolution or reconstruct object code. External USE modules
`COLLIER`, `MNINJA`, `MNINJARS`, `MPMODULE`, and `MTIRUST` are a declared boundary:
their `.mod` files are not pinned by the historical source record. Runtime
library/archive identity, compiler invocation and executable provenance remain
separate obligations of the isolated build protocol.

A production implementation needs a general graph proof plus an independent
consumer/alias proof over every compiled unit and include, generalized shape
and rank reasoning, and compiler/IEEE qualification. It must fail closed on
unknown custom consumers or modules. The known DP_WL alias may be generalized
only by proving its exact layout and seed-only accesses; removing the alias
check globally would invalidate this argument. Frozen source hashes are an
appropriate narrow experiment boundary, not a replacement for that future
generalization.
