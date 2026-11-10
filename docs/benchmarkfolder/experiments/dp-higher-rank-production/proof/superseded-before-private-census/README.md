# Production higher-rank source proof

The production consumer validator now admits private rank 2x1 and 3x1 calls with all three wavefunction sizes equal to four. Existing rank 1x1 validation follows the same path as before: the new coefficient-product table checks run only when higher-rank private calls are present in the expected producer map.

The additional gate accepts process polynomial ranks 3, 4, 5 or 6, exact output storage of 35, 70, 126 or 210 coefficients respectively, and five vertex coefficients. Every private output rank must fit the process rank. The proof independently reconstructs the canonical symmetric monomial multiplication table, checks its exact declaration and every literal initialized entry, and bounds every entry reachable by a supported higher-rank helper. Unused final rows legitimately contain products above the output storage bound; these rows are checked for canonical identity but remain unreachable by the supported helpers.

All expanded source units and includes are scanned for table writes, declaration shadows, renamed imports and address/argument escapes. Table-reading scopes must import the certified polynomial module directly; their only additional accepted imports are intrinsic IEEE modules. Existing contained IEEE helper functions and local TRANSFER bit views are accepted, while aliases of the coefficient table remain rejected. Existing build, include, COMMON layout/alias, coefficient-count, producer/consumer, immediate flag-capture and canonical CREATE/MERGE checks remain in effect.

`test-results.json` records 43 passing test methods and 144 passing subtests, with exact production source and test-file hashes. All previous 1x1 tests remain. New tests cover ranks 3 through 6, rank/storage/vertex restrictions, literal map dimensions and values including unused rows, missing/duplicate map rows, table writes and escapes, unknown reader imports, internal bit views, missing capture, producer/consumer rank mismatches and delayed terminal CREATE consumption. These tests execute Python source validation only.

`retained-process-proof.json` is a source-only smoke check using the existing isolated QCD n3 experiment clone: 147 private roots, 139 Fortran units, and 179 source/include/build paths, plus repository proof/template hashes. It proves input hashes remained unchanged during that audit. It is not fresh production export, compiled-amplitude or timing qualification; those records belong to the surrounding production experiment. The expected map retains the 26 earlier rank1x1 roots and adds 85 rank2x1 and 36 rank3x1 roots.

The Python files named `dp_terminal_proof.py` and `test_dp_terminal_wavefunctions.py` here are frozen source snapshots. `production-base.py` and `production-extension.diff` retain the exact original validator and change. `audit.py` imports the live repository module, checks a supplied expected map, and can replay the full closure proof without modifying the process:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest -q tests/unit_tests/loop/test_dp_terminal_wavefunctions.py
PYTHONDONTWRITEBYTECODE=1 python3 -B docs/benchmarkfolder/experiments/dp-higher-rank-production/proof/audit.py .tirust-build/dp-higher-rank/process/SubProcesses/P0_uux_zgg
```

Kernel operation order, compiler behavior, IEEE exception preservation, numerical amplitude parity and performance are separate qualification obligations. This proof does not infer those properties from terminal graph shape.
