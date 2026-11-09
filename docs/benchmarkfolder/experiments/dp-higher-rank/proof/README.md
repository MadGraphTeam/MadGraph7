# Private higher-rank consumer and table proof

This isolated copy extends the production DP consumer validator only to private rank 2x1 and 3x1 calls with `(LCUT_SIZE,IN_SIZE,OUT_SIZE)=(4,4,4)`. The existing 26 private rank 1x1 roots remain part of the expected map. No production source is changed by these files.

The transformed fresh QCD n3 source passes with 147 private producers: 26 rank 1x1, 85 rank 2x1, and 36 rank 3x1. The full-source inventory contains 139 Fortran units and 171 distinct source/include paths. `audit-result.json` records SHA-256 pins, the proof source identity, and equality of all pins before and after the audit. The live process may subsequently receive a benchmark driver; rerun the audit after any such source change.

`dp_terminal_proof.py` retains the production build, include, COMMON alias, complete producer/consumer, immediate entry-flag capture, canonical CREATE/MERGE and coefficient-count checks. The private additions require rank-four output storage (70 coefficients), exact literal `NCOEF_R`, the canonical full 70x5 `COMB_COEF_POS` declaration and all 350 entries. Expected table values are independently reconstructed by symmetric monomial multiplication in colexicographic order, including rank-five values in unused final rows. Supported kernels consume only rows 0..14 or 0..34; their mapped output indices are below 70. Table writes, declaration shadows, import renames and argument/address escapes are rejected throughout the expanded closure. The only accepted noninitializing table uses are the existing scalar `NEW_POSITION=COMB_COEF_POS(L,M)` and constant-column variants.

This certifies storage, table identity and consumers. It does not itself certify kernel arithmetic, IEEE exception preservation, compiler behavior, amplitude parity or performance. Those are separate parts of the private experiment. Delayed CREATE consumers are supported when the complete graph proves exclusive terminal consumption; immediate CREATE adjacency is not an eligibility requirement.

Seven source-only test methods pass, including 25 negative mutation cases for table contents/extents, writes/aliases/escapes, missing capture, extra wavefunction consumption and incorrect expected producer maps. Source mutations use in-memory mocks; retained exports are not edited. The original production module and exact extension diff are included. Independent read-only review found no new table/closure blocker (see `independent-review.json`).

Reproduce from the MadGraph repository root with bytecode writing disabled:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B docs/benchmarkfolder/experiments/dp-higher-rank/proof/test_proof.py -v
PYTHONDONTWRITEBYTECODE=1 python3 -B docs/benchmarkfolder/experiments/dp-higher-rank/proof/test_closure.py .tirust-build/dp-higher-rank/process/SubProcesses/P0_uux_zgg -v
PYTHONDONTWRITEBYTECODE=1 python3 -B docs/benchmarkfolder/experiments/dp-higher-rank/proof/audit.py .tirust-build/dp-higher-rank/process/SubProcesses/P0_uux_zgg --expected docs/benchmarkfolder/experiments/dp-higher-rank/proof/expected-producers.json
```
