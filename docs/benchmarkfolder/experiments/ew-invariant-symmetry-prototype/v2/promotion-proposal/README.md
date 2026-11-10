# Private production promotion proposal

This folder holds proposed copies of `madgraph/iolibs/template_files/loop_optimized/TIRUST_interface.inc` and `tests/unit_tests/loop/test_tirust_fortran.py`, plus `promotion.patch` against their recorded live hashes. The live files have not been changed. No compiler, unit test, reducer, generated process, or campaign was run for this proposal.

The template adds exactly the four qualified v2 blocks: DP/QP integer-format declarations and DP/QP whole-matrix finite checks followed by unequal-pair mirroring. The executable statements are unchanged from `../TIRUST_interface.f` (SHA-256 `177a7973ea9182978097d5256fab0f0c93c1d4c799919203f7855631bee294ad`). Only the comments lose “Private v2” and “this private source”. Removing the four exact blocks recovers the live template byte for byte.

The focused test extracts those declarations and operations from the production template into standalone DP/QP wrappers. An independent integer-word oracle checks finite asymmetric pairs, equal signed zeros and the diagonal, infinity, quiet NaN, and signaling NaN in both triangles and the diagonal. Every nonfinite case includes an unrelated finite asymmetric pair to check that the whole matrix remains untouched. Immediate snapshots check supported IEEE sticky flags, halting modes, and rounding modes; separate cases enable invalid halting. It uses no model or real reduction library and keeps the existing recording-module tests intact. The proposed test reads the production path, so execute it only after the promotion is applied and the production hold is released.

`source-comparison.json` records a source-only render using `FortranPolynomialRoutines(3)` and `FortranWriter`: 35 mapping entries, the qualified 13 registration calls and status checks, process prefix `ML5_0_`, 200 loop groups, one squared-order slot, and rank three. All 316 canonical fixed-form statements match the qualified adapter, including the existing contraction routine. Canonicalization joins continuations and ignores comments, unquoted whitespace, and letter case outside quoted literals. The Python test source was parsed, not executed.

The first render used the initially requested `AmplitudeReduction=True` context. Its five differing statements belonged to the existing contraction branch; the physical qualified export uses `AmplitudeReduction=False`. The first script, render, and mismatch metadata remain as `compare_source-first-attempt.py`, `rendered-AmplitudeReduction-true.f`, and `source-comparison-first-attempt.json`. The corrected context and successful comparison are explicit in the final receipt. The existing recording test intentionally retains its amplitude context.

After the hold is released and the reviewed patch is applied, the focused validation command from the repository root is:

```sh
source tools/tirust_env.sh
python3 -B -m unittest tests.unit_tests.loop.test_tirust_fortran
```

The source-comparison helper writes its outputs exclusively and refuses to overwrite this retained evidence. It is not a test runner. The proposal is source-ready; compiled validation remains pending.
