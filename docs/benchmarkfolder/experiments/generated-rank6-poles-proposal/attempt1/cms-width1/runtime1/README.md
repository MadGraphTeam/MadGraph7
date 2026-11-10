# Retained positive-width CMS diagnostic

This package preserves one completed private tensor diagnostic: two fresh EW
captures and 15 fixture-105 rows with the exact positive-width complex mass
bound in `frozen/repo/.../cms-width1/fixture.json`. It retains the original
execution record, inputs, outputs, runtime mappings, and all errors. Compressed
files are UTF-8 text, with their original byte counts and SHA-256 hashes recorded
in `copies.json`. No executable or numerical library is included.

Of the 15 new rows, six TIRust DP rows refuse the reduction, six TIRust QP rows
return successfully, and three COLLIER rows return successfully. All 15 retain
IEEE flags. Three input-policy mismatches prevent counting the COLLIER rows as
independent agreement. These results do not qualify physical amplitudes, general CMS
support, or performance. The combined analysis retains 191 open items:
123 IEEE-flag entries, 45 refusals, 18 input-policy mismatches, and five capture
refusals. Its 88 comparisons contain 61 passes and 27 refusals. Successful
protocol completion in `execution.json` does not mean numerical acceptance.

The unchanged CMS analyzer substitutes the 15 new rows only within its
comparison logic. The archived new rows and original EW/QCD control rows remain
separate. The original zero-width attempt and its failures remain unchanged.
`data/controls/` contains exact decompressed copies of the earlier portable
evidence, recompressed deterministically; `copies.json` records that provenance.
The frozen source index records the compile-stage admission state at its
creation; it is historical provenance, not a new execution permission.

From the repository root, the following command checks every package hash,
decompresses and validates every archived text copy, and runs the frozen Python
analyzers in a temporary tree. It must reproduce the original `results.json`
exactly. It requires only Python's standard library and does not compile, load
numerical libraries, execute the controller, or launch a native workload:

```sh
source tools/tirust_env.sh
PYTHONDONTWRITEBYTECODE=1 python3 -B docs/benchmarkfolder/experiments/generated-rank6-poles-proposal/attempt1/cms-width1/runtime1/replay.py
```

`index.json` binds every package file except itself. Absolute host paths in
retained logs are provenance only; offline replay does not access those paths.
The build record is included as text provenance; this package does not rerun or
requalify the original build. Precise timing remains Hua-Sheng's local TODO.
