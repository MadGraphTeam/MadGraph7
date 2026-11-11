# Current-source relocated deployment: runtime qualification

Nine fixed-mode numerical workers passed after relocating the actual CLI exports
and removing their private copied installation prefixes. Standalone modes were
CutTools `(1,1)`, NinjaRS `(8,1)` and TIRust `(9,1)`, `(9,2)`, `(9,4)`, `(9,5)`;
NLO virtual modes were CutTools `(1,1)` and TIRust `(9,1)`, `(9,4)`. Tuples denote
`(MLReductionLib, CTMode)`. Every worker agrees with its own export's CutTools
control for Born, finite and both pole coefficients, including raw split-order
coefficients. All comparisons pass both `3e-6` and `1e-8`; the largest symmetric
residual is approximately `4.34e-16`.

The tenth worker requested finite-disabled TIRust through the ordinary NLO
virtual diagnostic entrypoint. It exited with code 1 and the expected
`Poles-only tensors are not complete amplitudes` message, retaining an empty
result. This verifies the incomplete-amplitude guard; it is not Monte Carlo
integration or a poles-only numerical benchmark.

The source is revision `0fcd2cd873d61d134837e1d9e87afb13a47824fd`, exporter
inventory `afb6e0be…`, with installed TIRust DSO `eb3d67fd…`. The adjacent
[build-only packet](../current-source-deployment-qualification-1/README.md)
preserves the actual CLI commands, generated sources, builds, relocation,
compiler/ABI metadata and static NinjaRS archive identity. Its immutable seal is
`5c5ae6bc61d99b4312f6e69296e44672e6907b40351a4a9bf01a1e8ac2bb1b70`.
This runtime packet depends on that adjacent packet; it does not duplicate native
binaries or claim that the omitted artifacts were rehashed during portable replay.

`admission/` preserves the separately reviewed controller and additive
interpreter binding, sealed as `6e1539f8…`. The executable's real glibc 2.40
`PT_INTERP` (`1e08370b…`) was explicitly added alongside the unchanged historical
`ldd` diagnostic provider set. No original source or build receipt was rewritten.
The admission README and its `native_execution_admitted:false` source-index field
describe the preserved prelaunch snapshot; execution was admitted separately.

Each of the ten retained `LD_DEBUG` logs shows 12 initialized providers,
including the correct regular DSO inside its relocated export and the exact
prebound interpreter. There are 13 unique initialized paths across the two
exports. These actual initialization logs add runtime evidence to the earlier
build-only diagnostic `ldd` records. They do not identify static object-level
calls; the static NinjaRS archive is bound by the build receipt.

The generated driver remains unchanged with `NPSPOINTS=4`. Each `result.dat`
retains only the final call, so this evidence observes one final physical point
per worker. Modes 1/2 return unchecked DP status 110 and modes 4/5 unchecked QP
status 120; all scalar and split-order accuracy fields are `-1`. This is
deployment and fixed-mode parity evidence, not automatic stability qualification.
The cards retain full finite evaluation, all poles, caching and the QP integrand.
Both active runtime cards were restored to their exact pre-run hashes.

The whole controller took 8.900 seconds according to Bash's timer, including
initial source/artifact/provider verification, lock wait, workers and final
verification/restoration. Environment setup and the separate source-seal precheck
are excluded. The internal 900-second deadline begins after initial verification;
the observed whole process also completed below that limit. This duration is
diagnostic and provides no performance acceptance claim. All ten process groups
were absent after cleanup, the common numerical lock was probed free, and
protected production files remained unchanged.

Run the pure evidence replay from the repository directory:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/current-source-deployment-runtime-qualification-1/verify.py
```

The verifier checks both packet seals, replays only the three sealed pure
number/result/comparison functions, and cross-checks actual cards, raw results,
worker receipts, initialized-provider logs and cleanup records. It opens no
historical absolute paths, loads no native library and launches no process.
The independent audit also reparses results and recomputes comparisons without
relying on the controller's stored pass flags.
