# Private 512-bit threshold diagnostic

This experiment builds the historical bridge with 512 working bits and a
precision-dependent threshold `2^-(512-120) = 2^-392`. It preserves binary128
input parsing, `set_input_precision(113)`, and the resulting `2^-100`
lightlike-classification tolerance. Both full and poles-only workspace guards
receive the same change. The installed bridge, production card, original
stress stream, and earlier evidence remain unchanged.

The source transformation is exactly the three functional changes and one
comment in `../guard-audit/diagnostic-512-source.patch`. Its original SHA-256
is `1eab09d61652e4dfade900a22f8c63a071b4fecee2e7454e8144f934f17e34e0`;
the diagnostic source SHA-256 is
`59184358f8c2f91b1ac38397e76f8accc6e4ffce8aaa529bdde2acbaee985871`.

The guard is a conditioning heuristic, not an accuracy bound. Besides the
Gram checks, this workspace setter controls scalar and modified Cayley
guards, high-rank pentagon recursion, and six-point coordinate checks. Higher
working precision evaluates the same rounded inputs; it cannot recover
information lost when those inputs were constructed. A successful native
status alone will not qualify the tensor or complete amplitude.

`build.py` reuses the exact historical release dependency artifacts and flags
recorded in the earlier native bridge profile, changing only the bridge
source, output directory, and Rust crate name. It hashes every available
release Rust dependency and the three GMP/MPFR/MPC archives before and after
compilation, retains the compiler identity and command, and verifies the
same twelve public ABI exports. Output is confined to
`.tirust-build/ew-threshold-512/`; its private runtime keeps the historical
SONAME so the unchanged Fortran wrapper can load it explicitly.

Compilation is pinned to CPUs 6–8 with a 1,800-second bound. Concurrent
installer compilation on CPUs 2–5 and unrelated diagnostics on CPU 9 make
elapsed times ineligible for performance comparisons. The build does not
invoke a native tensor evaluation. Any subsequent replay must bind this
artifact, check its actual loader mapping, use a fresh cache namespace with
result caching disabled, and retain both served and rejected results.

From the repository root, once, with no existing private build directory:

```sh
source tools/tirust_env.sh
PYTHONDONTWRITEBYTECODE=1 python docs/benchmarkfolder/experiments/ew-threshold-adjudication/tensor-replay/higher-precision/build.py
```

`preparation.json` records the prepared policy and inputs. `build.json` records
the actual outcome, including failures; the presence of this source file or
recipe is not evidence that a build or numerical replay succeeded.
