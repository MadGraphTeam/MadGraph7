# Closed V2 private diagnostic build timeout

The separately admitted V2 build terminated with tool exit 1 after
885.038318 seconds. Its Cargo command exceeded the command deadline and
the frozen controller terminated its owned process group. No Cargo terminal
return code was recorded; this packet does not invent one. The log contains
no compiler error before the timeout and reaches the private bridge's
compilation stage. This is a build failure, not a tensor or amplitude result.

The private diagnostic DSO, C object and replay executable are absent.
An exhaustive static inventory found only the private crate's `.d` file
under its name. The required private getter's symbol-name bytes are absent
from every ET_DYN file in the two closed Cargo output trees. The other
ET_DYN files are compiler build helpers or proc macros. No private library
was loaded, no input was replayed, and no diagnostic numerical result exists.

The source proposal remains sealed as
`9c962ee3f7117eba4399d5db3253533ecf92dbcf138964b18b446c363f4ca1d4`.
V2 changed only the exclusive destination and the per-child address-space
limit from4GiB to16GiB; sources, lock, tools, jobs2, codegen-units1, release
flags and900-second deadline were unchanged. The failed first attempt and
its original resource errors remain preserved separately. This packet
authorizes no retry or resource-policy change.

The complete log, actual terminal receipt,1161 partial compiler-output
hashes, and complete staged-source/Cargo-source-cache before/after inventories
are retained. The source/cache inventories agree exactly, and original
source/tool/runtime pins still matched at the post-terminal observation.
Partial artifacts are not qualified libraries or numerical evidence.

`gmp-native-cache-observation.json` records a separate provenance limit.
The pinned gmp-mpfr-sys build script can reuse compiled C libraries under
its system cache. The admitted controller isolated the Cargo target/build
directories but inherited the sourced XDG cache selection. The retained
native-cache paths, output-library hashes and build-script output were
observed after terminal; they are not pre-build pins or proof of a cache hit.
Only fresh Cargo target/build directories are established. No claim that all
native dependencies were rebuilt, or that a complete compiler environment
was reconstructed, follows from this experiment.

Run the portable failure check from the required checkout:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/tirust-first-failure-capture/diagnostics-private-build-failed-2/verify.py
```

The verifier uses retained bytes only. It does not compile, load a library,
or execute tensor calculations. Optional inner-fallback diagnostics remain
open; this packet establishes no accuracy or runtime acceptance.
