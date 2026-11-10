# Preserve the donor runtime with a private SONAME directory

This is a static follow-up to the immutable six-payload run2 closure audit
(`index.json` SHA-256 `1dd2b40762df401685288e3579705a22c6a22d7020970ea90e7afe800dd2b3b6`).
It proposes preserving all fifteen original runtime paths and hashes rather
than adopting run2's three different compiler libraries.

No reordering of the seven existing library directories recovers the original
mixed compiler providers. The desired libquadmath requires `chqq…` before
`xc0…`; the desired libstdc++ and libgcc_s require `xc0…` before `chqq…`.
All 5,040 directory permutations were inspected using filesystem metadata and
hashes; zero satisfy the three original provider pins. The exact candidate
matrix and contradictory ordering pairs are in `resolution-audit.json`.

The proposed successor creates a new private directory with exactly fifteen
SONAME symlinks. Each symlink points directly and absolutely to its original
canonical provider. Keep the manifest outside this directory. At the final
target execve, `LD_LIBRARY_PATH` contains only the private directory; host tools
and Python retain their clean environment. The existing target environment
helper already strips inherited `LD_*` values before setting target variables.

All fifteen original providers and the fresh build2 executable have no
DT_RPATH. Every DT_NEEDED name is a basename supplied by the proposed directory.
LD_LIBRARY_PATH therefore precedes their DT_RUNPATH entries for both direct and
transitive lookups. Kernel PT_INTERP remains the original glibc 2.40 loader;
the loader's SONAME alias points to that same pinned file. No executable,
physics library or original compiler-library bytes need to change.

`validate_shim.py` is a filesystem-only preflight proposal for an already
constructed directory. It requires an exact alias set, unique canonical
providers, absolute direct symlink targets and matching SHA-256 values. It
rejects extra entries, including glibc-hwcaps directories. Its seventeen tests
use temporary non-executable fixture bytes and cover missing, altered,
redirected, relative, indirect and unexpected entries. They do not create an
actual diagnostic shim or invoke any inspected binary.

An actual successor must still run the bounded loader trace and strict worker
map checks against the **unchanged original fifteen canonical paths/hashes**.
Recheck shim metadata and provider hashes before and after those phases. Do not
weaken the existing map equality check, add exclusions or relink build2. Preserve
failed run2 and source2, and use distinct source3/run namespaces. Runtime dlopen
additions remain subject to the strict map check.

The static checks support this mechanism; they do not establish that a future
trace or worker will succeed. No numerical, performance or QP qualification
follows from this proposal. No actual diagnostic shim, loader trace or amplitude
was created or run by this audit.

Run the pure fixture tests with:

```sh
source tools/tirust_env.sh
python3 -B -m unittest discover \
  -s docs/benchmarkfolder/experiments/qcd-n3-genuine-qp-diagnostic/run2-donor-closure-resolution-proposal-1 \
  -p test_shim.py -v
```
