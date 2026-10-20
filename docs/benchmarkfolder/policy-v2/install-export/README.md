# CLI installation and internal-export attempt at `fd72f8914`

This retained attempt passed a real `install ninjars` command with pinned
downloads and no local source overrides, then generated standalone and full
NLO `u u~ > z` processes through actual CLI cards. The installation includes the
Fortran interface source, compiled module, static library, header and native
linking metadata. Its Rust archive matches the production archive byte for byte.

Both outputs use internal dependencies and were relocated. Before execution,
the private NinjaRS installation was renamed away and `LD_LIBRARY_PATH` and
`LIBRARY_PATH` were removed from the execution environment. The standalone
output passed forced DP (110), forced QP (120), and ordinary checked DP (218);
Born and all three Laurent coefficients agree with the retained CutTools
reference at `1e-11 * max(1, abs(reference))`. This is an installation smoke
test, not the eight-process benchmark gate.

**The NLO compilation failed.** Its `-fno-automatic` flag implicitly gives a
local `WORDS(2)` array the `SAVE` attribute inside an elemental, hence pure,
coefficient-classification function. GNU Fortran rejects this. The numerical
validator correctly rejected the CLI run even though the command interpreter
returned exit code zero. No NLO pole-cancellation or runtime success is claimed.
The raw compile log, generated classifier, makefile and successful preliminary
soft/collinear test are retained; `failure.json` records the outcome.

From the repository root in the pinned development environment, this attempt
is reproduced by:

```sh
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
python docs/benchmarkfolder/policy-v2/install-export/prepare.py
python docs/benchmarkfolder/policy-v2/install-export/export.py
python docs/benchmarkfolder/policy-v2/install-export/runtime_validator.py --build
python docs/benchmarkfolder/policy-v2/install-export/runtime_validator.py --hide-installation
python docs/benchmarkfolder/policy-v2/install-export/runtime_validator.py --run
```

Preparation requires a fresh work namespace. The original run hid the
installation using the same operations before `--run`; the named
`--hide-installation` entry point was added while that run was in progress.
The numerical execution and validation functions were unchanged. Installation
and forced-QP execution require exclusive use of the shared Symbolica process
slot. The complete compiler failure is preserved rather than overwritten by a
subsequent corrected export.
