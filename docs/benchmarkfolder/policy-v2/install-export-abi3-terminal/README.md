# ABI 3 standalone and NLO validation: NLO compile failure

This archived run validates committed source
`c5496ca9b148b16a931214e1d541ead100ed1aad`. The pinned-download CLI installation
and relocated standalone checks passed. The full NLO build failed because its
generated MP coefficient routine omitted the declarations needed by the new
terminal call. The failure is retained; this namespace must not be overwritten
or treated as successful NLO validation.

The installed ABI 3 native/compensated/native DP and genuine binary128-input
smoke passed. The installed archive is byte-identical to the main validated
archive (`1199bc84570bee76a6b162a86fd5c191479236b2c0617fe4bf250d5de0d9df32`).
Both exports were relocated and their private installation hidden. Their
library and interface files are regular copies with matching hashes.
Standalone DP, QP and normal stability returned 110, 120 and 218; Born, finite
and both pole coefficients agreed with the retained CutTools reference within
`1e-11 * max(1, abs(reference))`. GNU binary128 helper linkage was verified.

NLO `mp_coef_construction_1.f:93` calls
`IEEE_GET_FLAG(IEEE_INEXACT,WL_ENTRY_INEXACT)` without importing
`IEEE_EXCEPTIONS` or declaring `WL_ENTRY_INEXACT`. The Fortran compilation
therefore failed. The NLO CLI returned zero after reporting the compile error;
the validator correctly rejected the run because successful pole-check markers
were absent. No NLO pole or soft/collinear success is claimed. The optional
StdHEP compilation warning also remains: PY6/Herwig6 shower support was not
established. See `failure.json`, `failed-generated-source/`,
`nlo-compile.log` and `runtime-validator.log`.

From the repository root, with the shared Symbolica slot reserved:

```sh
source .ninjars-build/dev-env.sh
export CARGO_BUILD_BUILD_DIR="$PWD/.ninjars-build/cargo/build"
export CARGO_TARGET_DIR="$PWD/.ninjars-build/cargo"
export TMPDIR="$PWD/.ninjars-build/tmp"
export SYMBOLICA_LOCK_DIR="$PWD/.ninjars-build/symbolica-locks"
python docs/benchmarkfolder/policy-v2/install-export-abi3-terminal/prepare.py \
  --revision c5496ca9b148b16a931214e1d541ead100ed1aad
python docs/benchmarkfolder/policy-v2/install-export-abi3-terminal/export.py
python docs/benchmarkfolder/policy-v2/install-export-abi3-terminal/runtime_validator.py \
  --build --hide-installation --run
```

The installer used pinned downloads through the actual MG5aMC CLI, checked ABI 3
metadata and requires the compiled native/compensated/native DP and genuine
binary128-input smoke test. It does not use sibling source overrides.

Standalone and NLO exports copy their dependencies internally. The validator
relocates them, hides the private installation, removes external library-path
overrides, and checks standalone DP/QP/stability results, the full NLO build,
20 pole checks, soft/collinear tests and GNU binary128 symbol resolution.
Success requires the numerical markers and output files, rather than only a
zero CLI exit code. Work products stay in
`.benchmark-work/policy-v2/install-export-abi3-terminal`.

The commands above record this attempt. A corrected source revision must use a
fresh report/work namespace. The existing Cargo dependency-download cache was
reused; new compilation products and temporary files used the repository-local
paths recorded in `preflight.json`. No production source was edited by this
validator. The only preparation gap was the missing `nlo-compile.mg5`, supplied
with the exact previously validated `compile FO -f` and `quit` commands.
