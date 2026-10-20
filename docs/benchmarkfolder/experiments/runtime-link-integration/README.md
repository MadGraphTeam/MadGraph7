# Generated-export validation of GNU runtime link order

This experiment checks the final compact and expanded Fortran generators with
NinjaRS ABI 2 and compiler-runtime link metadata version 1. It is separate from
the earlier installer validation, which predates the runtime repair. The
supported environment exercised here is Linux x86-64 with GNU Fortran 15.3.0.

Validation passed: the relocated standalone returned 110, 120, and 218 in its
three requested modes, with all Laurent coefficients agreeing with the retained
CutTools reference. The complete NLO fixed-order executable built, the generated
soft/collinear tests passed, and all 20 pole checks passed at `1e-5`. Both NLO
executables and the standalone import versioned GNU binary128 helpers.
See `runtime-results.json`, `standalone-results.json`, and the retained logs.

`standalone.mg5` exports `u u~ > z [virt=QCD]` with internal dependencies.
The complete output is then moved to `standalone-relocated`, removing its old
path, before the subprocess is compiled. The NinjaRS archive, module, header,
configuration, prefix flags, and native library flags are all regular copied
files. The generated link flags refer to the relocated `lib` directory.
`LD_LIBRARY_PATH` and `LIBRARY_PATH` are removed from both the build and execution
environments.

The generated standalone driver's `stable` command-line argument enables the
documented `ML5_0_FORCE_STABILITY_CHECK(.TRUE.)` call, including initialization.
The driver uses 20 calls, since its original four calls still returned the
initialization code 130 with these filters. That initial attempt is retained
as `initialization-only-*`. Normal DP/QP runs retain the original stability
policy. Both the original and changed driver are retained.
`validate.py --run` checks normal DP (`CTModeRun=1`), forced QP (`4`), and stable
DP (`-1`) modes, expecting return codes 110, 120, and 218 respectively. An
initial invocation used mode 2, which is reversed-order DP; its evidence is
retained as `standalone-reversed-dp-*` and is not counted as a QP test.
The script reads the raw matrix elements
from stdout, avoiding the normalized Laurent coefficients in `result.dat`.
The comparison reference is the retained original CutTools value from the
earlier installer validation; all four coefficients must agree within
`1e-11 * max(1, abs(reference))`.

`nlo.mg5` exports the full `u u~ > z [QCD]` calculation. The NLO command
`compile FO -f` builds the fixed-order executable and runs the generated soft,
collinear, and 20-point pole checks. This is a real NLO compile, not only a
standalone virtual subprocess. The run card selects the bundled `nn23lo1` PDF,
matching the prior installer validation. The default LHAPDF selection first
stopped before compilation because `lhapdf-config` is absent; that log is
retained as `nlo-missing-lhapdf.log`. The external NLO export also needs the
original Ninja shared-library directory at execution; the script sets
`LD_LIBRARY_PATH` to the existing Ninja/Collier directories for NLO only.
The first invocation without that path is retained as
`nlo-missing-legacy-ninja-runtime.log`; it compiled successfully but ran zero
pole checks and is not counted as numerical validation.
Optional StdHEP export compilation failed in
this environment, as in the prior validation; this affects the PY6/Herwig6
shower route, which this fixed-order check does not exercise.

The independent source review found the required prefix in the vendor
Makefile, installer smoke link, standalone exporter, and NLO exporter.
Old installation metadata is rejected with a forced-reinstallation instruction,
and copied dependencies retain the new metadata file. Eight installer tests
passed (`installer-tests.log`). No production sources were changed by this
review.

The compiled executables are inspected with `nm -D` and `ldd` (the NLO build
strips ordinary symbols). Their binary128
arithmetic references must resolve to versioned GNU runtime symbols, rather
than definitions extracted from Rust's static archive. The selected symbol
lists are retained here; complete dynamic symbol lists remain in the ignored work directory and
are hash-bound in the JSON evidence. The broader directed-rounding and
arithmetic probe is retained separately in `../quad-runtime-linkage`.

Reproduce from the repository root, with no concurrent Symbolica MP process:

```sh
source .ninjars-build/dev-env.sh
export MAKEFLAGS="SHELL=$(command -v bash)"
python bin/madgraph --plain docs/benchmarkfolder/experiments/runtime-link-integration/standalone.mg5
python bin/madgraph --plain docs/benchmarkfolder/experiments/runtime-link-integration/nlo.mg5
python docs/benchmarkfolder/experiments/runtime-link-integration/validate.py --build
python docs/benchmarkfolder/experiments/runtime-link-integration/validate.py --run
```

Use a fresh work directory for a new reproduction: the script deliberately
rejects simultaneous original and relocated outputs. The original launch
mistakes (an obsolete CLI filename and an unsupported `set iregi` card entry)
are retained as `*-error.log`; the corrected adjacent cards were the successful
exports. These are experiment invocation errors, not export regressions.

`source-identity.json` records the explicitly frozen generator and runtime
source hashes observed after export. It does not claim that Git identity was
captured during generation. `generated-source-hashes.json` records the actual
exported sources, while `relocation.json` binds the copied dependencies and
standalone executable. Numerical results and NLO executable hashes are in
`runtime-results.json`. `compiled-source-hashes.json` and the
`nlo-after-compile-*` cards retain the sources and inputs after the real NLO
compile, including its automatic zero-width setting for the external Z.
`validate.py --collect` rechecks existing logs and symbols without invoking
numerical evaluation.
