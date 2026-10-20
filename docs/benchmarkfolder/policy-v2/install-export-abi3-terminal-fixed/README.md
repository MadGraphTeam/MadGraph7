# Corrected ABI3 installation and export validation

All required checks passed at committed source
`b44d20e725f1ec68589df20284e4ce9116e9e2ab`. See `validation.json` for the summary
and the adjacent logs and manifests for complete evidence.

- Pinned-download CLI installation passed the native/compensated/native DP and
  genuine binary128-input MP smoke. Its archive matches the main ABI3 archive.
- Relocated internal standalone DP/QP/stability calls returned 110/120/218 and
  matched all reference coefficients with the private installation hidden.
- Full NLO compilation, all 20 pole checks and soft/collinear checks passed.
  Standalone and NLO binaries resolve GNU's versioned binary128 helpers.
- All 16 focused Fortran, installer and return-code/statistics tests passed.
  The complete bridge source matches the previously sealed 16-test Rust proof.
- An export with NinjaRS disabled compiled and ran with bundled CutTools.
  It has no NinjaRS module, archive, link dependency or bridge calls. Exclusive
  backend 8 is rejected with an actionable message and no numerical stdout.
- Actual ABI3 installation from both local source overrides passed its smoke;
  before/after manifests match for all 3,193 copied-domain source files.

The optional StdHEP build warning remains; this run does not establish
PY6/Herwig6 shower support. It makes no performance acceptance claim.

`prepare.py`, `export.py`, `runtime_validator.py`, and `nlo-compile.mg5` come from
the sealed failed `install-export-abi3-terminal` run. Only the work/report
namespace changed. `prepared-from.json` records their original and adapted
hashes. No failed output or result was copied into this namespace.

The archived installation and main runtime commands, from the repository root:

```bash
source .ninjars-build/dev-env.sh
export CARGO_BUILD_BUILD_DIR="$PWD/.ninjars-build/cargo/build"
export CARGO_TARGET_DIR="$PWD/.ninjars-build/cargo"
export TMPDIR="$PWD/.ninjars-build/tmp"
export SYMBOLICA_LOCK_DIR="$PWD/.ninjars-build/symbolica-locks"
export MAKEFLAGS="SHELL=$(command -v bash)"
python docs/benchmarkfolder/policy-v2/install-export-abi3-terminal-fixed/prepare.py \
  --revision b44d20e725f1ec68589df20284e4ce9116e9e2ab
python docs/benchmarkfolder/policy-v2/install-export-abi3-terminal-fixed/export.py
python docs/benchmarkfolder/policy-v2/install-export-abi3-terminal-fixed/runtime_validator.py \
  --build --hide-installation --run
```

The existing Cargo dependency-download cache was reused; new build products
and temporary files stayed in the repository. These namespaces refuse to
overwrite prior work. Preserve this run and choose fresh paths for reproduction.

The focused commands were:

```bash
python docs/benchmarkfolder/policy-v2/install-export-abi3-terminal-fixed/focused_validator.py units
python docs/benchmarkfolder/policy-v2/install-export-abi3-terminal-fixed/focused_validator.py absent
python docs/benchmarkfolder/policy-v2/install-export-abi3-terminal-fixed/focused_validator.py absent-resume
python docs/benchmarkfolder/policy-v2/install-export-abi3-terminal-fixed/focused_validator.py local-source
```

The first absent-library runtime test incorrectly selected optional Ninja 6,
which internal-dependency exports intentionally omit. The export/build and
absence assertions passed, but the runtime correctly rejected unavailable backend 6.
That setup failure and the earlier validator are retained under
`absent-failed-ninja6/`. The corrected `absent-resume` uses bundled CutTools 1
and checks exclusive backend 8 rejection. The current `absent` action uses that same
corrected policy for a fresh run. No production source changed.

`focused-controls.md` preserves the earlier audit and command plan with the
completed outcome noted. The main installation remains root-owned and unchanged.
