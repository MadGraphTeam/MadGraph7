# Corrected ABI3 installation and export validation preparation

Prepared only. Do not execute until the root agent supplies the committed NLO
generator fix and releases the shared Symbolica slot.

`prepare.py`, `export.py`, `runtime_validator.py`, and `nlo-compile.mg5` come from
the sealed failed `install-export-abi3-terminal` run. Only the work/report
namespace changed. `prepared-from.json` records their original and adapted
hashes. No failed output or result was copied into this namespace.

From the repository root, after the source revision is supplied:

```bash
source .ninjars-build/dev-env.sh
export CARGO_BUILD_BUILD_DIR="$PWD/.ninjars-build/cargo/build"
export CARGO_TARGET_DIR="$PWD/.ninjars-build/cargo"
export TMPDIR="$PWD/.ninjars-build/tmp"
export SYMBOLICA_LOCK_DIR="$PWD/.ninjars-build/symbolica-locks"
export MAKEFLAGS="SHELL=$(command -v bash)"
python docs/benchmarkfolder/policy-v2/install-export-abi3-terminal-fixed/prepare.py \
  --revision "$NINJARS_VALIDATION_REVISION"
python docs/benchmarkfolder/policy-v2/install-export-abi3-terminal-fixed/export.py
python docs/benchmarkfolder/policy-v2/install-export-abi3-terminal-fixed/runtime_validator.py \
  --build --hide-installation --run
```

`NINJARS_VALIDATION_REVISION` must be the explicit fix commit, not an inferred
working-tree state. The existing Cargo dependency-download cache may be reused;
new build products and temporary files stay in the repository.

Additional focused coverage is audited in `focused-controls.md`. Its commands
are prepared, not executed. The main installation remains root-owned.
