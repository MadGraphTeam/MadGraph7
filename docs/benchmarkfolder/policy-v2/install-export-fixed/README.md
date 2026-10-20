# Successful installation and relocated internal exports

This clean attempt uses committed source
`85795ccbb68fa38721d8cb1eade5b4e938c3ba7d`, including the NLO elemental-classifier
fix. The previous failed compilation remains intact in `../install-export`.

The real `install ninjars` CLI command downloads the pinned dependencies without
local source overrides and runs its compiled Fortran DP/QP smoke tests. It
installs the interface source, module, header, archive and native-link metadata.
The archive SHA256 is
`72a820a1fb74904e9b3d3a43562132b365a947304ac41abced46f5d407a0c816`,
identical to the production numerical library.

Actual CLI cards generate standalone `u u~ > z [virt=QCD]` and full NLO
`u u~ > z [QCD]` outputs with internal dependencies. Both outputs are renamed,
the installation is renamed away, and `LD_LIBRARY_PATH` and `LIBRARY_PATH`
are removed before numerical execution.

The retained checks pass:

- Standalone forced DP, forced QP and ordinary checked DP return 110, 120 and
  218, respectively. Born, finite and both poles agree with the retained
  CutTools reference within `1e-11 * max(1, abs(reference))`.
- The complete fixed-order NLO compilation succeeds with the actual
  `-fno-automatic` flags. All 20 pole checks pass at the standard `1e-5`
  installation-test tolerance, and all soft/collinear checks have zero failures.
- Standalone, `check_poles` and `madevent_mintFO` resolve their binary128
  arithmetic helpers to versioned GNU runtime imports. Their dynamic
  dependencies resolve with the external library environment removed.

These are installation/export tests on Linux x86-64 with GNU Fortran 15.3.
They do not replace the eight-process numerical and performance gates.
Parton showers are not tested; the optional StdHEP build warning in the raw
log does not prevent this fixed-order NLO build or its numerical checks.

Run from the repository root in the pinned environment, using a fresh work
namespace:

```sh
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
python docs/benchmarkfolder/policy-v2/install-export-fixed/prepare.py
python docs/benchmarkfolder/policy-v2/install-export-fixed/export.py
python docs/benchmarkfolder/policy-v2/install-export-fixed/runtime_validator.py \
  --build --hide-installation --run
```

Installation and QP execution require exclusive use of the shared Symbolica
process slot. `installation.json`, `exports.json`, `relocation.json`,
`installation-hidden.json` and `runtime-results.json` retain the source,
commands, artifacts and results. The cards, raw logs and complete source
inventories are included. `evidence.json` binds every retained file.
