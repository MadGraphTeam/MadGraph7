# Installed Fortran interface source

The installer now places `mninjars.f90` beside `ninjars.h` in the installation's
`include` directory. Internal standalone and NLO dependency copying includes
the available source as a regular file in the exported `lib` directory.

From the repository root:

```sh
source .ninjars-build/dev-env.sh
python docs/benchmarkfolder/experiments/interface-source-packaging/reproduce.py
```

Choose a new repository-local directory with `--work` for another run. The
script exercises the changed installation copy recipe with the already
validated archive, copies the seven installed artifacts through the real
exporter, verifies their hashes, and recompiles the exported module source.
It does not claim another numerical-library build or precision test; those
are retained in `../runtime-link-integration`. `installer-tests.log` records
the eight passing lifecycle/export unit tests after this packaging change.
