# Virtual Born–loop API examples

Activate the pyAmpliCol installation which provides `rusticol-config`. From a
language directory, build with `make` and run with
`make run ARGS='PROCESS MU_R2 [resolved]'`. `PROCESS` is the saved process ID;
`MU_R2` is an explicit positive renormalization scale squared. Python uses
`python API/python/check_standalone.py PROCESS MU_R2 [--resolved]` from the
artifact directory.

The examples read every matching row in `API/validation_points.dat`, then make
one bulk evaluation. Native examples use binary64. Python additionally accepts
`--arithmetic`, `--precision`, `--parameters` and `--stability` (exact JSON files).
They print independent Born/virtual squared-order catalogues, all finite/single/
double pole coefficients, output shapes and lane diagnostics. Resolved colour
components may be signed; they are not probabilities. All helicity and colour
axes sum to the inclusive result. Numerical failures are reported explicitly.

The ordinary installed SDK supplies the C, C++, Fortran and Rust interfaces and
one combined native archive. The UMAMI example uses the explicitly versioned
pyAmpliCol virtual extension, not standard UMAMI virtual interoperability. It
uses the same runtime, mandatory statuses and bulk offset/count/stride layout.
No tree matrix-element or finite-only fallback is used.
