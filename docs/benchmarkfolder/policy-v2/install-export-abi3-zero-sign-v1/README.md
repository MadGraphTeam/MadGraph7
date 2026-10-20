# Corrected binary128 transport: fresh installation and internal exports

The actual `install ninjars` command passed from archived numerical revision
`9f13d10cca5ab6ae5410b6db1f146fa8eddba9b3`, using pinned downloads and no
reference-source overrides. The installed Fortran smoke covers native DP,
compensated DP and genuine binary128-input rescue. Its new static archive is
`832fc0d1198814b9cc80a65e47bc7aa5dc204f7d25701f206c910cd188fbe80b`.
The main installation was not replaced during this validation.

Actual CLI commands generated standalone and NLO outputs with internal
dependencies. Both outputs were relocated and the private installation hidden.
The standalone calls returned DP110, forcedQP120 and stable218; Born, finite
and both pole coefficients agree with the retained CutTools reference at the
predeclared 1e-11 relative-or-absolute bound. The NLO fixed-order compile passed
all twenty pole cancellations and the soft/collinear test with zero failures.
Versioned GNU binary128 helpers resolve through libgcc_s in all checked binaries.
No sibling reference clone or visible private NinjaRS installation is needed.

Sixteen focused Python tests pass. The eighteen-test corrected transport suite
and all 1,392 exact stored-request replays are reused only after matching every
tested bridge build-input byte against the archived installer source. Installed
archive/interface identities and the fresh compiled smoke/runtime results bind
the actual published installation; the installation does not publish Rust
source files. See `focused-unit-results.json` and `installation.json`.

This is Linux/GNU Fortran 15.3 fixed-order validation, not shower validation.
The existing StdHEP build warning is retained in the NLO log; NLO pole and
soft/collinear checks themselves succeed. Compilation/initialization durations
are diagnostic and are not warmed benchmark samples.

Offline evidence verification, from the repository root:

```sh
source .ninjars-build/dev-env.sh
python -B docs/benchmarkfolder/policy-v2/install-export-abi3-zero-sign-v1/verify.py
```

Add `--live` to rehash the preserved hidden installation, copied archives and
executables. The checker never compiles or evaluates an amplitude. Build
products remain in `.benchmark-work`; only inputs, source/link identities,
logs, measured outputs and the evidence verifier are published here.
