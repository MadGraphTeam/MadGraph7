# Private adaptive optimized-library build

The optimized library was built from the exact qualified adaptive revision 3,
source index `f5550b8719e76e177ad6b73d456cc655815770eb7092a8fabdc776c415c03213`.
No numerical source, manifest, lockfile, installed library or production file
changed. This is compile/link evidence only; the new library was not loaded.

The approved controller index is
`22dab40ded40e662aa7fddcee018fa5837a260e2e6a205a1258930e11125def6`.
It used Cargo release mode, the existing thin-LTO/one-codegen-unit profile,
locked offline dependencies, one build job, and an inclusive 900-second budget
for version inspection plus compilation. There was no affinity override,
CPU reservation, automatic retry or continuation. The reviewed owned-process
supervisor reaped the compiler group before final evidence collection.

The exact private DSO is identified in `bindings.json` and `elf-metadata.json`.
The original debug library `63298bdd…`, qualified test binary `d22953ed…`, all
source-bound historical/current DSOs and source dependencies were checked
before and after. Only hashes and text are packaged; no executable is copied
into this evidence directory. The runtime copy remains isolated under
`.tirust-build/adaptive-qp-rescue-release-build/attempt1/runtime/`.

`cache-use.json` reproduces every Cargo compiler-artifact record, distinguishing
fresh and rebuilt artifacts. The prior source-only cache inspection is retained
in the controller. Its available artifacts were not assumed fresh; the actual
build log establishes selection and reuse. `elf-metadata.json` is obtained by
an authenticated pure ELF-byte parser, not by loading the library.

The earlier 20/20 direct-linked debug Rust unit results and any later debug Fortran
qualification are separate artifacts. They do not automatically qualify an
optimized DSO. Actual release Fortran transport/telemetry, runtime identity,
IEEE behavior and mixed-tier physical stress remain subsequent gates. This
build is not a production promotion or physics acceptance. Final precise
performance measurements are Hua-Sheng's local TODO; shared-host observations
remain provisional and this package makes no timing-acceptance claim.

`python -B verify.py` checks the portable inventory, exact source/controller
bindings, Cargo profile and build records without native execution. Add
`--live` to rehash original protected inputs and the isolated DSO, and to
reparse its ELF metadata without dynamic loading.

The completed compiler process took 444.99065815936774 seconds (administrative duration only). Cargo reused 140 artifacts and rebuilt 4. The isolated DSO SHA256 is `c5de24e6c41a90caf9e34684392d0193588522f26e6bbac646bdf74e5f151cb7`, with 69157632 bytes.
