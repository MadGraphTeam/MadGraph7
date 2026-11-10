# Local launcher entrypoint correction proposal

The real tracked launcher is `bin/madgraph`. The existing local installer uses
absent `bin/mg5_aMC`; its old mocked unit test expected the same wrong token.
`local-launcher-entrypoint.patch` changes exactly that argv token in the tool
and strengthens installation tests. `proposed/` holds the exact replacements.
The patch passes `git apply --check` and has not been applied.

The six focused tests exercise real emitted argv and regular launcher bytes,
tracked source-inventory binding, successful source/native/log receipt provenance,
and retained nonzero-exit, ENOENT, missing-publication and source-drift failures.
All subprocess installation calls are mocked. The negative control confirms
that the old tool fails the new exact-argv assertion. Ten successor handoff
checks cover immutable input/source/card ancestry, parser commands and the
explicit pinned `nix develop` environment. `final-tests.log` records 16 passing
tests. The earlier exploratory log is retained separately.

`candidate-timing-handoff-4/` is a complete proposed additive successor for
`docs/benchmarkfolder/campaigns/invariant-symmetry-1/timing-handoff-4/`.
Its index deliberately uses those final canonical key prefixes. The packet
preserves every handoff3 file hash, pins corrected tool/test and tracked
launcher bytes, and documents Nix shell activation followed by environment
setup. It retains fresh local installation, all eleven six-lane numerical
qualifications and precise timing as Hua-Sheng's local TODOs. The handoff
checks omit the old assumptions that active numerical directories are absent.

The installation receipt schema and before/after exporter inventory checks
are unchanged. The launcher was already in the exact canonical generator
inventory; tools/tests are outside that inventory. No source override, alias,
symlink, native artifact or campaign-manifest change is introduced. A general
unit test checks the current tracked launcher inventory; frozen canonical
inventory equality stays in the dedicated handoff checks.

Production files, sealed handoff3, canonical evidence and the active numerical
supervisor are untouched. No installation, compiler, amplitude, CPU reservation,
shared-host timing, commit or push was performed. Promotion must apply the exact
two-file patch, add the exact six handoff4 files, and update the separate report
successor's handoff pin atomically under the root's review. The proposal test
harness is for pre-promotion use because it expects the old production argv as
its negative control. Post-promotion, run the normal focused production unit
tests and the handoff4 test file directly.
