# Upstream checkpoint and integration boundary tests

The final-round read-only check found NinjaRS main still at
`0d0d7e7a8033431b60a8580731084742e0655cda` and oneloopmaster main still at
`09b2c7497743db875ed446da95123175dd00b1c8`. Both match the committed Cargo
inputs. There are no advertised version tags. The checker inventories every
transitive Git source and rejects floating, short or inconsistently resolved
Git revisions before contacting a remote. It never changes a pin or reference
checkout. A new snapshot requires a new output path.

Thirty focused MG7 Python tests pass, including nineteen new tests in
`test_ninjars_upstream.py` and `test_ninjars_integration_boundaries.py`.
The new integration checks cover absent precision metadata, literal QP
capability, incompatible metadata, publication rollback, partial installations,
actionable unavailable-library errors, failed smoke followed by successful
retry, simultaneous isolated NinjaRS/oneloopmaster overrides, and mixed legacy
and contextual statistics aggregation. Compilation is mocked in installer
failure tests; staging, preservation, publication and copying use actual
temporary files. These tests supplement the numerical bridge tests.

A three-line diagnostics fix makes a direct Python stability check with
`MLReductionLib=[8]` and an entirely absent `ninjars_dir` setting fail before
generation with the same actionable `install ninjars` message used for an
invalid directory. The test exercises absent, disabled, missing and empty
directory configurations. The CLI already rejected this configuration.

The independent, read-only live verifier also rehashed the corrected private
installation and relocated standalone/NLO builds. Its 63-file proof passes:
compiled installed smoke, standalone DP110/forcedQP120/stable218, twenty NLO
pole cancellations and the soft/collinear check. This audit executes no
reduction engine. The retained proof documents the platform and shower-testing
limit. Its archive is
`832fc0d1198814b9cc80a65e47bc7aa5dc204f7d25701f206c910cd188fbe80b`.

The separately inspected Symbolica fork needs care. Its default main
`e0bbb78d560c6efaffd1198ba356d9ea8d80fb9e` is an ancestor 596 commits behind
the explicitly required `e679e8c129ce15ec50c46ffb4ff4b59aeb7813b8`, rather
than an upgrade. The `ninjars-pin` branch has one new descendant,
`9cdc2949cdb153478c1eca0d40690dc0abfc616e`. Its retained three-file diff
changes mixed-precision Float multiplication/division to one rounding of the
exact operation and keeps multiplication by exactly ±1 exact at the other
operand's precision. Its new tests address the 257-bit result of a 256-bit
square root, complex conjugation cancellation and evaluator cancellation.
Those upstream tests were inspected, not executed here. The API appears
compatible by source review, but output values and tracked precision can
change. NinjaRS and oneloopmaster still explicitly require the earlier commit;
no independent transitive update is adopted during this final round. A future
upstream update requires isolated numerical and performance validation.

Recheck while the campaign is active, using a fresh checkpoint path:

```sh
source .ninjars-build/dev-env.sh
python -B tools/ninjars_upstream.py --output /path/to/new-checkpoint.json
python -B -m unittest tests.unit_tests.various.test_ninjars_upstream \
  tests.unit_tests.various.test_ninjars_integration_boundaries \
  tests.unit_tests.various.test_ninjars_install \
  tests.unit_tests.various.test_ninjars_statistics
python -B docs/benchmarkfolder/upstream-final-audit-v1/verify.py --sources
```

Exit statuses from the upstream checker are 0 for matching main pins, 1 for a
pending update, and 2 for invalid local inputs or an unavailable remote. Tags
are recorded as an inventory, not ranked as releases. This command is an
explicit check, not a promise of unattended monitoring after the task stops.
All clone/build products remain under `.benchmark-work`; no source checkout,
binary, cache or credentials are published with this evidence.
