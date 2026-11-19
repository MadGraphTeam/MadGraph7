# Compensated callback regression fixtures

These are fixed regression inputs and numerical oracles, not benchmark output.
The ordinary Rust suite reads them locally; tests require no downloads.

`inputs.json` and `corrected-results.json` retain six MadLoop QCD3 requests and
their complete ordinary/compensated reduction results. `expansion-results.json`
retains the 30 `t3` callbacks exercised by
`guarded_frozen_callbacks_match_all_480_oracle_components`: all callback inputs
and all 480 real components of the Float256 reference rounded to binary64.

The callback fixture is a lossless projection of the
[original capture](https://github.com/MadGraphTeam/MadGraph7/blob/cc6f6fe6dd344a0a767dcfe52341c147e39a193a/vendor/ninjars/tests/data/compensated/expansion-results.json)
(SHA256 `8bdf9a439db23f793ca38f774997d866d6a0fa279e00a086d81cee21a3d67b76`).
For every outer row it keeps `direction`, `loop`, and
`results[0].callbacks` with `kind == "t3"`; each callback keeps `kind`, `inputs`,
and `high_rounded`. Unused native callback outputs and other captured callback
kinds/results remain in the immutable original. Every retained numeric value,
including its binary64 bits and signed zero, is unchanged. The projection reduces
the fixture from 1,350,447 to 131,790 bytes without reducing test coverage.

The complete commands and scientific provenance are linked in
[`UPSTREAM.md`](../../../UPSTREAM.md).
