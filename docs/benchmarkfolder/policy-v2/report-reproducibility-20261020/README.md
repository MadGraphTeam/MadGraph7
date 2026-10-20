# Deterministic report metadata

The v2 renderer explicitly passes `--creation-timestamp 315532800` to Typst
and records that command in `report-data.json`. Its displayed date remains
`generated_utc`. The printed reproduction instructions preserve that date,
choose a new output directory, and identify any explicit generation/profile
inputs needed to reproduce the same content.

All eighteen focused report tests pass. The new compiled control renders the
same fixture with ambient `SOURCE_DATE_EPOCH` values 0 and 1893456000, then
requires identical snapshot JSON and PDF bytes. It uses the installed Typst and
fonts; those inputs must also match for binary reproduction elsewhere.

Reproduce from the repository root:

```sh
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
python -B -m unittest tests.unit_tests.loop.test_ninjars_report_v2
```

`evidence.json` binds the tested source and retained test output. The PDF builds
use isolated temporary test fixtures; they are not measured benchmark results.
The final eight-process dataset, acceptance gates and report remain pending.
