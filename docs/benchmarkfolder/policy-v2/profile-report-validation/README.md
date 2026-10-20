# Sampled-profile report validation

All17 report tests pass, including five retained-profile checks. The profile
reader requires membership in the final dataset and identical role/source/runtime,
then recomputes every raw block and stack-category count. Changed measurements,
workers, values, sample shares, raw recordings and acceptance labels are rejected.
The frozen collector and performance decisions are unchanged.

`benchmark.pdf` and `page7.png` are labelled **layout fixtures**, not a final
benchmark or a performance claim. The first profile uses actual controller
samples; the second uses deliberately equal synthetic category counts to check
the complete table layout. Typst0.15.1 compiled the eight-page fixture, and page7
was visually inspected for readable tables and no overflow. `report-data.json`
retains the exact fixture data; it is not a published policy dataset.

Reproduce tests from the repository:

```sh
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
python -B -m unittest tests.unit_tests.loop.test_ninjars_report_v2
```

Compile the retained fixture with Typst using its own directory as input root.
Final profiles must be supplied separately through `--profile-spec`; earlier
controller or research data cannot replace final measured-worker evidence.
