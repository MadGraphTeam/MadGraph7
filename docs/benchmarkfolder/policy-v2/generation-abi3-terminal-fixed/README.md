# Fresh generation after the empty-prefix NLO correction

All thirteen CLI exports and clean subprocess builds completed at
`b44d20e725f1ec68589df20284e4ce9116e9e2ab`, after the NLO empty-prefix declaration
correction. The prior13 standalone exports remain
in `../generation-abi3-terminal`; they are not modified or silently relabelled.

`summary.json` indexes the completed builds. The independent inventory at
`../final-runtime-audit-fixed/inventory.json` rechecks source, executable and
dependency hashes and finds one common full numerical runtime group for all
thirteen exports. This supports the prepared representative coverage; it does
not replace amplitude comparisons or performance measurements.

Run from the repository root after committing the numerical fix:

```sh
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
python -B docs/benchmarkfolder/policy-v2/generation-abi3-terminal-fixed/regenerate.py
python -B docs/benchmarkfolder/policy-v2/generation-abi3-terminal-fixed/summarize.py
```

The runner requires clean committed numerical sources, ABI3 with genuine QP,
all13 fresh CLI exports/clean builds, identical original physical cards/points,
exact model-correction deltas, and unchanged dependencies/protected sources.
It performs no loop-amplitude evaluation. Fresh candidate parity, focused
controls, all eight complete numerical streams and timing gates remain required.
Products stay inside `.benchmark-work/policy-v2/generation-abi3-terminal-fixed`.
