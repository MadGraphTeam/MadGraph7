# Fresh generation after the empty-prefix NLO correction

This namespace is prepared for the committed ABI3/terminal generator after the
NLO empty-prefix declaration correction. The prior13 standalone exports remain
in `../generation-abi3-terminal`; they are not modified or silently relabelled.

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
