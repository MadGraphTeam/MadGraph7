# ABI 3 candidate generation at c5496ca9b

All eight core processes and five controls were generated and clean-built at
`c5496ca9b148b16a931214e1d541ead100ed1aad`, with the installed ABI 3 bridge.
`summary.json` indexes all 13 exports. Original physical inputs, protected
generated sources and dependency hashes stayed unchanged. No loop amplitude
was evaluated; this is generation/build evidence, not numerical or performance
acceptance.

A separate internal NLO export found the empty-process-prefix IEEE declaration
bug retained in `../install-export-abi3-terminal/failure.json`. These successful
standalone builds do not close that NLO failure. Keep this namespace unchanged;
the fixed generator requires its own generation and runtime evidence.

From the repository root:

```sh
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
python docs/benchmarkfolder/policy-v2/generation-abi3-terminal/regenerate.py
python docs/benchmarkfolder/policy-v2/generation-abi3-terminal/summarize.py
```

The runner refuses existing outputs, changed or untracked numerical sources,
and an installation without ABI 3 and genuine higher precision. It generates
all eight core processes and five controls through actual MG5aMC command files.
Each receives a clean subprocess build, source inventories, the exact original
physical card and fixed point stream. The model delta must remain confined to
the already verified precision corrections. No loop amplitude is evaluated by
this preparation step.

Build products remain under
`.benchmark-work/policy-v2/generation-abi3-terminal`. Historical generation,
numerical and timing evidence remains in its original namespace. Final workers
must explicitly select the intended outputs and bind their exact generated
sources and installed libraries.
