# Final candidate generation preparation

This separate namespace will hold fresh CLI exports after the compensated DP
retry (ABI 3) and terminal QP wavefunction optimization are committed and the
corresponding NinjaRS library is installed. Preparation alone is not numerical
or performance acceptance.

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
must explicitly select these new outputs and bind their exact generated
sources and installed libraries.
