# Fresh generated-runtime audit

Run `audit.py` after `generation-abi3-terminal-fixed/status.json` records all
13 exports complete, preserved inputs and unchanged dependencies. This is a
read-only source/runtime inventory; it neither rebuilds nor evaluates a process.

```sh
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
python -B docs/benchmarkfolder/policy-v2/final-runtime-audit-fixed/audit.py \
  --output docs/benchmarkfolder/policy-v2/final-runtime-audit-fixed/inventory.json
```

The audit compares live complete source manifests with retained clean-build
manifests and generation identities, rehashes every recorded generation
dependency and clean-build executable, then computes the frozen policy's exact
runtime signature for every backend of every case. It includes IREGI, all other
linked reduction/scalar archives, modules, native-link metadata and both
generated link-input files. Only process-specific MODEL/DHELAS archives are
excluded from the runtime signature, exactly as in the frozen policy.

The eight-target representative plan assumes one common runtime group per
backend across all 13 exports. A split is retained and reported as a failure
of that assumption; every additional group needs its own complete DP/QP proof.
The inventory is not itself representative proof or a numerical validation.
