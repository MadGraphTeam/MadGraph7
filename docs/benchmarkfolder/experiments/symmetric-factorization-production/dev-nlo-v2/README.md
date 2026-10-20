# Diagnostic NLO factor export, second harness attempt

The first fresh CLI export in `../dev-nlo-v1` succeeded and selected nine private empty-prefix factor calls. Its standalone source-inventory helper then rejected an absent `dire_fortran.cc` link before any compilation or numerical evaluation. That attempt is preserved unchanged.

This independent second attempt re-exports `u u~ > z g [QCD]` through the actual CLI, binds the current dirty generator patch, and performs the same Ninja6-only FO compile/soft-collinear/pole controls on CPU2. No physics or generated production source is edited. The local inventory records every absent template link with its exact target and reason. The exporter links optional Dire, obsolete `mint.inc`, and external OLP contract files unconditionally; no FO make dependency uses Dire or `mint.inc`, and no active generated Fortran includes `mint.inc`. `run_card.inc` is pending before compilation and must be resolved afterward: `Template/NLO/Source/makefile` invokes `treatcards run`, also called by the normal compile command. All unknown missing inputs are rejected.

Commands (from the repository root):

```sh
source .ninjars-build/dev-env.sh
python docs/benchmarkfolder/experiments/symmetric-factorization-production/dev-nlo-v2/run.py > docs/benchmarkfolder/experiments/symmetric-factorization-production/dev-nlo-v2/controller.log 2>&1
```

This is dirty-source integration evidence, not final performance or all-process acceptance.
