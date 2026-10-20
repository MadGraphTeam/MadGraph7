# Diagnostic NLO factor export, third harness attempt

The first fresh CLI export in `../dev-nlo-v1` succeeded and selected nine private empty-prefix factor calls. Its standalone source-inventory helper then rejected an absent `dire_fortran.cc` link before any compilation or numerical evaluation. That attempt is preserved unchanged.

This independent third attempt re-exports `u u~ > z g [QCD]` through the actual CLI, binds the current dirty generator patch, and performs the same Ninja6-only FO compile/soft-collinear/pole controls on CPU2. No physics or generated production source is edited. The local inventory records every absent template link with its exact target and reason. The exporter links optional Dire, obsolete `mint.inc`, and external OLP contract files unconditionally; no FO make dependency uses Dire or `mint.inc`, and no active generated Fortran includes `mint.inc`. `run_card.inc` is pending before compilation and must be resolved afterward: `Template/NLO/Source/makefile` invokes `treatcards run`, also called by the normal compile command. All unknown missing inputs are rejected.

Commands (from the repository root):

```sh
source .ninjars-build/dev-env.sh
python docs/benchmarkfolder/experiments/symmetric-factorization-production/dev-nlo-v3/run.py > docs/benchmarkfolder/experiments/symmetric-factorization-production/dev-nlo-v3/controller.log 2>&1
```

This is dirty-source integration evidence, not final performance or all-process acceptance.

The second harness attempt is also retained unchanged: its generic fixed-form source scanner indexed a short preprocessor line before any build. This attempt checks active INCLUDE statements directly for the optional `mint.inc` audit. Actual factor-batch parsing and all physics/build settings are unchanged.

The normal CLI FO build passed the soft/collinear checks with zero failures and all20 analytic pole checks using Ninja6. The polynomial make recipe uses `-O3` together with the NLO `-fno-automatic` setting; exact make inputs are retained. All nine selected calls have empty MP process prefixes, per-batch IEEE imports/declarations, and a fresh entry-flag query immediately before the call. Runtime link evidence includes versioned GNU binary128 helpers and the `-lgcc_s` prefix before NinjaRS.

The unmodified forced-QP follow-up is preserved under `forced-qp/`: it was stopped after36.8seconds because the stock FKS wrapper requests stability checking while forced `CTModeRun=4` deliberately returns unchecked H1. The wrapper repeatedly labels these results initialization T4 and the stock pole driver discards them indefinitely. This is a driver-policy incompatibility, not a numerical failure. Cards, source, and executable were restored unchanged.

A separate clone changes only the pole-check driver (`forced-qp-driver-v2/driver.patch`): disable the wrapper's automatic stability repeat before each call, print return codes, and abort after100 attempted calls. The first wrapper call still overrides the setting once. Both Ninja6 and NinjaRS8 then pass all20 poles, each with20 counted status120 results (genuine QP, explicitly unchecked H1), after one initialization140. No numerical model, reduction, or polynomial routine was changed. Fast/fallback factor counts were not instrumented and are not claimed. The existing analytic-pole tolerance1e-5 is independent of the strict final coefficient comparison gate.

The final compiled parameter card is retained separately because normal NLO compilation sets the external Z width to zero for gauge cancellation. Initial cards and all failed attempts remain available. CPU2 was used throughout; elapsed build/control durations are diagnostic only. All numerical children exited, and the Symbolica slot was released.

Read-only verification: `python docs/benchmarkfolder/experiments/symmetric-factorization-production/dev-nlo-v3/verify.py`. Add `--live` to rehash the disposable generated source/runtime/executables as well. The manifest covers all three NLO attempt directories, excluding itself and Python caches; it does not alter the earlier191-file helper-validation seal.
