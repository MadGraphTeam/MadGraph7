# Isolated DP raw-finite-check experiment

**Decision: rejected for production; no native adapter comparison or timing
was run.** The public diagnostic must continue rejecting nonfinite values
even if a faulty or mocked provider reports success. Removing the DP scan
changes that defensive behavior. Existing QP nonfinite-success tests remain
intact, but their current coverage does not excuse changing DP semantics.

The bridge already validates every packed DP component before reporting
success. This experiment removes the generated Fortran adapter's duplicate
raw scan only, using the original compiler flags and frozen split-workspace
DSO. It does not change production files. The exact scope, error-path argument,
QP transport distinction, and retained stability/contraction checks are in
`finite-contract-proof.md`.

`adapter-finite-build/build-metadata.json` records the compiler recipe,
environment, original reconstruction, unchanged link inputs, and the sole
changed object. The original executable is reconstructed byte-for-byte before
changing that object. QP disassembly is unchanged and both versions retain
the GNU Fortran IEEE procedure entry/exit calls (`static-audit.json`).

`audit_existing_contract.py` rechecks all raw payloads from the previously
retained split-bridge native validation without loading a library. Its output
records 80 finite successes and 10 failures with untouched output sentinels,
including full/partial selections, cache reuse/bypass, and failure recovery.

The prepared, **unexecuted** adapter fixture covers N=1–6, ranks 0–2, CTMODE
1/2, and DP/QP. It would compare exact output and input-mutation bits, stability flags, and IEEE
invalid/divzero/overflow/underflow/inexact flags, with bounds checking. Those
72 planned calls per variant are not numerical evidence. No numerical or
runtime benefit is claimed. The isolated timing executable was built with
the original flags but never run; it contains no fixture instrumentation.

Source-analysis reproduction and the completed isolated build recipe follow.
Run from the MadGraph checkout. The historical build name already exists;
use a new name to reproduce it without replacing evidence:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
source tools/tirust_env.sh
python3 docs/benchmarkfolder/experiments/n1-profile/fortran-dp-finite/audit_existing_contract.py
python3 docs/benchmarkfolder/experiments/n1-profile/fortran-dp-finite/prepare_finite.py \
  --name adapter-finite-build --build
```

The following commands document the prepared protocol only. They were not
run after the decision to retain the public diagnostic's defensive checks:

```sh
python3 docs/benchmarkfolder/experiments/n1-profile/fortran-dp-finite/check_layout.py \
  --name layout-parity-1 \
  --library .tirust-build/n1-profile/split-bridge/libmadloop_tirust.so --run
python3 docs/benchmarkfolder/experiments/n1-profile/whole-amplitude/run.py \
  --name split-adapter-finite-1 \
  --library .tirust-build/n1-profile/split-bridge/libmadloop_tirust.so \
  --build-metadata docs/benchmarkfolder/experiments/n1-profile/bridge/split-build-metadata.json \
  --fortran-build-metadata docs/benchmarkfolder/experiments/n1-profile/fortran-dp-finite/adapter-finite-build/build-metadata.json \
  --original-use-candidate-library --cpu 9 --loops 200 --pairs 7 --profile-repetitions 0
python3 docs/benchmarkfolder/experiments/n1-profile/whole-amplitude/analyse.py \
  docs/benchmarkfolder/experiments/n1-profile/whole-amplitude/split-adapter-finite-1
```

The controllers serialize native Rust reductions using
`.tirust-build/numerical-tests.lock`. The whole-amplitude controller compares
the frozen NinjaRS baseline, original adapter with the split DSO, and changed
adapter with that same DSO. It binds point/card/compiler/executable/runtime
identities and retains exact values, statuses, and actual backend counts.
Any future run would be a new experiment and require a separate decision.
The current source-only result cannot satisfy any performance acceptance
requirement. A future implementation that preserves public diagnostic
behavior must be measured independently; the rejected removal establishes no
runtime benefit for such an implementation.
