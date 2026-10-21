# Final pinned installation and relocated exports

The actual pinned CLI update, installed Fortran smoke, copied/relocated standalone DP/QP/stability calls and full FKS/NLO checks passed. All twenty NLO pole points cancel at the recorded `1e-5` smoke-test tolerance, the soft/collinear test has zero failures, and all 47 focused archived-source tests pass without skips. These are installation/export controls; the stricter core pairwise numerical and performance gates remain separate. It adapts the real `policy-v2/install-export-abi3-final-v1` procedure; historical evidence remains unchanged.

After root freezes a tested commit and grants the shared Cargo/MP resource slot, run from the repository root in the pinned development environment:

```sh
python -B docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/install-export/prepare.py \
  --revision "$FINAL_REVISION" --cpu 190 \
  --bridge-freeze "$FINAL_BRIDGE_FREEZE" \
  --bridge-freeze-sha256 "$FINAL_BRIDGE_FREEZE_SHA256" \
  --previous-installation "$PREVIOUS_INSTALLATION" \
  --previous-archive-sha256 "$PREVIOUS_ARCHIVE_SHA256"
python -B docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/install-export/export.py
python -B docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/install-export/runtime_validator.py --build
python -B docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/install-export/run_runtime.py
python -B docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/install-export/focused_validator.py
python -B docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/install-export/seal.py --live
```

`PREVIOUS_INSTALLATION` must point to a preserved installation whose archive matches `PREVIOUS_ARCHIVE_SHA256`. For this recorded update, the old archive is `800a941faf3cdfdab764fdffe1d5ab73f84984a786aa194596b753af6b6f5d42`, retained intact under `.benchmark-work/policy-v3/final-pins-20261021-v1/install-export/previous-installation`. A fresh reproduction must supply an equivalent old installation in a new campaign namespace; the actual recorded cards remain unchanged. Main-prefix publication subsequently replaced `HEPTools/ninjars` with the accepted `b0d69deb8c16deb13293176296afc5e18dbeeebe778a20fa39b61804fba89d4c` image, so that current main path is not the old installation input for reproducing this update. The separate publication record is [`../main-install-publication/transition.json`](../main-install-publication/transition.json).

The primary CLI installation exercises `install ninjars --update` on a private byte-exact copy of the previous validated installation and its common aliases. The original and a separate old-image copy remain intact. The update uses pinned Git dependencies without local overrides. It executes the ordinary locked release recipe and compiles/runs the installed Fortran native DP, compensated DP and binary128-input smoke. Its complete installed image, exact archive and installed dependency metadata are bound to the archived Cargo manifest/lock. The final source freeze is copied into this durable report directory, so offline verification does not require its private build-time path. The temporary installation and shared Cargo cache remain inside the working repository. This campaign updates only its private prefix; main-prefix publication is recorded separately.

Actual CLI cards generate standalone and full FKS/NLO outputs with internal dependencies. Both outputs are relocated and the private installation hidden before execution. Copied archives, interfaces and linking metadata must be regular files, both exports must resolve the GNU binary128 helpers, standalone calls must preserve return codes 110/120/218 and agree with retained CutTools coefficients, and the NLO checks must pass all twenty pole points and soft/collinear tests.

Focused archived-source checks cover installer failure/retry/rollback, local source isolation, unavailable-only-8 errors, capability detection and old/new statistics. Their logs and exact test names are retained. They are separate from actual engine evaluation.

This proof qualifies installation/export behavior. The new dependency precision/convergence proof and core performance measurements have separate authorities; old exact-replay totals are not installation acceptance conditions. No shower validation is claimed.

The private prefix was unavailable throughout relocated evaluation, then restored byte-exactly in `finally`; the runtime controller and its process group were reaped before the MP slot was handed back. The optional StdHEP compilation warning is retained, and no shower validation is claimed. `verify.py` reads retained evidence only; `--live` additionally hashes the preserved private installation and compiled exports.
