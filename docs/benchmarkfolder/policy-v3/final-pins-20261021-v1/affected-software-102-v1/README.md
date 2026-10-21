# Clean DP62/report40 software checkpoint

This fresh software-only run uses clean tracked source archive `9b5abb345452becddd37ee603c9bed17aed7db04` with **exactly four explicit overlays**: the frozen DP62 producer/test and report40 consumer/test. No untracked CLI helpers or tests are copied. Both ignored parent directories, `.ninjars-build` and `.benchmark-work`, are absent before discovery. The report fixture creates its own ignored parent when needed.

The actual outcome is **102 tests run: 101 passed, one unchanged conditional skip, zero failures/errors**. DP contributes 62 passing tests. The report contributes 40 cases: 39 passed and its actual-generated-QCD2 count control is skipped because a pristine checkout does not contain the ignored generated process. Root explicitly requested no generated-process smoke. This skip is reported, not relabelled as a pass. Runtime is 2.789 seconds unittest; the owned command ran on CPU5 and its complete process group was reaped.

`test-inventory.json` records the exact 102 unique IDs and module counts. `run-result.json` records every outcome. `test.log` independently binds those 102 IDs/statuses and the 102-test/one-skip footer. The read-only verifier checks both representations, exact frozen source hashes before/after in the archive and working checkout, and the absence of ignored parents/CLI overlays. Ten exact source images (four overlays, imported package initializers, the observer Fortran **source**, and two hash-retained timing/numerator helpers) are retained in a deterministic text-only archive. No ELF, object, static/shared library, bytecode, synthetic PDF or PNG is published here.

The exact old source-only independent receipts are retained byte-for-byte: DP `d62d8b3c27954138d4779cea165b206df98939836fa47d8ddbf16f42d074728c` and report `33c2684dc243f62b03c413efee3faaa65ff68ea26cbe41be3a5a9e928078eb24`. They are source reviews, not additional test runs or native observations. The DP receipt also discusses separate held preparation; this capsule does not promote that preparation to completed physics.

No MadLoop/native amplitude evaluation, compiler, generated-process check, performance profile or actual-data figure is run. Existing proofs, the main benchmark PDF/style and frozen numerical source remain unchanged. The earlier ec5/36 and previous62 owner observations remain historical snapshots rather than being relabelled as this fresh run.

## Verification

From the repository root:

```sh
python -B docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/affected-software-102-v1/verify.py
python -B docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/affected-software-102-v1/verify.py --live "$PWD"
```

Offline verification uses only retained software evidence; `--live` additionally requires the ten exact tested source images to match the current checkout. A later source change correctly fails `--live`; use the retained source snapshot for historical verification. Optimized Python is explicitly rejected. `manifest.json` is a direct payload SHA256/byte inventory, following the previous affected-software capsule; it has no separate policy-seal identity formula.

## Reproduction

Create a new empty archive directory and run `git archive 9b5abb345452becddd37ee603c9bed17aed7db04` into it. Copy only the four paths in `source-before.json` from `software-sources.tar.gz` over that archive, after checking their SHA256s. Leave ignored parents absent and do not copy any CLI overlay. Use the pinned project Python/Matplotlib/font environment, set `TMPDIR` to a separate directory inside the working repository, and keep the shown thread environment variables at 1. Run the retained runner from the archive:

```sh
cd NEW_CLEAN_ARCHIVE
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="$PWD" \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
taskset -c 5 python -B ABSOLUTE_CAPSULE_PATH/run-suite.py \
  --archive "$PWD" --output NEW_SOFTWARE_LOG_DIRECTORY
```

`command.json` preserves the actual observed absolute paths, interpreter, environment and owned process metadata. The large original source tar remains private; its SHA256/bytes and base revision are retained in `archive-preparation.json`. The small software-source archive does not contain native binaries. Full physics and final-publication gates remain separate pending authorities.
