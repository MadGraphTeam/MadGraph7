# Explicit physical-card ancestry for a new local campaign

The reviewed private proposal was applied byte-for-byte to three live benchmark
tools and one new test. All 61 Python tests pass against the actual live modules,
whose resolved paths and unchanged before/after hashes are retained in
`qualification.json`. There were no real campaign registrations, compilations,
native evaluations or changes to the old handoff in this work.

The opt-in `register-campaign --inherit-physical-cards` freezes only the immediate
named predecessor's observed physical-card inputs inside the new campaign
manifest. It requires complete case coverage, matching model/policy/point inputs,
actual card bytes and consistent observation ancestry. It does not copy or
relabel results. The local launcher imports only those explicitly declared card
hashes and revalidates their evidence; fresh source/native/installation-receipt
binding, all local builds, and all eleven numerical cases remain mandatory.

The old campaign remains bound to its original source. A future campaign may be
registered only after the actual source promotion/commit and reviewed current
installation. A source-only `summarize --campaign NEW_ID` must then create its
honest pending summary before local preparation. The prior handoff and measured
results remain unchanged. This package does not claim that a successor campaign
already exists or has passed any physical test.

`source-text.json.gz` contains all fifteen UTF-8 files from the frozen private
proposal, including its original index, exact source snapshots, patch, 11-case
read-only ancestry preview and private test log; it also includes the two
unchanged existing test files used by the live qualification. It contains no
native files, caches or Python pickles. `proposed.patch` is repeated separately
for review. The original proposal index SHA256 is
`ed0d91e6cc71ac555b67ec44145bc1f4cb8ccbbfb68e3ea6a3ee19821a119110`;
the exact applied patch SHA256 is
`3cd7fc056ed4d4fded35a7ffa4978ebd8ac812d74b5450d0b0965643d7839697`.

Portable hash/source/receipt verification, with no repository or native runtime:

```sh
python verify.py
```

Re-run the live tests from the repository root, using the pinned environment:

```sh
source tools/tirust_env.sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest \
  tests.unit_tests.loop.test_tirust_physical_card_origins \
  tests.unit_tests.loop.test_tirust_benchmark_local \
  tests.unit_tests.loop.test_tirust_benchmark_campaign
```

The live receipt SHA256 is
`fbcfd030b6b49bbc62fa3a0cc9dbeb8573498cada6f1a66abde4b7bcc4ba49b0`.
The new tests cover explicit registration with no measurements, changed or
missing evidence, model/policy/point/case mismatches, ancestry conflicts, unsafe
paths, and unchanged old-manifest behavior. The existing tests retain the strict
source/native receipt, raw numerical prerequisite and historical-failure gates.
