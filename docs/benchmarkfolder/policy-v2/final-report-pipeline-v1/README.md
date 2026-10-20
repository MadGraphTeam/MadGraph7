# Final report pipeline extension

This prepares the final report without replacing the historical top-level
`benchmark.typ` or `benchmark.pdf`. No numerical or runtime gate is certified by
the synthetic layout fixtures in the report tests. The final source/runtime,
complete eight-case dataset, loop-induced supplement and installed dependency
provenance still have to be collected, validated and reviewed.

`tools/ninjars_report_v2.py` accepts the existing validated core dataset and adds:

- `--supplement FILE`: the separately validated `madloop-final-supplement-v1`
  loop-induced/threshold/precision manifest;
- `--runtime-provenance FILE`: a sealed
  `madloop-final-runtime-provenance-v1` manifest described below;
- `--validation-spec FILE`: the reviewed MG7 regression manifest, independently
  checked by its retained `verify_validation.py` adapter against the exact final
  candidate generator files.

The supplement must match the core dataset identity, the unique corrected
candidate generator identity, the NinjaRS reducer-runtime signature and literal
original revision `27147b3300f556a75661a886e92bdbe00558ee25`. Each raw-bound
supplement reader recomputes comparisons and timing conclusions. Missing evidence
stays pending. A complete core pass alone cannot complete the final campaign.
The renderer passes the complete sealed measured reducer-runtime signature to
the supplement precision reader. Its installed archive/module/config/native-link
metadata must match that descriptor; an arbitrary declared runtime-identity
string cannot relabel a stale precision archive as the measured runtime.

## Numerical stability

`stability_snapshot` independently recomputes each corrected target's largest
discrepancy against every other corrected target. The scale is the common
point's largest absolute Born/Laurent value across corrected targets and all
orders. An erroneous uncorrected original value cannot enlarge that scale.
The output retains the worst point/order/component and the number of failures
against the unchanged component-dependent acceptance tolerance.

MadLoop estimates are a separate population: minimum, median, nearest-rank 95th
percentile, maximum and exact-zero count are retained over all point/order slots
and over the total-order slot separately. Estimates may differ between split
orders. An estimate is never called a measured error or converted into a claim
of known significant digits. Successful DP/QP statuses count once per physical
point; timed histograms retain every traversal. A finite stream is not a bound
on the instability rate elsewhere in phase space.

## Final installed runtime provenance

The old setup `provenance.json` contains ABI-2 metadata and historical dependency
pins. Those fields are removed from the new report snapshot. ABI is read from
the exact measured candidate's retained `ninjars-config.json`; final revisions
require the installed Cargo files and installation evidence. The live source
checkout cannot identify an earlier linked archive.

The new manifest is created with `policy.seal` and has the following fields:

```json
{
  "schema": "madloop-final-runtime-provenance-v1",
  "core_dataset_identity_sha256": "CORE_DATASET_IDENTITY",
  "literal_baseline_revision": "27147b3300f556a75661a886e92bdbe00558ee25",
  "candidate_generator_identity_sha256": "GENERATOR_TREE_STATE_IDENTITY",
  "candidate_runtime_identity_sha256": "NINJARS_REDUCER_RUNTIME_SIGNATURE",
  "installation": {"path": "retained/installation.json", "sha256": "SHA256"},
  "cargo_toml": {"path": "retained/Cargo.toml", "sha256": "SHA256"},
  "cargo_lock": {"path": "retained/Cargo.lock", "sha256": "SHA256"},
  "installed_config": {"path": "retained/ninjars-config.json", "sha256": "SHA256"},
  "dependencies": {
    "ninjars": "FULL_40_DIGIT_GIT_REVISION",
    "oneloopmaster": "FULL_40_DIGIT_GIT_REVISION",
    "symbolica": "FULL_40_DIGIT_GIT_REVISION"
  },
  "identity_sha256": "POLICY_SEAL_CONTENT_DIGEST"
}
```

The installation record uses the existing CLI-installation evidence format:
`passed=true`, `exit_code=0`, `compiled_smoke_passed=true`, `abi_version=3` and a
`files` map containing `Cargo.toml`, `Cargo.lock`, `lib/ninjars-config.json` and
`lib/libninjars.a`. Every retained file hash must match, and the archive/config
hashes must equal the measured target binding. Cargo lock entries for `ninjars`,
`oneloop` and `symbolica` must carry full Git revisions, with the direct NinjaRS
and Symbolica override pins agreeing with the installed Cargo manifest. Declared
revisions, stale archives, changed metadata, old ABI and hidden local source
overrides are rejected. A final campaign cannot be marked passed without this
provenance gate.

## Reviewed MG7 regression suites

The regression adapter independently verifies completed raw unittest footers,
failure/error counts, explicit skip names/reasons and reviewed expected IO
fixtures. It compares each tested production hash in the generator scope with
the final timing candidate's retained `tree_state.tracked` map. Exact checked
source/test/fixture/log hashes join the report input manifest. The verifier is
loaded without creating bytecode inside its sealed evidence directory.

The current bounded broad campaign records ninety cases: eighty-nine passed and
one existing GoSam Python-3 restriction explicitly bypassed. Its retained failed
and incomplete predecessors stay separate. Those generator/Fortran/export suites
do not execute the Rust reduction or scalar engine; numerical installed-archive
controls remain a distinct gate. The PDF lists suite counts and skip reasons,
and missing regression evidence prevents a complete campaign declaration.

## Rendering and publication

```sh
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
python tools/ninjars_report_v2.py CORE_DATASET --compile \
  --supplement FINAL_SUPPLEMENT --runtime-provenance FINAL_RUNTIME_PROVENANCE \
  --validation-spec FINAL_MG7_REGRESSION_MANIFEST \
  --generation-inputs FINAL_CLI_INPUTS --profile-spec FINAL_PROFILE_MANIFEST \
  --timestamp 'RETAINED_VISIBLE_UTC' --output NEW_REPORT_DIRECTORY
```

Only provide generation/profile options when their independently bound evidence
is available. The renderer refuses an existing destination. Its fixed PDF
creation timestamp and explicit visible date make PDF bytes reproducible despite
the ambient `SOURCE_DATE_EPOCH`. Template/style/report/supplement source hashes
and all input hashes are retained in `report-data.json`.

The draft template keeps the attributed pyamplicol style: Libertinus typography,
the same margins and navy headings, striped tables, pale panels and logarithmic
green–white–red cards. It adds loop-induced timings, actual versus estimated
stability, a fixed top-threshold grid and genuine higher-precision controls.
Historical kernel/process improvements and negative DP results remain separate
from final full-evaluation acceptance. Final promotion to the top-level report
must preserve the old snapshot, follow complete validation, and include PDF
visual inspection; it is not performed by these preparation changes.

`validation.json` records the focused MG7 report tests. It is a software-check
record, not a final physics dataset or a benchmark result.
