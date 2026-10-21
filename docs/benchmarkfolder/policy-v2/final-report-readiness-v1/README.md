# Final report assembly readiness

This is a source-only assembly note after installation/publication commit
`2bf47cd47`. It changes no reader, policy, measured input or historical report.
No tests, generation, profiling or numerical evaluations were run for this note.

The final installed NinjaRS archive is
`800a941faf3cdfdab764fdffe1d5ab73f84984a786aa194596b753af6b6f5d42`,
from revision `d05b59f3277ce0d727cb2816d8b7ad45107e58c4`. Its installation,
Cargo files, ABI-3 configuration and Fortran/interface/link metadata are retained
in `policy-v2/install-export-abi3-final-v1/`. The publication transition in
`policy-v2/main-install-publication-v1/` records the identical main-prefix copy.
These are installation proofs, not completed benchmark datasets.

## Retain these inputs during generation and collection

| Stage | Minimal retained files and bindings |
|---|---|
| Actual CLI export | Exact `.mg5` card and complete CLI log, `generation.json` with clean source identity, full generator `tree_state` and its digest, parameter-card hash and measured generation duration. |
| Build | Original generated-source and post-compilation manifests; `compilation.json`/origin compilation record with compiler, flags, duration and build kind; model masses and answer/split-order dimensions. |
| Physical inputs | Exact parameter-card bytes, effective per-backend MadLoop card, fixed momentum text bytes and mass/conservation metadata. Keep threshold, rotation and scale schedules separately with their ordinary-point correspondence. |
| Live worker | Build record and executable hash; complete source-role binding; all linked reducer/scalar-library hashes; `mninjars.mod`, config, native-link and link-before metadata; `Source/make_opts` and `SubProcesses/MadLoop_makefile_definitions`. Run `runtime-bind` before removing disposable builds or replacing libraries. |
| Measurement | Complete protocol/adapter/spec, CPU/order and repetitions, initialization/calibration records, raw per-command numerical rows including all orders/coefficients/accuracy estimates/statuses, timed status histograms, dimensions, logs, failures and timeouts. |
| Actual final profiles | Exact measured worker and measurement/binding reference, cards and points, copied source/runtime hashes, complete raw blocks, perf recording and stacks with their hashes. Collect after timing, before cleanup; historical profiles cannot certify a new archive. |

The ordinary collector already retains most of these. In particular, preserve
the final CLI `generation.json` and matching `compiled-sources.json` under the
report inputs root; otherwise the renderer deliberately leaves generation time
pending instead of reusing an older generator's time. Cargo/config/installation
files need only be retained once per identical installed archive; process-specific
generated and compiled source inventories remain separate.

## Remaining actual evidence

1. Complete the fresh eight-case corrected-original reference collection and
   acquire its source/runtime bindings. Finish final candidate CLI exports and
   representative fresh/persistent DP/QP coverage using the installed archive
   above. The proof must cover each full source-role/runtime signature, loop-
   induced amplitudes and mixed split orders.
2. Run all seven timing targets per core case for at least seven paired,
   alternating repetitions under the frozen policy on the reserved CPU. Publish
   the full eight-case dataset through the existing policy tool. Literal original
   Ninja at `27147b3300f556a75661a886e92bdbe00558ee25` remains the denominator.
   Preserve every failed or unresolved numerical/performance gate.
3. Complete the separate ggH/ggHH 100-point seven-target timing campaign and
   corrected-reference comparisons, the twelve-point threshold grid and actual
   installed-archive precision proof. Their existing 1.05 upper-confidence gates
   remain unchanged. The current private Rust proof can supply its exact-source
   oracle; it does not replace the public installed-archive replay proof.
4. Complete both additional two-to-four stress cases with twenty ordinary,
   four rotation and eight scale-sequence points, all four candidate backends and
   two complete commands per stream. Optional corrected references remain
   separately identified. These two cases add no performance acceptance gate.
   Early build preparation must not invent a provisional core dataset identity;
   final stress assembly waits for the actual published core identity.
5. Assemble the final installed-runtime provenance manifest using the retained
   installation record, Cargo manifest/lock and config, the actual core identity,
   unique candidate generator identity and full measured reducer signature.
   Historical setup ABI-2/pin fields cannot identify the installed archive.
6. Collect the actual final profiles and bind them to final measurements. The
   existing ninety-case MG7 regression spec remains the report input when its
   tested production hashes match the final generation map; its one existing
   GoSam Python-3 bypass remains explicit. The final-source binding addendum
   documents the commit scope without rewriting that earlier test evidence.

The final renderer already recomputes corrected numerical comparisons,
observed discrepancies, raw estimate distributions, actual status counts and
paired timing intervals. Core failures/unresolved intervals, failed ggH/ggHH
gates, missing/failed mandatory stress controls or absent final provenance keep
`complete_campaign_passed` false. Intermediate reports may still be rendered
with their real failed/pending status; they do not complete the assigned goal.

## Commands after the corresponding collection stages

Use new output paths. Bindings and manifests below are placeholders for actual
retained files, never substitute historical earlier-runtime paths.

```sh
python tools/ninjars_benchmark_policy_v2.py runtime-bind \
  --bindings FINAL_ROLE_BINDINGS --correction CORRECTION --output FINAL_RUNTIME_BINDINGS
python tools/ninjars_benchmark_policy_v2.py proof \
  --policy POLICY --bindings FINAL_REPRESENTATIVE_RUNTIME_BINDINGS \
  --output FINAL_REPRESENTATIVE_PROOF
python tools/ninjars_benchmark_policy_v2.py publish \
  --policy POLICY --proof FINAL_REPRESENTATIVE_PROOF \
  --bindings FINAL_TIMING_RUNTIME_BINDINGS FINAL_REFERENCE_RUNTIME_BINDINGS \
  --output NEW_CORE_DATASET_DIRECTORY
```

The installed precision driver is prepared and executed through the existing
`ninjars_precision_v2.py prepare SPEC`, then
`execute PREPARED --cpu CPU --execute-numerical` under the serialized MP handoff.
It must use the complete final measured reducer signature. Supplement assembly:

```sh
python tools/ninjars_supplement_v2.py assemble \
  --report FINAL_SUPPLEMENT_REPORT --expected FINAL_CORE_IDENTITIES \
  --precision-proof FINAL_INSTALLED_PRECISION_RESULT --output NEW_SUPPLEMENT_DATASET
```

After final stress assembly and runtime provenance sealing, render into a new
directory with an explicit retained visible timestamp:

```sh
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
python tools/ninjars_report_v2.py FINAL_CORE_DATASET --compile \
  --supplement FINAL_SUPPLEMENT_DATASET --stress FINAL_STRESS_DATASET \
  --runtime-provenance FINAL_RUNTIME_PROVENANCE \
  --validation-spec docs/benchmarkfolder/policy-v2/final-regressions-v1/validation.json \
  --generation-inputs FINAL_GENERATION_INPUTS --profile-spec FINAL_PROFILE_SPEC \
  --timestamp 'RETAINED_VISIBLE_UTC' --output NEW_FINAL_REPORT_DIRECTORY
```

## Three deferred PDF presentation updates

After the generation/source freeze ends, update the template with targeted
software coverage, preserving the current readers and all existing gates:

- Include ggH/ggHH in the actual timed DP/QP usage table as well as the core
  cases. Their raw usage is already retained by the supplement reader.
- Display ggH/ggHH's median NinjaRS/original-Ninja ratios and 95% intervals
  alongside the existing decisions. The complete values are already in the
  validated supplement snapshot.
- Add the accepted allocation study's 3.15% full-process result and the rejected
  native-kernel screen with their exact retained measurement/source identities.
  Keep the older 9.24% callback result and 3.19% Fortran QP result attributed to
  their distinct timing boundaries/revisions. Do not add their percentages or
  replace the final original-Ninja acceptance measurement with a research result.

Finally compile the actual final snapshot, visually inspect the PDF, check byte
reproduction with its retained timestamp, preserve the earlier top-level report,
and promote the reviewed Typst/PDF/data together. No PR is opened before the
user's separate instruction. The goal remains active until all required gates
and final publication/push work pass.
