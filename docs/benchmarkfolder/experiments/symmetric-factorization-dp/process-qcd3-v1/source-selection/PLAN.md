# Actual-shape DP factorization: bounded process diagnostic

This is a prepared experiment plan. No output clone, compilation or amplitude evaluation has been performed by this namespace. Production files and the completed kernel/correctness seals remain unchanged.

## Bound source and applicability

Use `.benchmark-work/experiments/symmetric-factorization-production/dev-qcd3-v1/outputs/candidate/qcd-3` as the source. Its existing private QP factorization remains byte-identical in every clone. The retained fresh Ninja6 and NinjaRS8 100-point records are in the sibling production experiment's `dev-qcd3-v1/data` directory.

`inventory-v3.json` binds source, cards, compilation manifest, updater, vertex and reference-record hashes. It finds136 eligible adjacent DP vertex/updater calls: FFV1L1_2 ranks2/3:51/12; FFV1L2_1 ranks2/3:56/17. There are no rank4 calls in this QCD3 output. The actual `LOOPMAXCOEFS=70` supports the present output rank4; the experiment must not invent rank4-input/rank5-output calls requiring126 coefficients. Rank4 helper correctness remains covered by the independent210-coefficient oracle only.

Fresh vertex files differ from the sealed source snapshots in declaration ordering. The ordered executable statements, declaration multisets, all160 coefficient expressions and128 derivative-row identities match. The initial strict whole-file-hash rejection and a subsequent orientation-specific inventory bug are retained as `inventory.stderr` and `inventory-v2.stderr`; neither indicates an amplitude failure. The successful inventory verifies both loop-routing orientations explicitly.

## Isolated transformation

Create new `original`, `capture-only`, and `factor-trace` output copies below `.benchmark-work/experiments/symmetric-factorization-dp/process-qcd3-v1`, with retained evidence below the matching DP experiment namespace. Relocate internal symlinks and verify all parent source hashes before mutation.

Preserve each original public updater and every QP routine/call. Append private DP full helpers generated from the frozen factor.py, using the actual module constants and existing compact public fallbacks. Only the136 proved immediately adjacent `FFV1L1_2`/`FFV1L2_1` → rank2/3 updater calls may change. Assert distinct input/output wavefunction storage, dimensions4 and allowed cut size. Record every replacement and source diff. No terminal-DP path, rank1 transformation, A-zero pruning or unrelated vertex is introduced.

The coefficient batch currently has no IEEE import. Add a local `IEEE_EXCEPTIONS` import and local entry-INEXACT variable in both `capture-only` and `factor-trace`, then query immediately before each selected updater call. In `capture-only`, call the unchanged original updater. This extra control isolates the new batch IEEE procedure boundary from changed arithmetic; it must reproduce the original100-point output/status stream before any factor result is interpreted. The private helper keeps its independently tested nearest-rounding, halting-mode, finite-range and stored-row-relation guards. Fresh sticky-INEXACT is an admission condition, never forced true by the process driver. Broader trap-mode behavior of a future production batch boundary would still require dedicated integration controls before promotion.

Instrument only `factor-trace`, with per-helper total/fast/fallback counters and, if practical without changing arithmetic, fallback-reason counters. Reset counts at the start of the retained evaluation pass, after warm-up. Verify `total=fast+fallback` and complete reporting for every selected helper. Preserve an uninstrumented generated-source variant separately, but do not time it in this correctness stage.

## Fixed numerical checks

Keep the physical parameter card and the original100 point bytes unchanged. Configure all clones for backend6, normal `CTModeRun=-1`, `MLStabThres=1e-9` and `COLLIERRequiredAccuracy=1e-12`, matching the bound fresh-dev policy-v2 settings exactly. Preserve all other settings and retain the actual cards. Run one complete normal pass after ordinary initialization for original, capture-only and factor-trace, retaining all Born/Laurent rows, accuracy estimates, statuses, dimensions and raw return counters. Confirm the original clone reproduces its bound fresh-dev backend6 record.

Use the existing policy-v2 comparison: `abs(a-b) <= 1e-7*max(abs(a),abs(b)) + 1e-10*S`, where S is the common maximum absolute Born/Laurent component at that point across the compared streams. Compare every split-order row and all four components; require complete point/order dimensions and valid stable own-backend DP/QP statuses. Record every changed coefficient bit, accuracy estimate and rescue decision. In particular, the two common original QP points are a hypothesis to observe, not a target to enforce. Reduced rescue count alone cannot establish correctness; all output comparisons remain mandatory. Any added failure/rescue or out-of-bound component remains a retained failure requiring investigation, without threshold changes.

After Ninja6 controls pass and the sole Symbolica slot is explicitly handed over, repeat the original/capture-only/factor-trace checks for backend8, using the unchanged installed bridge and runtime. Compare against the bound fresh-dev backend8 stream and cross-check the two new backend results under the same common norm. Keep all existing compensated-DP retry and own-QP logic unchanged.

## Cost and interpretation

Preparation reuses an existing CLI export and installed libraries; only copied coefficient batches, polynomial object and driver need compilation. Expect roughly2–4 minutes of local build work and a few minutes per three-clone correctness cohort, depending on initialization and actual QP rescues. No Symbolica is used by the Ninja6 cohort; the backend8 cohort waits for explicit MP-slot ownership. Use an idle diagnostic CPU apart from reference207 and other assigned work. These estimates are scheduling guidance, not calibrated sample counts.

This stage is a changed-association correctness/applicability/stability diagnostic, without a performance claim. A later uninstrumented fixed paired timing series requires a separate declared protocol and quiet-window coordination. The completed source-derived kernel results remain mixed, including two losses; they do not justify production DP promotion on their own.
