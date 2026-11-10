# Current-source report refresh: source-only proposal

Select `invariant-symmetry-1` for the next report only after all eleven numerical
cases close and receive an immutable checkpoint. Keep exact old timings in their
original campaign. Precise performance acceptance remains Hua-Sheng's local TODO.
No active numerical packet was read for this proposal; no canonical file changed.

## Current binding contract

The canonical 19-page PDF is the `split-dp-terminal-1` report built in commit
`584dae7875d20e881a4f9dc27bdb67f5e0fa5512`; its SHA-256 is
`02428f3a7659b33ab55ea116288963f1d9b470ea1bc3a7d2877d1f60fc38387f`.
`report-build.json` binds 4,294 input paths, Typst0.15.1, font environment,
creation timestamp, campaign ID, and the three Typst inputs: summary, manifest,
context. `proposal.json` pins the committed source and diagnostic records read.

`build_report.py:188` selects `campaigns/<id>/{summary,cases,manifest}.json`,
checks policy/source/native identities and retained ancestor failures, then
validates an optional explicit dossier under that same campaign.
`campaign_report.py:221` re-collects that dossier through its frozen file allowlist
and requires exact derived equality. `benchmark.typ:1–13` reads the selected
summary/manifest/context; the context points to the dossier. Formal timing tables
come from the selected summary; untimed numerical rows come from the independently
reparsed dossier. These are different datasets. A passing formal headline still
requires seven pairs, all eight main-process runtime gates and four existing-backend
regression gates; untimed success cannot manufacture that headline.

`campaign_report_refresh.selection()` admits only explicitly indexed closed
100-point, six-lane numerical checkpoints. `next_campaign()` is deliberately
registration-only and rejects measured results: remove that old “next campaign”
selection when the new campaign becomes the report's selected identity.
`campaign_report_provisional.py:90–112` requires its selected report's exact
campaign/source/native identity. Do not relax this guard to attach old timings
to the new campaign.

## Minimal update after closure

1. Archive the existing PDF, Typst sources, renderer/helper sources, context,
   closed selection/dossier, canonical README and build ledger under a new
   `history/report-before-invariant-symmetry-qualification/`. Follow the existing
   compact archive pattern: pin original commit584dae, verify copied bytes against
   Git blobs, and retain the complete ledger so all 4,294 input bytes can be
   reconstructed from that commit. Do not render the old PDF using new helpers.
2. Add a new current-campaign closed checkpoint and immutable selection/dossier.
   Replay every current raw lane with the existing numerical collector: 11 cases,
   66 lanes, 100 points per lane, original cards/streams, source/native and compiler
   bindings, actual accepted DP/QP counts, failures and 1e-8 target misses. Keep
   unavailable numerical entries and formal timing entries pending; no estimates.
3. Prefer the smallest historical-timing treatment: link the archived19-page
   report and keep the existing ancestor-failure panel. Leave `closed_provisional`
   absent from the new selection. If full old timing tables are desired, add a
   separately named historical panel validated against its pinned old manifest
   and old dossier; never insert it into new `timings`, `paired`, ratios or gates.
4. Add four explicit package roles to the existing implementation supplement:
   rank6 Feynman parameters, adaptive fixed-QP replay, independent backend8 audit,
   and adaptive masked/trap replay. Bind full package indices and derive bounded
   claims from their sealed summaries/raw replay results. Route these new adaptive
   roles before the current generic `role.startswith('adaptive_')` Fortran-parser
   branch. Preserve old diagnostics as historical findings and describe the new
   adjudication additively; do not rewrite old failed summaries.
5. Label the hard-coded old QCDn4 target-miss and QCDn3 triage notes explicitly
   `split-dp-terminal-1`. Current misses must come only from the new raw numerical
   dataset. Update the first-page pending/new-source text and the local handoff
   to handoff3. Keep style.typ, page geometry, navy headings and timing palette.
   Nineteen pages is the current archive size, not a requirement to squeeze or
   omit new failures.
6. Keep `delivery_complete=false`, stress limitations and local performance
   deferral explicit. Build with the existing `--campaign invariant-symmetry-1
   --campaign-report campaigns/invariant-symmetry-1/report-snapshots/<new>.json`
   arguments, then use the same arguments with `--verify-only`. Do this only after
   report input closure and rendering admission; this proposal executes neither.

The higher-rank Fortran and statistics proof source subsets still match the new
manifest. The existing lifecycle validator correctly rejects the historical
full-source proof at precisely two changed files (listed in proposal.json).
A fresh current-source lifecycle replay is preferable; otherwise show that
criterion pending and retain the old proof only against its old manifest. Do not
weaken exact-source validation or infer a new lifecycle pass from numerical rows.

## New diagnostic claims and limits

| Closed evidence | Defensible new text | Required limit |
|---|---|---|
| `rank6-feynman-parameter-qualification-1`, indexc2fe806d… | Independent direct integration corroborates seven sampled finite components; max observed TIRust residual4.762e-17. Five of six disputed real/imaginary parts miss3e-6 for uncached COLLIER, consistent with its retained flag−1. | Not all210 components or a physical amplitude. Gaussian analytic-error-only bound gives max3.558e-7; finite arithmetic/node error has empirical checks, no whole-program certificate. No COLLIER cache-bug attribution. |
| `adaptive-qp-physical-replay-qualification-1`, indexa728fbeb… | Private replay serves400 calls, recovers37 former refusals, preserves363 served outputs bitwise, removes74 W03 lines. | FixedCT4 status120 unchecked; no per-call512-bit claim or independent truth. Original post-native analyzer failure remains retained. |
| `adaptive-qp-physical-replay-independent-8-audit-1`, index5b9100fc… | Independent backend8 final sums agree within2.685e-16 at ordinary and2.029e-16 at threshold point. | At threshold all200 native scales differ because backend8 squares before promotion. Same physical inputs, not identical native input records; no truth/stability promotion. |
| `adaptive-qp-checked-trap-proposal-1/qualification-1`, indexd60f909d… | Masked adaptive threshold output passes independent automatic checks, status323;600 QP calls served,489 prior served outputs bitwise unchanged. | Both old/adaptive trapped workers SIGFPE during DP warmup2; combined trap gate false. Fault localized to shared DP Cayley division, exact hardware exception remains inferred. Physical truth, explicit rotation identity, per-call tier and runtime remain unqualified. |

These are separate synthetic/private candidate identities; none replaces the
ordinary campaign's installed eb3d bridge or closes general physical stress.
Their worker/quadrature wall durations are administrative diagnostic timings.

Focused follow-up tests should reject old-provisional/new-campaign joins,
changed package pins, unclosed lanes, promotion of fixed120/W03 output, false
trap passes, whole-rank claims from seven samples, and lifecycle source mismatch.
Existing report input/PDF/context drift tests remain required. Offline report
collection must not execute quadrature, reducers, builds or numerical workers.
