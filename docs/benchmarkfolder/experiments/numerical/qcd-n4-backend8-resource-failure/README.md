# QCD 2→4 backend-8 resource-failure forensic check

The original `strict-policy-qcd-n4-warmup-v2/candidate-8` request remains **failed**, with `BlockingIOError(11, 'Resource temporarily unavailable')`. This adjacent audit preserves that status and provides no formal numerical or performance qualification.

Reparsing the unchanged raw files with their retained worker parser yields 200 coefficient rows for 100 points: 96 accepted NinjaRS DP results and four QP results, at points 84, 89, 93 and 94. The CutTools reference reparses identically to its saved observation and result. The coefficient comparison passes the existing `3e-6` criterion, with maximum raw relative error `8.258432567588928e-7`; it fails the tighter `1e-8` target.

The raw clocks record 635.912028739 seconds for initialization and warm-up and 675.223177665 seconds for one traversal. They are retained data, not paired performance evidence. Backend counts describe final accepted results and do not enumerate internal attempts or retries.

No completed post-measurement verification is established. The failed request has no `result.json`, `raw/observation.json`, or `inputs-after.json`. Its archived numerical controller did not contain an explicit post-observation verification step, and the failure has no traceback, so the exact failing operation cannot be inferred. This audit compares retained provenance records and frozen inputs only; it does not run postchecks, native calculations, or compilers, and does not access the active backend-9 folder.

`forensic.json` binds the original failure, raw files, input records, frozen points, and retained protocol sources by SHA-256. `forensic.py` reproduces those read-only checks from the repository root and creates the JSON exclusively, without replacing existing evidence. Shared-host notes elsewhere in the experiment remain contextual; no causal claim is made for individual timings or coefficient differences.
