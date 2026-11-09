The original `strict-policy-qcd-n4-warmup-v2` attempt remains unchanged. Four lanes retain completed 100-point controller observations for `u u~ > z z g g [virt=QCD]` with `loop_sm`:

| Lane | DP | QP | Comparison with baseline CutTools |
| --- | ---: | ---: | --- |
| Baseline CutTools | 88 | 12 | Reference |
| Candidate CutTools | 88 | 12 | Identical retained coefficient values |
| Candidate Ninja | 97 | 3 | Passes `3e-6`; misses ordinary `1e-8` target |
| Candidate COLLIER | 100 | 0 | Passes `3e-6`; misses ordinary `1e-8` target |

The maximum raw relative differences for Ninja and COLLIER are `8.258362793805673e-7` and `8.3018197546771e-7`. The audit recomputes the complete stored comparator, including its explicit small-pole exception; it does not adjudicate those ordinary-target differences. Counts describe final public-call statuses, not internal reduction attempts or retries. Recorded timings are excluded from performance acceptance.

Candidate NinjaRS remains a **failed controller result**, with `BlockingIOError(11, 'Resource temporarily unavailable')`. Its retained raw files contain 200 coefficient rows for 100 points: 96 DP and 4 QP, the latter at points 84, 89, 93 and 94. There is no completed `result.json`, `raw/observation.json`, or post-measurement verification evidence. The failure has no traceback, and the archived controller has no explicit post-observation verification step, so its exact failing operation is unknown. This audit binds and cross-checks the existing [backend-8 forensic](qcd-n4-backend8-resource-failure/README.md), including all 25 evidence hashes, without promoting the failed lane to a completed numerical qualification.

Candidate TIRust times out during initialization before READY. It has no completed observation stream. Its 56 coefficient rows form **28 completed public initialization calls**: bootstrap call 0 on point 1, followed by calls 1 through 27 on points 1 through 27. Thus 27 unique input IDs are seen and point 1 occurs twice. The trace has no incomplete coefficient pair; 6 calls have status `219` (TIRust DP) and 22 have status `323` (TIRust QP). These are unqualified partial initialization data, and completion of the next call is unknown. The warm-up SHA256 is `33968008a3cd21ec02d49735eab04a44ea434c315115b02a09373a289f1eddf2`.

The coordinating launch supplied `--timeout 1800`. Its numeric value is session-reported: the original specification omits the argument and the failure records encode a protocol timeout without a duration. The index does not infer a timeout duration from timestamps.

The index binds every original attempt file, all five retained protocol sources, generation commands, physical points and cards, the historical baseline source identity, the candidate generator source snapshot, the existing backend-8 forensic, and this audit/parser helper. It reconstructs the native worker from the frozen template and checks the recorded driver hash. All lanes reused earlier compiled numerical objects while rebuilding the worker; this is not relabelled as a fresh full numerical rebuild. Executable and runtime-library identities are historical records, not fresh checks of mutable installations. Reduction policies are verified against retained effective-settings metadata; physical card bytes are separately hashed.

Reproduce the source/hash/raw-output audit from the repository root:

```sh
source tools/tirust_env.sh
python3 docs/benchmarkfolder/experiments/numerical/audit_qcd_n4_warmup_v2.py
```

Default mode verifies `qcd-n4-warmup-v2-index.json`; `--write` creates it exclusively. Neither mode starts compilers, workers, postchecks, or native calculations. No original attempt or prior forensic file is modified.
