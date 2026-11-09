# EW n4 ordinary-point target triage

This read-only check reparsed the five completed streams in
`../../experiments/numerical/strict-policy-ew-n4-warmup-v2/`: baseline CutTools
and candidate CutTools, Ninja, COLLIER and NinjaRS. All 100 saved points and
all output columns matched their retained raw observations exactly. The
directory name refers to the revised warmup driver; these results are ordinary
post-warmup observations, not the later optional warmup-only extraction.
TIRust was still running and is not included in this partial triage.

All available comparisons pass the `3e-6` regression criterion. The stricter
ordinary target uses the same symmetric criterion at `1e-8`; its union of
missed point IDs is 14, 38, 69, 81, 85 and 87. The table reports ordinary
relative differences `abs(a-b)/max(abs(a),abs(b))`, not symmetric errors.

| Points | Observed grouping | Largest relevant relative difference |
|---|---|---:|
| 14, 81, 85, 87 | Ninja, COLLIER and NinjaRS finite coefficients agree closely apart from CutTools | `2.28734e-7` at 81 |
| 69 | CutTools and COLLIER cluster; Ninja and NinjaRS form a different cluster | `3.15360e-7`, COLLIER/NinjaRS simple pole |
| 38 | COLLIER double pole differs from CutTools/Ninja/NinjaRS | `2.25320e-8`, CutTools/COLLIER |

Baseline and candidate CutTools rows are identical. At point 81, CutTools
reports accuracy `6.24867e-7`, while Ninja/COLLIER/NinjaRS report estimates
between approximately `9e-12` and `7e-11`. At point 69 the situation changes:
Ninja/NinjaRS estimates are `2.09821e-7` and `9.14003e-8`, versus approximately
`2e-9` for CutTools/COLLIER. These are estimator values, not independent error
bounds. Agreement between implementations is also not a proof of truth.

If a later bounded higher-precision adjudication is needed, original points
81, 69 and 38 cover the three patterns. Such diagnostics must preserve the
same physical inputs and verify reduction FPE flags as well as finite output;
unchecked fixed-QP status alone cannot establish a complete tensor sum.
No new native calculations, compiler work, threshold changes, or timing
measurements were performed for this triage.

The common point SHA-256 is
`dd1bc3aeed4017110f3dfea62715eb83be3b9d72e46d464b39fd4057ac445c62`;
the physical card SHA-256 is
`891e4b89d522804e1f21c3f8b19b8cc2209b29cbfb7809189a5e7062c1363ea8`.
The completed per-stream `result.json` hashes are:

```text
baseline-1  6d66ccd59b650e8a7db851168350230829993337690644e8c7726f05cb473faa
candidate-1 ed89b69e9e34ef81837ca9f36bacce5a4e8e33dfbbd96bb6fb16aa5c96225252
candidate-6 6398919e69cee3d2aa0423df78075b2c6e9276fd51616ebbecb6047e47495676
candidate-7 8278184d6855221c1fd0e0cebd7348b534129369316eb3ba7871f3bd82ccbf73
candidate-8 af614c841b6e6027e44642a2eb8c631e76d986dc69da16f4bac6e0a26e5bae58
```
