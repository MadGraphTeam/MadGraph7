# Historical scheduling estimate

The available historical observations imply about **17.0 hours of known main-matrix work**, plus the unmeasured heavy lanes and all candidate rescue-chain costs. This is a planning subtotal based on nonquiet historical timings and explicit proxies, not a lower bound or a forecast for the optimized candidate. The three controls add about 10.8 minutes before their candidate-chain costs. No new numerical measurement was run to produce this estimate.

| Case | Known subtotal (minutes) | Missing explicit lanes | Baseline chain proxy missing |
|---|---:|---|---|
| `qcd_n1_uux_z` | 2.95 | None | No |
| `qcd_n2_gg_ttx` | 3.29 | None | No |
| `qcd_n3_uux_zgg` | 18.36 | None | No |
| `qcd_n4_uux_zzgg` | 663.16 | backend-8: baseline, candidate; backend-9: candidate | Yes |
| `ew_n1_udx_wp` | 3.03 | None | No |
| `ew_n2_epem_mupmum` | 3.23 | None | No |
| `ew_n3_epem_ttxa` | 25.38 | None | No |
| `ew_n4_epem_mupmumaa` | 300.83 | backend-9: candidate | No |
| `mixed_uux_ga` | 4.12 | None | No |
| `li_gg_h` | 2.99 | None | No |
| `li_gg_hh` | 3.65 | None | No |

Every case still lacks a measured final candidate-chain cost in this calculation. Baseline standalone backends 6, 7 and 8 use the historical candidate implementation as a cost proxy; the final candidate likewise uses its historical predecessor as a proxy. The original default chain is represented by its first backend only. Standalone TIRust costs are not valid bounds on a TIRust-first rescue chain, since the latter tries the remaining DP backends before QP.

Each lane includes its recorded initialization plus warmup interval, an initial complete traversal of all 100 saved points, one sustained calibration batch, and seven full-point batches. At the recorded rate, the batch traversal count is `max(1, ceil(2 seconds / traversal_seconds))`. This assumes unchanged throughput during calibration; the actual controller recalibrates using its sustained observation. For expensive cases one traversal already exceeds the two-second sizing target. All 100 points remain in every traversal.

The QCD four-final-state backend-1 pair alone contributes about 5.38 hours. Its available backend-1/6/7 pairs plus the baseline half of the backend-9 comparison contribute about 11.05 hours; backend 8, candidate backend 9 and both rescue-chain costs are missing. The EW four-final-state known subtotal contributes about 5.01 hours and still lacks candidate backend 9 and its rescue-chain cost.

Compilation, external queues, Python preflight and startup/SETPARA, stress qualification, failures/retries, and unknown lanes are excluded. Historical warmup timings exclude startup and SETPARA. Per-point rescue frequencies may change with the final implementation. The incomplete EW4 warmup and failed QCD4 backend-8 request were not extrapolated to a completed traversal.

The JSON retains every consumed observation/raw file hash, the exact configuration hash and estimator hash. Its scope and missing values are explicit. Reproduce into a fresh append-only label from the repository root:

```sh
source tools/tirust_env.sh
python3 docs/benchmarkfolder/experiments/estimate_campaign_schedule.py --label another-planning-snapshot
```

Retained `estimate.json` SHA256: `365fc8e04fe53810762a2c58c785de52b5a6e85385a8e06683977b8690f7eea0`.

This second snapshot binds each default rescue chain to the generation record embedded in the already hashed retained baseline inputs, rather than an ignored working export. The independently audited cost formula and all numeric totals are unchanged. The first snapshot and its original estimator bytes remain preserved.

Execution order for the final campaign is numerical validation, then all eight main TIRust-first chain comparisons, then explicit-backend regression checks and controls only if the main gates pass. A failed main gate remains reported and defers the costly later phase. This ordering changes neither the seven-pair/full-point protocol nor the per-process criterion.
