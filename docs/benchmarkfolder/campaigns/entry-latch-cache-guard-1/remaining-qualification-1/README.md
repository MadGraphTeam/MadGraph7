# Nine-case numerical continuation

This additive supervisor finishes the remaining nine cases of
`entry-latch-cache-guard-1`, using the exact reviewed n1 adapter, existing CLI,
baseline/build verifiers, raw numerical replay and bounded cleanup. It changes
no physics, source, card, point, worker, model or installed library. The closed
n1 packet and every predecessor failure remain unchanged. No timing is run.

Run from the repository root after `source tools/tirust_env.sh`:

```sh
python3 -B docs/benchmarkfolder/campaigns/entry-latch-cache-guard-1/remaining-qualification-1/run.py
python3 -B docs/benchmarkfolder/campaigns/entry-latch-cache-guard-1/remaining-qualification-1/run.py --preflight
```

Both invocations are source-only. Only after explicit root admission:

```sh
python3 -B docs/benchmarkfolder/campaigns/entry-latch-cache-guard-1/remaining-qualification-1/run.py --execute
```

Cases run in this fixed order: `li_gg_h`, `qcd_n2_gg_ttx`,
`ew_n2_epem_mupmum`, `mixed_uux_ga`, `li_gg_hh`, `qcd_n3_uux_zgg`,
`ew_n3_epem_ttxa`, `qcd_n4_uux_zzgg`, `ew_n4_epem_mupmumaa`.
Each completes private baseline reuse, fresh candidate generation, both builds
and numerical replay before the next case starts. Baseline numerical objects
retain their exact replay proof; candidate objects are fully built from source.
Configuration changes, source/runtime/card drift, numerical failure or uncertain
cleanup stop the continuation. There is no automatic retry or resume.

| Stage | n ≤ 2 | n = 3 | n = 4 |
|---|---:|---:|---:|
| Baseline reuse | 600 s | 600 s | 600 s |
| Candidate generation | 900 s | 1800 s | 3600 s |
| Both builds | 1200 s | 3600 s | 7200 s |
| Six numerical lanes | 7200 s | 7200 s | 28800 s |

Every stage includes a 60-second cleanup reserve. Numerical stage limits include
mutex admission and pre/postchecks. They do not replace the registered worker
limits: 3600 seconds for n ≤ 3 and 14400 seconds for n = 4. Checkpoint publication
has a separate 120-second administrative bound, with no numerical work. Tiny
failure/status publication after deadline is administrative and cannot qualify
an incomplete case.

Builds use nonexclusive CPUs 12–15/jobs4. Numerical workers use nonexclusive
CPU12, serializing backend8/9 through the existing common lock. No CPU is
reserved. `PYTHONHASHSEED=0`, `set --no_save`, full finite terms and poles, strict
stability1e-6, original saved100points and physical cards are unchanged. The
per-case original materialized default chain is frozen from predecessor
generation, including the control-specific chain. Each numerical lane selects
exactly baseline1 or candidate1/6/7/8/9; no chain is substituted.

The existing warmup-only reader requires the full final100 steady point window,
all coefficient/order columns and3e-6 agreement, retaining ordinary1e-8 misses.
Final accepted DP/QP status counts are recorded; internal retries and FPE flags
remain unobserved. Initialization durations are not accepted timing measurements.

Each completed case gets an immutable `CASE/checkpoint.json` binding its stage
logs/status, complete raw observations, build/reuse inputs, cards, points and
source seal. The root `status.json` may then advance without changing a sealed
case. Failed or interrupted cases retain their original files but receive no
passing checkpoint. Existing n1 success plus this continuation requires separate
review before any full-matrix claim; performance is outside this qualification.

The only adapters to the frozen n1 supervisor are the current single-case list,
private per-case output directory, continuation source seal and declared-card
checker using each exact inherited default chain. Stage execution, signals,
observed-descendant/PID-start cleanup, provenance and numerical replay are reused
unchanged. No historical `main()` is called.
