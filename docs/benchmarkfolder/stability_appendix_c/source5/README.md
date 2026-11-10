# Raw-DP stability pilot source5

Source4's [closed failure](../failed-run-4/README.md) retained all 20 requests
without native starts. Its prelaunch check incorrectly expected the physical
parameter card in the subprocess directory. The compiled Fortran reader
searches that filename first, then `./Cards/param_card.dat` with zero through
five parent prefixes. The private export's `Cards` file is the fourth attempt.

Source5 implements that exact search order. It binds the complete compiled
`lha_read.f` and `rw_para.f` sources and rendered driver, stops at the first
existing match, and requires a private regular file with the frozen physical
card hash. It rejects an earlier wrong or external shadow instead of skipping
it. No production reader or physical input is changed.

The numerical run and the actual dry proof use the same `prelaunch_worker`
and `worker_postguard` functions. Before sealing, all 20 fresh private worker
copies must pass actual card writing/readback, exact physical-card resolution,
absent diagnostic outputs, executable and saved-point hashes, and all
read-only postguards. The proof blocks process and native-library launches
and confines writes to its separately named ignored namespace. It never
creates the actual work5 directory or obtains the numerical lock.

The unchanged passed source3 drivers, runtime closures and 400 cut-qualified
points are inherited without builds, loader queries or resampling. The 12
nonlocking and eight locking requests, 6,000 forced-DP probes, scientific
worker, consistency formula, controls, precision policy and caps are fixed.
Source4/work4 and every earlier attempt remain unchanged.

Only root-admitted stages may create actual work5 receipts and native workers:

```
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/stability_appendix_c/source5/pilot.py inherit --execute
```

After root verifies that inherited checkpoint:

```
python3 -B docs/benchmarkfolder/stability_appendix_c/source5/pilot.py run --execute
```

These diagnostics measure unchecked forced-DP internal consistency. They do
not independently establish true error or precise performance. Effective
card-selected backends and unavailable points remain explicitly reported.

Before sealing, all 20 actual worker copies passed the shared prelaunch and
postguard under the launch/write audit guard. The [closed proof](actual-prelaunch-proof.json)
records the fourth search candidate for every process/backend, zero forbidden
attempts and an absent actual work5 namespace. All 128 pure/mock/source tests
passed. Two earlier proof-instrumentation failures are preserved separately
and summarized in [dry-proof-history.json](dry-proof-history.json); they made
no numerical calls and did not change the controller.
