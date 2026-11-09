# Historical two-final-state chain diagnostics

Both predeclared seven-pair, 100-point probes completed with numerical agreement, using the frozen historical candidate and libraries. Every timed amplitude was accepted by baseline NinjaRS DP or candidate TIRust DP. These probes were explicitly ineligible for formal acceptance before execution; their intervals do not qualify the final candidate.

| Case | Candidate/baseline geometric mean | Descriptive 95% interval | Baseline median per point | Candidate median per point |
|---|---:|---:|---:|---:|
| `qcd_n2_gg_ttx` | 1.02164 | [0.98703, 1.05746] | 4.67809 ms | 4.83454 ms |
| `ew_n2_epem_mupmum` | 1.01790 | [0.97366, 1.06416] | 4.31790 ms | 4.28189 ms |

The actual timed chains were `8|6|7|1` and `9|8|6|7|1`; both used CPU 9 and the same strict stability policy. Each role contributed 3,500 timed amplitude evaluations across seven batches, each containing all 100 saved physical points. Counts describe final accepted backend/precision, not every internal attempt. Initialization plus warmup is retained separately.

The surrounding shared-host incident produced very high load and a resource-unavailable failure in the independent QCD4 numerical preparation. The exact stage of that QCD4 error is unknown; its forensic audit does not prove that it occurred during process creation. These probes retain their own per-batch host snapshots plus the parent-reported environment observations in the two adjacent host-context JSON files. High load alone does not establish CPU contention; no causal attribution of a particular timing to a host mechanism is made. Both already-launched probes had completed by the safe administrative hold check, so their results remain unchanged. No new native job was launched after the hold.

Earlier CPU-2 preparations remain unlaunched and preserved under their original labels. Before any measurement there, known installation compiler affinity prompted fresh CPU-9 labels; the recorded scheduling notes preserve that decision. These are not selected best-of-rerun timings.

`n2-diagnostics-index.json` binds the complete original files, summary hashes, numeric comparisons, provenance and compact host ranges. The original per-repetition records remain authoritative.
