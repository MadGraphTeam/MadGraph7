# EW threshold qualification: refusal in both arms

The QP-mass source change did **not** resolve this EW threshold stress point. Both original and candidate returned status **420**, ACC **1**, and a QP software exception latch. Neither arm supplied an accepted QP result or an observed native QP-mass input. The four changed returned buffer words are diagnostics from failed evaluations; they establish no numerical accuracy, improvement, or degradation.

The process is the existing `loop_qcd_qed_sm_Gmu` EW `e+ e- > t t~ a` export. Each fresh worker completed the full 100-point ordinary initialization sweep (101 outer calls including bootstrap, 202 raw rows, 200 steady rows), preserving its own prior ordinary words, status, IEEE and software state. Target 1 is the same ordinary control, accepted DP219 and exact in both arms. Target 2 is the threshold fixture, refused420 in both arms. Overall and sole split-order outputs duplicate physical values; their two rows are not two independent points. The `3e-6` and `1e-8` thresholds apply to accepted pair/control comparisons, not independent absolute accuracy.

Both complete warnings and all raw output are retained. At target 2 the software flags change `[0,0]` to `[0,1]`; observed IEEE vectors remain unchanged. This is a software-latch observation, not evidence of trap freedom, hardware exception behavior, IEEE lowering, or accepted internal QP precision. Caller `P` and `MU_R` words stay exact before/after and across arms, but are not a witness of repaired, canonical, or native reducer inputs.

The five actual READY_V2 snapshots per arm record pole mode 0, forced stability, a source-derived filter bypass, and the complete stored profile: `CHECKPHASE=true`, `HELDOUBLECHECKED=false`, offset10000, 32 selected helicities, and 200 selected loop groups. Those stored INIT flags are **not learned filters**. The internal local bypass variable is not directly observed. The compiled COMMON/source includes and exact source of the pole0 bypass rule are retained.

The earlier original RUN1 failed before READY because its driver required persistent learned flags despite the pole0 bypass. It never requested either target or a candidate worker. Its complete ten-file failed report, old predicate/parser bytes, diagnosis, and the minimal READY_V2 policy correction are retained as separate history. The pure replayer reproduces that predicate rejection from its initial V1 snapshot. The subsequent BUILD2 rebuilt only two drivers and linked two executables, reusing the scientific objects under the existing flags. No repair of the scientific mass source or fresh native execution occurred during retention.

Run the standalone file-only replay from any copy of this folder:

```sh
python3 -B replay.py
python3 -B test_replay.py
python3 -O -B test_replay.py
```

An independently supplied SHA256 of `index.json` can be passed as `--index-sha256`. The verifier authenticates the exact physical inventory, deterministic gzip text, full source namespaces543cf (Python RUN2,24 payloads) and d00a9 (compiled BUILD2,17 payloads), all341 BUILD report files/110 command triples, raw pair reports, 54 canonical LOAD identities per arm,15 providers, strict READY/TARGETS/STOP/BYE closure, PID/start/session/CPU/cleanup joins, source-bound raw parsing, full learning parity, and the conditional comparison that excludes target2. It AST-selects the unchanged collector functions and executes only contained pure parsers under filesystem/process/loader/network/signal/mutation blockers. It does not execute the retained controllers, audit scripts, scientific mains, or toolchain.

`collection.json` maps logical UTF-8 records to compressed members. Source-index namespaces remain exact; historical subsets are explicitly labelled incomplete. `excluded-native-identities.json` retains ELF/object/module/library identities, with zero binary contents. `link-aliases.json` separately records read-only path normalization during retention joined to the already closed LOAD identities; it is not a new build or runtime observation. The complete large linker maps and symbol listings account for most compressed size.

The public ordinary EW capsule `fortran-qp-mass-ew-qualification-1`, index `48a8b22cdb652cb173ebe2714aab7cf8efee07ec5793b1a30ceedfabdec6b04a`, supplies GEN/source/base-BUILD ancestry. This capsule retains that exact manifest, the ordinary component-state pins, required own-arm results, and base-BUILD identities without copying that whole campaign. Replay uses no external reads or project imports and does not purport to re-execute GEN or the base scientific build.

Independent physical truth, true >53-bit physical mass inputs, complex/derived mass qualification, native input coverage, successful QP threshold evaluation, broader process/backend/stress coverage, cache behavior, local runtime acceptance, and production promotion remain open. The implementation remains unapplied/unpromoted. No PDF or benchmark acceptance is updated by this capsule.
