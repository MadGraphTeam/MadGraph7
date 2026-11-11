# Standalone caller-factory and environment-probe qualification proposal

SOURCE/MOCK ONLY. No compiler, loader, tensor library, amplitude, native probe or candidate worker has been executed by this packet. Root admission is required for each proposed native stage. The unchanged caller module, generated include and C probe come from sealed aba903450f49e0e18ba3fff40dbf6074e1cd67e9af1c67cd83bbd6fd583c1047. `producer_kind_ok` is public in that exact Fortran module; its explicit result joins route_matches and the passed environment contract before build/finish.

The fixed corpus contains all 26 routes × modes 1/2/4/5 (104 cases), plus 70 controls: boost scaling, ±zero, smallest/largest subnormals, normal boundary, NaN/Inf/sNaN integer encodings, imaginary components, wrong format/upper DP bits, route/mode/WE/PAIRING/shape mismatches, ENV=false and widened-DP QP producer rejection. These are synthetic words, not captured phase-space events. Eleven additional synthetic integer tokens test the exact environment acceptance masks. The pure Python reference supplies every chord error exponent, pair class, availability flag and difference error exponent for exact compiled comparison.

`driver.f90` reads the actual MXCSR/x87 control token before and after the whole corpus, requires current-host readiness, and checks unchanged controls around each case. It reads IEEE flags before and after each factory operation without setting or clearing any flag/mode. All external and shift payloads are read as INTEGER(16) hexadecimal; no signaling NaN is converted to floating point. The C probe contains only stmxcsr/fnstcw reads. This validates standalone factory transport and present-host readiness if it passes; it cannot prove the future full amplitude's entry-to-use environment history, genuine QP producer, source-routing plumbing, compiler equivalence, physical invariant correctness, coefficient agreement or performance.

Proposed commands (repository cwd, only after admission):

```
source tools/tirust_env.sh
python3 -B .tirust-build/caller-bound-native-proposal-1/control.py build
python3 -B .tirust-build/caller-bound-native-proposal-1/control.py run
```

Build: four serial commands, 60 seconds aggregate, 4 GiB address-space ceiling, no tensor libraries. Run: one standalone worker, 10 seconds aggregate, 4 GiB. Neither needs the scientific reducer lock or a reserved CPU. Results are fresh one-shot directories `.tirust-build/caller-bound-native-qualification-1/{build,run}`. The reviewed owned-process helper is copied byte-for-byte from qcd-topology-rescue-higher-precision-runner2/supervise.py; SIGINT/TERM/HUP feed its deferred-spawn/group-kill/reap cleanup. Final hashing and adjudication must finish inside the stage budget. No automatic retry is provided. Compiler and utility environments are minimal and omit every LD_* override and inherited build flags; the actual source paths/tool hashes/argv/exit/accountability time/executable hash are retained. Runtime mappings are not claimed by this standalone harness.

Pure mocks: eight tests pass. They replay exact corpus generation, verify the 104-case route/mode matrix, and reject lost/extra/duplicate rows, changed route/pair/bound fields, token errors, IEEE changes and changed controls. Three early test logs are retained: two tuple-versus-JSON-list equality failures (the second correction command lacked activated Python, so the unchanged test reran) and one negative-control search-string mistake. Neither changed scientific inputs, reference values, caller code or native outcomes. The final test makes each mutation's existence mandatory.

The separate a51a685a conditional trace calculation shows that the unchanged factory admits the four known CMS raw24 interventions under its assumptions. That evidence does not substitute for this compiled factory check or the separately reviewed full amplitude candidate.
