# Held corrected-original core collection

`command.sh` is the exact prepared command for all eight core processes, all
100 retained points per process, and CutTools/Ninja/Collier. **It has not been
launched.** Wait for the root agent to release the coordinated quiet window.

The configuration is a byte-identical retained copy of `../cases.json`:
`MLStabThres=1e-9`, `COLLIERRequiredAccuracy=1e-12`, unchanged physical inputs.
The frozen reference collector uses two persistent commands, CPU 207, and the
`original_helper_corrected` role. These reference timings are diagnostic only;
the performance denominator remains the separate untouched original Ninja run.

The ordinary collector repeats the clean MODEL/subprocess preparation before
collection. The earlier eight build-only checks are retained in
`../reference-preparation`. Repeating compilation avoids modifying the frozen
collector or adding redundant fresh-process physics evaluations. Fresh-process
parity is covered separately by complete representative streams for each exact
runtime signature group.

`launch.json` binds the command, configuration, correction, collector and policy
sources, point streams, expected output namespaces, and the explicit held state.
It records preparation rather than authorization. Use a new plan/run namespace
if any bound input changes; retain this plan unchanged.
