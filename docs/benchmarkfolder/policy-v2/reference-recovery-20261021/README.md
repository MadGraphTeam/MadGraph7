# Restart of the interrupted corrected-original reference campaign

The user resumed the integration campaign on 21 October 2026. The earlier
collector and its log monitor no longer exist. Its only remaining numerical
process is stopped, reparented to PID 1, and has no peers holding its command,
diagnostic-output, or protocol pipes. The old report remains incomplete.

The read-only `pretermination-audit.json` records the exact PID/start ticks,
executable and pipe identities, and hashes every old report file. It independently
checks all 18 calibrations and 36 timed records from the six completed cases.
These records remain historical helper correctness evidence; they do not form a
complete eight-case reference series. Existing inherited output files in the
unfinished QCD4 working directory are not new measurements.

Root authorized terminating only this verified orphan, preserving the complete
old attempt, and starting one fresh full eight-case corrected-original campaign.
The new campaign uses the unchanged collector, unchanged frozen worker protocol,
all 100 points, original cards and seeds, CPU 207, the same numerical thresholds
and poles, backends 1/6/7, and the existing 86,400-second per-phase timeout.
QCD4 and EW4 run first; all eight cases are collected into one complete report.
There is no composite adapter or edited historical status. The explicit case
order changes scheduling only, with no per-case numerical or timing changes.

Source inspection explains why QCD4 may take hours: it has 144 helicity
configurations and 745 loop groups (1,378 loop diagrams), while the unchanged
worker requires 100 consecutive post-initialization evaluations before READY.
CutTools can invoke genuine quadruple-precision rescue under the fixed 1e-9
stability threshold. Logs do not reveal the orphan's precise iteration, so no
claim is made about progress within its 2h19m CPU time.

`actions.py` verifies PID start ticks, command, executable, working directory and
process-group isolation before terminating the stopped orphan. SIGKILL avoids
briefly resuming an orphan that has lost its controller. The process may remain
as an exited zombie under PID 1; that is not a live numerical worker. The script
also records the exact frozen-source hashes, argv, PID and start ticks for the
new detached collector. It refuses to overwrite an existing launch or report.

The new report is helper correctness evidence, with two diagnostic repetitions
per backend. Final performance still requires fresh committed candidate runtime
proofs and the complete seven-target, seven-or-more alternating repetition
campaign against literal original MadLoop with Ninja. This restart cannot certify
any NinjaRS performance or numerical gate on its own.
