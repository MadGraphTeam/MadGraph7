# Paused at the user’s request

User requested a pause on 2026-10-20 at approximately 23:55 UTC.
Do not resume work or jobs until the user explicitly resumes this task.

The exact checkpoint is `references/pauses/20261020T235607Z/checkpoint.json`.
All five recorded jobs are verified stopped: allocation controller 2896495,
allocation worker 3369596, reference collector 1973783, legacy reference worker
1995662, and upstream watcher 2507588. Check PID start ticks and command/source
identities from the ledgers before sending any signal; controllers have vanished
across earlier pauses, so never assume a PID still belongs to this task.

Twelve complete allocation-study QCD3 pairs and one original arm are historical
completed data. The allocation arm of zero-indexed pair 12 was stopped during
its timed RUN and MUST NOT count as an accepted sample. No EW2 sample ran.
The 21+7-pair performance gate is incomplete. Preserve the interrupted study;
restart affected performance measurements in a new declared namespace as needed.
Never resume a pause-spanning timing interval as accepted evidence.

The original reference worker is stopped before READY after about 38m50s CPU.
Its collector and all partial reports remain intact. Initialization time spanning
this pause is diagnostic only. Confirm pipes, deadlines, source/runtime bindings
and all surviving identities before deciding how to resume its correctness work.

The native study has no live job. Its 18 ordinary tests, complete three-arm
replay/convergence and 12 supplemental state controls passed. The original test
executable was restored byte for byte; seven-triplet screening has not started.
Its large evidence package remains unsealed locally. Complete compression/sealing
only after resumption; no optimization has been promoted from either study.

The installer transaction fix and seven new tests are UNTESTED drafts. The stress
report extension and four new tests are also untested. The new stress adapter and
its tests are UNPARSED and UNTESTED; first fix optional-helper failure propagation
into per-case status, finalize the reader API and run syntax/unit checks. Earlier
36-report/55-combined-test seals stay historical, unchanged. The 90-case broad
MG7 campaign passed 89 with one existing GoSam Python 3 skip; the final source
must still match its bindings.

The extra two-to-four plan is approved: gg > tt~ gg QCD and uu~ > e+e- mu+mu-
EW, twenty fixed points each plus rotations/scales. Existing core and ggH/ggHH
performance gates remain unchanged; these two added stress cases add no new
performance gate. Actual generation/evaluation remains held for final source and
installed-runtime freeze. No process-specific implementation is authorized.

On resume, finish the two prepared optimization decisions and installer hardening,
then freeze numerical source, add the proposed panic regression, run final Rust
and MG7 tests, install and validate actual standalone/NLO copied exports, and run
the complete core/loop-induced/stability/stress campaign. See
`references/FINAL_FREEZE.md`. Finish publication cleanup and the reproducible,
visually inspected benchmark.pdf. Do not mark the goal achieved unless every
required gate passes; do not open a PR without the user’s instruction.

Publication audit remains unsealed. Its exact runtime-asset inventory and portable
ignore-rule check have NOT run. Preserve required metadata, historical failures,
private source proofs and preexisting `vendor/StdHEP/fail`; do not blanket-delete
build roots. All implementation remains on branch `madloop_ninjars`.
