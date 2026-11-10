# Attempt 5: alternate-card numeric spelling

Attempt 4's card-only smoke completed the nominal read with the exact
91.188 GeV binary64 mass and no reducer context. The alternate read failed
because the generated reader stores values in `CHARACTER*20`: its fixture
token `9.2000000000000000e+01` became an invalid truncated exponent. The
attempt returned 2, did not reach restoration, and ran no physical worker.
Its inputs, partial output and failure evidence remain unchanged.

This amendment replaces that one token with `9.200000e+01`, exactly the same
number 92. All other card bytes are preserved. Every numeric token in both
physical cards and both ident cards is checked against the actual reader's
20-character width before freezing and before every new invocation. The
new fixture, single-token patch and validation inventory are retained.

The already-built attempt-4 smoke/plain/observed executables, driver source,
objects, compiler flags, library closure, runtime and all expectations are
unchanged. There is no compilation or link stage. The wrapper uses the exact
frozen attempt-4 runner, admission checks, output guards, process cleanup,
CPU 10 and common numerical lock, with fresh attempt-5 input/output paths.
The prior build metadata is copied byte-for-byte and still names the prior
unchanged executables. Both original source freezes and the new fixture
freeze are checked on every entry and after execution.

Run `attempt5/resume.py freeze` only after source review. Root authorization
is required for `smoke` (30 seconds) and `run` (four sequential 180-second
workers), using `PYTHONDONTWRITEBYTECODE=1 python3 -B` from the repository
root with `tools/tirust_env.sh` loaded. A failure stops without an automatic
retry. The smoke must again prove all three actual reads, complete records,
absent contexts and zero reducer exports before physical workers can start.
There is no timing acceptance claim.
