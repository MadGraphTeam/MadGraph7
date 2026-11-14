# Proposed canonical latch regression test

`canonical-regression-proposal1.patch` adds `tests/unit_tests/loop/test_tirust_reduction_latch.py`.
The patch is unapplied. It reuses the hash-pinned, sealed source-proposal1
synthetic renderer and the same eight scenarios, but compiles the live optimized
MadLoop template and makes candidate-only assertions. It neither substitutes
the saved candidate nor treats the original sticky behavior as an expected pass.

The intended test fails on the current unpatched template when a healthy public
call inherits the earlier failure latch. It also requires retention through
healthy groups and internal direction/rotation jumps, automatic DP-to-QP rescue,
terminal-failure recovery, healthy answers, and both NinjaRS wrapper variants.
It is bounded to two compilations and sixteen fresh synthetic processes, with
30/5-second individual and 300-second inclusive limits. No retries are built in.
All build artifacts use /tmp; loader override variables are removed.

Only pure patch applicability and Python syntax/content checks have been run for
this proposal. No additional compiler, stub, reducer or physical-amplitude run
is authorized or claimed here. The earlier sealed stub qualification is separate.
After explicit admission and applying the production fix plus this patch, the
intended focused command is:

```bash
source tools/tirust_env.sh
python3 -B -m pytest -q tests/unit_tests/loop/test_tirust_reduction_latch.py
```
