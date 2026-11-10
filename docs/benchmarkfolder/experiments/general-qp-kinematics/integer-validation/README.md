# Private integer lookup serialization check

This check preserves the frozen classifier index `97f5100a...`. It copies the
exact QCD lookup bytes and derives expected classes and signed witnesses from
the independently retained source-routing oracle. It exercises every QCD pair
in CT4/5, aliases CT1/2, both pair orientations, all actual N=2 groups, invalid
groups/modes/sizes/indices, and a separately rendered empty N=1 boundary with
unknown and absent group rows. N=1 is a serialization boundary, not a physical
process qualification. The oracle does not call the classifier under test.

The Fortran driver and lookups contain only integer/logical operations and
integer/character I/O; no amplitude, reducer, model or floating-point arithmetic
is linked or called. The compiler's usual language runtime is required. All
checks compile with `-O0 -g -fcheck=all -fbacktrace`.

Source preparation (no compiler or executable):

```sh
source tools/tirust_env.sh
PYTHONDONTWRITEBYTECODE=1 python -B .tirust-build/general-kinematics-integer-check/prepare.py
```

After separate execution authorization, one CPU-13 job runs the exact compiler
argv recorded in `run.py`, then the integer executable. A combined 120-second
deadline kills the active process group. Inputs are hash-checked before/after,
logs survive failures, and an existing execution record prevents automatic retry.

```sh
source tools/tirust_env.sh
PYTHONDONTWRITEBYTECODE=1 python -B .tirust-build/general-kinematics-integer-check/run.py
```

This test can validate compiled serialization and indexing only. It makes no
numerical, shell-policy, Born-consistency or performance claim.
