# Integer-classification component qualification, revision 2

This repeats the exact revision-1 driver corpus and integer-only analyzer with
the revised adapter's integer exponent classification. The driver template, DP
and QP drivers, shared kinematic helpers, include and analyzer are byte-identical
to revision 1. The wrapper changes derive directly from candidate177a7973; the
negative controls remove respectively its finite guard and equality guard.

The earlier failure is retained at ../../fortran-fixtures/qualification-1/; no
result or source from that attempt is changed. The same two compiler settings,
four rounding modes, entry flags, sizes, input cases,48trap comparisons and two
negative controls remain mandatory. Signaling-NaN exceptions are not waived.
Matching trapped termination establishes no post-call numerical equivalence.

The pinned controller differs from revision 1 only in repository/work paths and
the version label. It uses the same180-second whole-corpus deadline, CPU12
nonexclusively, compiler/toolchain/runtime bindings and owned-child cleanup. It
links recording-only components; no reducer or amplitude is executed.

Default `python3 -I -B run.py` verifies the frozen plan without executing it.
The explicit --execute command uses a fresh append-only ignored attempt. Full
adapter transport, physical amplitudes and performance require later checks.
