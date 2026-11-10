This private replay uses the retained synthetic N4/rank5 correlated-error
fixture with the current Fortran adapter, current extracted host rescue code,
and installed bridge `eb3d67fd…`. It does not evaluate a physical amplitude.
The original fixture and all historical results remain unchanged.

`build.py` reuses the existing source generators in a fresh ignored directory,
on CPU14 with a combined 120-second compiler deadline. It neither executes a
reducer nor acquires the numerical license. It records exact compile commands,
generated-source hashes and native artifacts. Numerical replay is a separate
step and must hold the shared numerical lock.

The completed replay retains both the historical default threshold `1e-3` and
the frozen qualification threshold `1e-6`, each with zero/two rotations and
real/imaginary projections. Compare every result and call mode with retained
historical rows. Default-policy limitations remain reported separately from
the qualification policy; no accuracy threshold or host safeguard is changed.
No empirical truth-error guarantee follows from host consistency alone.

All eight synthetic cases completed with the actual current bridge, host and
adapter mappings verified. The public result and accuracy bits match the
historical replay in every case. At `1e-6`, all four projections used QP rescue;
their largest symmetric error against the retained fixture truth is
`5.973e-16`. At the unchanged `1e-3` default, all four inaccurate DP projections
remain accepted, with maximum finite symmetric error `7.029e-4`. These default
limitations remain visible and are separate from the approved qualification
policy. No production threshold was changed.

The child completed and was reaped in 1.495 seconds under the 120-second cap.
This is diagnostic process duration, not an amplitude benchmark. Full physical
initialization, the required physical stress cases and performance acceptance
are not qualified by these synthetic results.

`package.py --live` verifies the portable evidence and the original source and
native-artifact bindings without loading a library. Omit `--live` to verify a
relocated package. Native libraries are recorded by hash only; generated
Fortran, exact build commands, runtime mappings, original references and all
eight results are retained as text. The separate `analyzer-smoke` directory
contained only a historical-data parser check and is deliberately excluded
from the new native evidence.
