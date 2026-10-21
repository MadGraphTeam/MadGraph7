# Actual direct-QP results

Both actual generated-Fortran streams passed. At all ten points in each
stream, NinjaRS returned120 and accuracy -1, matching the public direct-QP
mode's semantics. The timed status histograms also contain120 for all ten
evaluations. The raw numerical rows, histogram, dimensions, driver patch,
physical and MadLoop cards, complete compiled manifests and runtime
descriptors remain retained.

The unchanged comparison routine found zero failures:

| Comparison | Coefficients | Maximum fraction of tolerance |
|---|---:|---:|
| QP retained contexts versus passed DP | 80 | 0.000328832 |
| QP finalized contexts versus passed DP | 80 | 0.000328832 |
| QP retained versus finalized contexts | 80 | 0 |

The two QP context policies have exactly equal coefficients. Both compiled
test drivers differ from their passed ordinary counterparts by only the
recorded public force-stability switch. Their DP/MP state assignments and
fixed top-mass/scale schedules are unchanged. The complete reduction runtime
remains `8b454519cb6818ef1fafe37378930c276f7e85394a2333ebc9fdb00e75726cff`.

Preparation and execution exited0; their controller identities are absent
and all owned children exited normally. CPU190 was released to root after
the final stream. These direct-mode outputs demonstrate actual QP state
rebinding and context reset. They provide no automated stability estimate,
automatic-rescue acceptance result or performance claim.
