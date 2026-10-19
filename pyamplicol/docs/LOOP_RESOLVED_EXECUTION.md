# Helicity-resolved virtual evaluation

Source-owned model-rule loop artifacts may carry complete single-helicity
projection kernels. The authenticated loop descriptor records this capability;
the public metadata must agree. Older inclusive artifacts remain loadable and
reject `evaluate_resolved` with an explicit compatibility error.

`Runtime.evaluate_resolved(points, mu_r2=...)` returns Born values with axes
`[point, born squared order, helicity, contracted colour component]` and virtual
values with axes `[point, virtual squared order, helicity, contracted colour
component, epsilon]`. The epsilon axis is finite, single pole, double pole.
Helicity identifiers follow `runtime.physics.helicities`.

The colour axis currently contains one component, `full`: the complete colour
contraction for that helicity. This is not a decomposition into colour flows or
positive probabilities. Each helicity contribution retains the original
incoming spin average, colour average and identical-particle normalization.
Consequently `.total()` sums helicities and the singleton colour axis and
reconstructs the ordinary inclusive result without an additional averaging
factor. All coupling-order axes remain distinct.

Generation compiles the existing Born, family and UV/R2 projection expressions
for one helicity. Evaluation executes the existing currents once in a native
point-by-helicity batch, retains helicities through family reduction, and
applies the complete counterterms to those rows. Exact family invariants and
source parameters remain shared per physical point. There is no Python loop
around single-event or single-helicity evaluation. DoubleFloat and arbitrary
precision retain their native carriers through projection, reduction and the
final sums.

Native row identities combine the caller's physical point identity and the
helicity index with checked integer arithmetic. Public reduction failures name
the original point index and helicity. Empty batches, irregular tails and lane
permutations use the same schedule. Success still means the reducer completed;
automatic accuracy certification and rescue remain separate unfinished work.

This interface does not yet deliver directed spin/colour correlations, LC/NLC
components, flavour mappings or native SDK resolved operations. Its explicit
full-colour axis and preserved helicity axis provide a base for those additions.
