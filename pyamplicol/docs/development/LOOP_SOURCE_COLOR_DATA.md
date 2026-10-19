# Source colour data in model bundles

`LoopColorDeclaration` in `pyamplicol.models.loop_color_declaration` carries an
explicit SU(Nc) continuation in the existing raw-model metadata key
`pyamplicol_loop_color`. `with_loop_color_declaration` accepts either a raw UFO
snapshot or a compiled model. The ordinary model JSON writer persists it; the
compiled copy drops its cached serialization. Other source metadata, exact
expressions and model capabilities remain intact.

This is model input, not a certificate inferred from a model name. In particular,
the presence of parameters called `CA`, `CF` or `Ncol` does not admit every
effective tensor. A model provider must declare the actual continued entries.
The public generation service still requires its separate admission integration.

## Data contract

The declaration contains:

- The full `LoopSchemeDeclaration`, which must equal the effective loop scheme.
- A formal variable name such as `Nc`, distinct from actual source parameters.
- Exact UFO-syntax parameter identities, for example `CA: "Nc"` and
  `CF: "(Nc**2-1)/(2*Nc)"`.
- Concrete kernel keys `(collection, vertex_id, colour_index, lorentz_index,
  loop_group_index)` and field keys `(particle_id, correction, loop_group_index)`.
- Required scheme parameter values as exact decimal strings or complex pairs.
- Optional `WholeRuleColorRecipe` records. The current recipe is
  `qcd-four-gluon-r2-hv`, identified by the original collection, vertex, loop group
  and coupling-order vector. Its gauge coupling and gauge/fundamental-Dirac
  contribution are explicit inputs.

No tensors, alternative species lists or numerical evaluator states are copied
into this declaration. Actual source groups own their particles, multiplicities,
couplings and diagnostics. A recipe never supplies its own inferred flavour sum.

## Preparing an owned source view

The internal preparation sequence is:

```python
data = resolve_loop_color_declaration(raw)
fixed = source_color_fixed_parameters(
    raw, data, fixed_parameters=explicit_fixed, parameters=explicit_parameters,
)
prepared = prepare_loop_request_model(raw, fixed_parameters=fixed, **order_bounds)
prepared, colour = prepare_declared_source_color(
    prepared, data, scheme=effective_loop_declaration.scheme,
)
program = prepared.compile(
    incoming, outgoing, selected_corrections=corrections,
    source_color=colour, color_accuracy="lc", **numerical_options,
)
```

The fixed-map helper compares exact values before merging. A conflicting user
parameter or generation constant fails; the existing source owner subsequently
rejects conflicting runtime updates atomically. Internal scheme values are proved
from their formulas rather than overwritten. An unresolved active diagnostic in
a required scheme constant prevents preparation.

The resolver parses formulas with the existing exact UFO frontend and resolves
source references with the same scalar owner. It verifies every supplied
identity symbolically at Nc=3 over the retained external inputs. Referring to a
parameter's own original source value preserves that parameter's active
diagnostic. A newly introduced diagnostic-bearing reference to another parameter
currently fails explicitly because its paired continuation has not been defined.
Unknown symbols also fail before numerical compilation.

Every applied whole-rule recipe invokes the existing complete Nc=3 tensor check,
including source species/group and HV constraints. It is then bound to the
request-owned source entries, never objects retained from a previous model load.
Recipes outside the admitted order scope remain unused. Unknown groups, partial
groups or conflicting pre-existing counterterm replacements fail. Selected
ordinary/UV/R2/field coverage and ordinary scalar Nc-independence are still
checked by the process compiler.

A full-colour request without this declaration retains its original preparation
path. This data boundary adds neither a model-name dispatch nor a new numerical
engine, artifact manifest, coupling-selection policy or multiplicity limit.
