# Explicit source colour input

These examples use the supplied MadLoop 3.7.0 `models/loop_sm` source, model R2
rules and UV counterterms in its declared HV convention. The JSON is explicit
provider-style model input for the actual ordinary, UV, R2 and external-field
entries used by the Drell–Yan, four-gluon and stable W plus gluon QCD
controls. It does **not** declare
a universal SM continuation. Other processes work only when every selected rule
and field has declared coverage and passes the compiler's actual capability checks.
Neither the service nor this declaration selects implementations by process name.

The data keeps CA=Nc, CF=(Nc²−1)/(2Nc), TF=1/2 and fixed Nf. The supplied four-gluon
R2 coefficients are specific to Nc=3: two whole-group recipes replace them using
the checked general-Nc HV tensors. Each recipe proves complete equality to its
actual source tensor at Nc=3 and retains the source loop species and diagnostics.
The mandatory `lhv=1` tie is proved through the existing exact source owner.
Unrelated fixed-Nc effective rules are deliberately absent from this input.

The charged-current addition keeps the actual `d~ u W-` tree and R2 source
entries selected by `u d~ > W+ g`, plus the selected u/d two-point and gluon
UV/R2 entries. Their elementary fundamental `Identity`/`T` tensors and explicit
Ncol, CA and TF dependence use the same declared continuation. Loop-particle
groups, mass-dependent logarithms and field keys are unchanged. The separately
selected `u~ d W+` tree/R2 entries preserve their own conjugated CKM coupling and
loop group for crossed `d W+ > u g` and outgoing `d u~ > W- g`. Their coverage
comes from actual request inventories, not inferred charge conjugation. This is
QCD/model-rule colour data, not an EW-correction or massive CMS certificate.

From the checkout, with the normal native extension installed or staged:

```sh
python examples/loop/source_colour/prepare_model.py \
  FOR_REFERENCE_ONLY/MG5_aMC_v3_7_0/models/loop_sm \
  workspace/models/loop-sm-colour.pyAmplicol-model.json
pyamplicol generate --card examples/loop/source_colour/drell_yan_lc.toml --dry-run --json
pyamplicol generate --card examples/loop/source_colour/gluon_scattering_lc.toml
```

The parallel `gluon_scattering_nlc.toml` card requests NLC.
Adjust the card paths to your working directory. `prepare_model.py` applies the
existing checked four-G0 scalar-arity compatibility view, attaches the explicit
loop and colour declarations, and writes one ordinary compiled-model bundle.
There is no runtime sidecar to keep in sync.

A dry-run prepares model data and graph/order catalogues; it explicitly leaves
numerator, reducer, process-wide formal-colour coverage and LC canonical-component
checks deferred. Missing declarations and conflicting scheme values fail clearly.

```python
from pyamplicol.api import Runtime

runtime = Runtime.load("workspace/artifacts/gg-nlc")
point = [[500, 0, 0, 500], [500, 0, 0, -500],
         [500, 300, 0, 400], [500, -300, 0, -400]]
result = runtime.evaluate([point], mu_r2="1000000")
resolved = runtime.evaluate_resolved([point], mu_r2="1000000")
assert result.color_accuracy == "nlc"
print(result.finite, result.single_pole, result.double_pole)
print(resolved.total().values)  # preserves squared-order and Laurent axes
```

NLC resolved output has one contracted `nlc` component. LC has signed canonical
Born-tensor components, which can be overcomplete; they are not positive tree
flows or sampling probabilities. Both use the common sector-leading degree before
partitioning the Born rows.

**Dependency status:** open-fundamental LC canonicalization requires the corrected
typed-port implementation in FeynKit PR #155. Its isolated Community build passes
the strict port/dummy-renaming controls; the ordinary installed dependency is not
yet assumed to contain that change. An affected request fails explicitly rather
than merging tensors with altered physical ports. Unaffected tensor sets and NLC
remain subject to their actual checks. Fresh public artifact publication with the
unified ordinary package is the remaining integration check for these examples.
