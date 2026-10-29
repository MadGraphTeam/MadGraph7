<!-- SPDX-License-Identifier: 0BSD -->

# One-loop QCD on loop_sm, coming from MadLoop

pyAmpliCol ships MG5_aMC 3.7.0's one-loop model `loop_sm` with all its
restriction cards, unmodified (file by file, see
`src/pyamplicol/assets/models/PROVENANCE.toml`; MG5_aMC's licence is in
`licenses/MadGraph5_aMCatNLO.txt`). A one-loop QCD request runs on it out of
the box:

```toml
action = "generate"

[model]
source = "loop_sm"              # the shipped model; a path of that name wins
restriction = "no_b_mass"       # MG5_aMC: import model loop_sm-no_b_mass

[process]
entries = [{ expression = "u u~ > e- e+ g", name = "dy_jet" }]
loop_coupling_orders = ["QCD"]  # MG5_aMC: [virt=QCD]

[loop]
engine = "v2"

[loop.preparation]
fixed_parameters = { WT = "0", WZ = "0", WW = "0", WH = "0" }

[generation]
output = "artifacts/dy_jet"
```

No `[loop.model]` section and no colour setting are needed:

- **Scheme declaration.** A UFO loop model states its counterterms but not
  their scheme. pyAmpliCol ships loop_sm's declaration (QCD corrections,
  complete UV and R2 counterterms, 't Hooft-Veltman, four-dimensional
  external states and gamma5, mass-dependent decoupling, real on-shell
  masses: MadLoop's conventions) for the exact source it was written for: it
  applies to a model whose UFO declarations have the recorded digest, which
  the shipped copy, MG5_aMC's own `models/loop_sm` directory and every
  restriction of either share. An explicit `[loop.model]` always wins. The dry
  run shows the declaration used (`estimated_coverage.model_declaration`).
- **Colour.** A one-loop request without `color.accuracy` is full colour, as
  MadLoop. Leading and next-to-leading colour are opt-in
  (`color.accuracy = "lc"` or `"nlc"`) and need the model's formal-N_c colour
  declaration; tree requests keep the leading-colour default.
- **Source repair.** loop_sm declares its four-G0 vertex `V_1001` with the
  three-slot constant tensor `SSS1`; the same constant with four slots,
  `SSSS1`, is in the model. Model preparation moves such references by
  structure (a constant scalar tensor, the model's unique tensor of the
  vertex's spin slots and the same exact value) and records each repair
  (`pyamplicol_source_repairs` in the published compiled model, and
  `source_compatibility_diagnostics` in the dry run). Any other slot mismatch
  is refused, typed (`lorentz-slot-arity`), when a request reaches the vertex.

`pyamplicol self-test` generates and evaluates a one-loop virtual on the
shipped loop_sm (`u u~ > e- e+ / Z [virt=QCD]` on `loop_sm-no_b_mass`) and
compares it with MadLoop.

## MadLoop process syntax

| MadLoop / MG5_aMC | pyAmpliCol |
| --- | --- |
| `import model loop_sm-no_b_mass` | `[model] source = "loop_sm"`, `restriction = "no_b_mass"` |
| `generate u u~ > d d~ g [virt=QCD]` | `expression = "u u~ > d d~ g"`, `process.loop_coupling_orders = ["QCD"]` |
| `QED=2` (Born order) | `process.max_coupling_orders = { QED = 2 }`; with `coupling_order_policy = "minimal"` the minimal Born orders are found |
| `QCD^2==4` (squared order) | every squared-order sector is a separate partial result: read it out (`result.for_order(...)`) |
| `/ z` | `generation.particle_veto = ["Z"]` (internal lines, loops and counterterms) |
| `p p > t t~` with a multiparticle label | the same expression: one artifact entry per subprocess (`p` and `j` are the model's massless quarks and gluon, so the b is in them on `no_b_mass`), each equal to its own single-process generation; load one with `Runtime.load(artifact, process="g g > t t~")` (the stable ID or the expression; the artifact records loop_sm's own spelling, `G G > t t~`, and any letter case selects it) |
| `$ x`, `$$ x`, decay chains `a > b c, b > d e`, `@N` | not served |
| `[noborn=QCD]` (loop-induced), `t > b w+` (decays) | not served (see the refusals below) |

MG5_aMC syntax inside a process expression is refused with the key
`madloop-syntax`, and the message names the run-card setting above.

## Restriction recipes

| Flavour scheme | Recipe |
| --- | --- |
| 4FS with tops (massive b) | `restriction = "no_widths"` (all widths zero, MB = ymb = 4.7) |
| 5FS (massless b) | `restriction = "no_b_mass"` with `[loop.preparation] fixed_parameters = { WT = "0" }` whenever a top propagates in a tree current |
| 5FS with a b and a t on one fermion line (`u d~ > t b~`, `u b > d t`, `g b > t w-`, `b b~ > w+ w-`) | `restriction = "no_b_mass"` with `fixed_parameters = { WT = "0" }` as above, or the default restriction with `{ MB = "0", ymb = "0", WT = "0" }`: the same parameter values and the same virtual (binary64 to 1e-13) |
| massive charm | `restriction = "c_mass"` |
| non-diagonal CKM | `restriction = "zeromass_ckm"` or `"ckm"` |

loop_sm's other restrictions (`default`, `no_masses`, `no_tau_mass`,
`lepton_masses`) load the same way. A W line that joins a massive and a
massless quark (the top and a massless b at `no_b_mass`, a massive and a
massless quark through the CKM matrix at `zeromass_ckm`, `ckm` and `c_mass`)
generates at one loop: the Born closes at a vertex. Tree-level requests
refuse such a line with `dirac-weyl-closure` (tree lanes install no
Dirac-Weyl closure contraction); keep the light quark massive there, with
its mass fixed to zero in the default restriction. The one-loop engine is
real on-shell: a
coloured particle with a width in a tree current (the top at every shipped
restriction but `no_widths`) is refused with `coloured-width`, whose message
names both recipes; widths come with the complex-mass scheme.

## Refusals

Requests outside the served scope fail with a typed error whose message
starts with a listed key and names the remedy; `pyamplicol generate
--dry-run --json` lists every key (`loop_admission_refusals`,
`process_request_refusals`, `native_export_refusals`). On loop_sm:

| Key | Request |
| --- | --- |
| `decay-process` | a 1 -> n decay (tree and one-loop requests alike) |
| `madloop-syntax` | MG5_aMC syntax in the process expression |
| `loop-induced` | no Born amplitude within the coupling-order limits (`g g > h h`, `g g > z z`, ...) |
| `coloured-width` | a coloured particle with a nonzero width in a tree current |
| `lorentz-slot-arity` | a vertex tensor whose slots do not match its particles (other than the repaired constant scalar ones) |
| `majorana-fermion`, `multi-fermion-contact` | Majorana fermions; vertices with more than two fermions |

See [UFO Model Coverage](ufo-model-coverage.md) for every one-loop refusal and
[Configuration](configuration.md) for the v2 engine's options.
