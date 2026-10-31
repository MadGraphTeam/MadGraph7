---
title: "UFO Model Coverage"
nav_order: 2
parent: "Configuration"
---
<!-- SPDX-License-Identifier: 0BSD -->
# UFO Model Coverage

This page is the exact reference of what a generic UFO (or serialized JSON)
model may contain for pyAmpliCol, what is accepted only under a condition, and
what is rejected or ignored. Every row was derived from the code of
pyAmpliCol 1.0.0 with ufo-model-loader 1.0.0 and Symbolica 3.0.0; the paper
summarises it in one table. Read
[Models and Processes](models-and-processes.md) first for the workflow, and
use `pyamplicol model inspect <source>` to see the issues reported for your
own model.

Status words used below:

| Status | Meaning |
| --- | --- |
| supported | accepted without further condition |
| conditional | accepted when the stated condition holds, otherwise the stated diagnostic |
| experimental | accepted, but the path is not validated against a reference |
| rejected | refused with the stated diagnostic |
| ignored | loads without error but never affects generated amplitudes |

## Summary

| Area | Supported | Rejected or restricted |
| --- | --- | --- |
| Input | UFO directories (trusted Python), serialized JSON, compiled IR files, prepared kernel bundles | anything else; ufo-model-loader older than 1.0.0 |
| Fields | scalars, Dirac fermions, vectors, massless spin-2 (validated); massive spin-2 (experimental) | Majorana and fermion-number-violating fermions, spin 3/2, higher spins; ghosts and Goldstones only internally |
| Colour | one SU(3) with 1, 3, 3̄, 8; external coloured legs are triplet fermions or octet vectors | sextets, other groups, N_c other than 3, d^abc, epsilon and sextet tensors |
| Lorentz | Identity, Gamma, Gamma5, ProjM, ProjP, Sigma, Metric, P, Epsilon (four-index Levi-Civita; validated on the loop_sm axial R2 vertices); PSlash in propagators | C, IdentityL, form factors, functions outside the registry |
| Three-point vertices | any spins with colour 1, delta, T^a, f^abc | d^abc; one- and two-point terms are ignored |
| Four-point vertices | colour-singlet (any Lorentz structure); the two-f gauge contact; HEFT Hggg | other coloured contacts (four-fermion operators, delta couplings, d^abc) |
| Five-point and higher | colour-singlet contacts of any valence (tested to ten scalars); HEFT Hgggg | every other coloured contact |
| Propagators | standard scalar, Dirac, vector and spin-2 kernels with fixed widths; custom UFO propagators that lower exactly | complex-mass scheme; gauge choices other than Feynman (massless) and unitary (massive) |
| Parameters | external and internal, real or complex, the fixed function registry, restriction cards, runtime cards for external parameters | custom function bodies, epsilon-expanded values (finite part kept), counterterm vertices (dropped) |
| Processes | 2 to N tree level, LC, NLC and full colour, FFT contraction in the trace or adjoint basis | loop-induced channels (omitted) |

## Preflight diagnostic codes

`pyamplicol model inspect` lists these codes; any error code makes the model
unsupported for generation, warnings do not.

| Code | Severity | Trigger |
| --- | --- | --- |
| `unsupported-spin` | error | a particle spin code outside -1, 1, 2, 3, 5 |
| `unsupported-color-representation` | error | a particle colour code outside 1, 3, -3, 8 |
| `majorana-fermion` | error | a spin-1/2 particle whose name equals its antiname |
| `form-factors` | error | the model declares any form factor |
| `function-arity` | error | a declared function with a registry name but a different argument count |
| `unknown-functions` | error | a function name outside the registry and the UFO tensor heads in any parameter, coupling, propagator, Lorentz or colour expression |
| `unsupported-contact-color-lowering` | error | a vertex with more than three legs whose colour structure could not be decomposed into contact kernels (reported after tensor lowering) |
| `experimental-massive-spin-2` | warning | a spin-2 particle whose mass is not `ZERO` |

Some rejections happen at model compilation rather than at preflight and
carry a `ValueError` message instead of a code; they are marked below.

## Sources and loading

| Feature | Status | Condition and consequence |
| --- | --- | --- |
| Source kinds | supported | `built-in-sm`, `built-in-sm-heft`, a UFO directory containing `__init__.py`, a `.json` model in ufo-model-loader's schema, a compiled IR file `*.pyAmplicol-model.json`, or a prepared bundle `*.pyamplicol-model`. A directory without `__init__.py`, a missing path or unreadable JSON is a `ValueError`. |
| Built-in selector spelling | conditional | Use `built-in-sm`; the `builtin_sm` spelling is accepted by the config resolver but not by the typed model layer. |
| UFO Python execution | supported | A UFO directory is imported as Python with your privileges; only load trusted models and prefer the JSON form for automation. During the import the `UFO_SCALARS_MODEL_*` and `UFO_GRAVITY_MODEL_*` environment variables are hidden and no `.pyc` files are written. |
| ufo-model-loader version | conditional | External UFO and JSON models require an installed release 1.0.0 or newer (development builds are refused with a `RuntimeError`); built-in models do not need it. |
| Required UFO attributes | conditional | `all_orders`, `all_parameters`, `all_particles`, `all_lorentz`, `all_couplings` and `all_vertices` must exist; `all_propagators`, `all_functions`, `all_form_factors` and `all_CTparameters` are optional; objects of class `CTVertex` are dropped silently. |
| Serialized JSON schema | conditional | The file must end in `.json` and provide `name`, `restriction`, `orders`, `parameters`, `particles`, `propagators`, `lorentz_structures`, `couplings` and `vertex_rules`. |
| Loader errors | conditional | Errors raised inside ufo-model-loader (missing restriction file, external parameter without default, malformed JSON) surface as raw Python tracebacks, not as pyAmpliCol `ModelError` messages. |
| Restriction selector | supported | `default` applies `restrict_default.dat` (UFO) or `restrict_default.json` (JSON) when present, otherwise the model defaults without zero-pruning; `none` applies no card; any other name needs `restrict_<name>.dat` or `.json`. |
| Restriction file location | conditional | A restriction given as a path must be named `restrict_<name>.dat` (UFO, inside the model directory) or `restrict_<name>.json` (JSON, beside the model file); other names or locations are a `ValueError`. |
| What a restriction changes | supported | A card sets external parameter values only; `.dat` cards are SLHA parameter cards, `.json` cards map names to `[re, im]`. With `simplify=true` (default) parameters set to zero become constants and the couplings and vertices they kill are pruned. |
| Restrictions on built-in and precompiled sources | rejected | `restriction` and `simplify=false` apply only to UFO and JSON sources; built-in, compiled and prepared models refuse them. |
| Compilation cache | conditional | Compiled models are cached under `$PYAMPLICOL_CACHE_DIR/models` (or the platform cache) keyed by source contents, options and toolchain. Editing a `restrict_*.json` beside a JSON model does not change the key: pass `--no-model-cache` or the file path after editing it. |
| Dry run | conditional | `--dry-run` never compiles a UFO or JSON model; compile it or run `model inspect` first so the cache is populated. |
| Compiled IR files | conditional | Tied to the exact pyAmpliCol, Symbolica and ufo-model-loader versions that wrote them; regenerate after any upgrade. |
| Prepared bundles | conditional | Verified member by member (sizes and SHA-256) on every load; must match the current bundle schema and model-compiler version. |
| Parameter card | supported | `CompiledModel.write_parameter_card()` writes the runtime-adjustable external parameters after the restriction as `{name: [re, im]}`; parameters fixed to zero by the restriction are absent and immutable. |
| Model CLI | supported | `pyamplicol model inspect` loads even unsupported models and lists every issue; `model compile` and `model processes` refuse unsupported models. There is no separate preflight subcommand. |

## Particles

| Feature | Status | Condition and consequence |
| --- | --- | --- |
| Spin codes | supported | -1 (ghost), 1 (scalar), 2 (Dirac fermion), 3 (vector), 5 (spin-2); any other code is `unsupported-spin`. |
| Spin 3/2 and higher | rejected | `unsupported-spin`; no downstream path exists. |
| Scalars | supported | standard propagator i/(p^2 - m^2 + i m Gamma). |
| Dirac fermions | supported | particle and antiparticle must be distinct; standard massive or massless Dirac propagator. |
| Massless fermions | conditional | Treated as two-component Weyl spinors when the mass is literally `ZERO` or an internal parameter equal to zero and the propagator is the loader default; an external mass parameter that merely defaults to zero keeps four-component currents. |
| Majorana fermions | rejected | `majorana-fermion` when name equals antiname; fermion-number-violating flow is not implemented. |
| Vectors | supported | massless vectors get helicities -1, +1 and Feynman gauge; massive vectors get -1, 0, +1 and unitary gauge, decided from the restricted default mass at compile time. |
| Massless spin-2 | supported | 16-component tensor, helicities -2 and +2 only, de Donder propagator; validated by the packaged scalar-gravity model. A model without its own spin-2 propagator must define the `dim` parameter used by the loader default. |
| Massive spin-2 | experimental | accepted with `experimental-massive-spin-2`; the Fierz-Pauli propagator and five helicities are implemented but only the propagator projector is unit-tested; such a model must declare its own propagator because the loader has no default. |
| Ghosts | ignored | any particle with a nonzero ghost number loads, but it can never be external, is excluded from every current, and has no propagator kernel. |
| Spin -1 without ghost number | rejected | classified as a non-external auxiliary whose propagator has no contract; model preparation fails when it is needed. |
| Colour codes | supported | 1, 3, -3, 8; anything else is `unsupported-color-representation`. Colour always means SU(3) with N_c = 3. |
| Coloured external states | conditional | an external coloured leg must be a triplet or antitriplet fermion or an octet vector; coloured scalars, octet fermions and coloured tensors are refused when a process names them (`ValueError`). |
| External-state eligibility | supported | a particle can be named in a process only if it has positive spin, zero ghost number, is not a Goldstone, is propagating and is not a compiler auxiliary; names resolve exactly, then case-insensitively when unique. |
| Particle identity | supported | antiname must exist and be involutive, non-self-conjugate pairs use opposite-signed PDG codes, names and PDG codes are unique, self-conjugate particles are neutral. |
| Mass | conditional | `ZERO` or a model parameter; the massless or massive class (helicities, gauge, Weyl projection) is fixed once at compile time from the restricted default and cannot be changed by a runtime card. |
| Complex masses and widths | rejected | mass and width values must be real; one-loop requests take the complex-mass scheme with real mass and width inputs (`[loop.model.scheme] mass_scheme = "complex-mass"`, see the configuration guide). |
| Width | supported | `ZERO` or a parameter; enters only the fixed-width denominator of massive propagators and can be overridden at runtime. |
| Goldstone bosons | conditional | must be propagating scalars; matched to the unique massive vector with identical colour, quantum numbers and mass and then absorbed into its unitary-gauge propagator; an ambiguous match is an error, an unmatched Goldstone stays an internal scalar. |
| Propagating flag | supported | non-propagating particles cannot be external but are compiled normally. |
| Electric charge | supported | recorded as a bookkeeping quantum number only; not used to check charge conservation. |
| Other attributes | ignored | `texname`, `antitexname`, `line`, `counterterm`, `LeptonNumber` and `Y` never affect amplitudes. |
| Custom propagator attribute | conditional | a particle whose propagator differs from the loader's Feynman default is used as written and loses the Weyl projection and Goldstone absorption. |

## Parameters, couplings and functions

| Feature | Status | Condition and consequence |
| --- | --- | --- |
| Parameter nature | supported | `external` (runtime input) or `internal` (substituted symbolically); the common `interal` typo is corrected; anything else fails in the loader. |
| Parameter type | supported | `real` or `complex`; complex external parameters are set as `[re, im]`; a real parameter can never receive an imaginary part. |
| External defaults | conditional | every external parameter needs a value in the UFO or in the selected restriction card. |
| Expression syntax | conditional | standard UFO spellings only: `cmath.sqrt`, `cmath.pi` or `pi`, `cmath.sin/cos/asin/acos`, `complex(x, y)`, `complexconjugate`, `cond`, `Theta`, `reglog`, bare `exp`, `log`, `tan`, `atan`, and `**`; other `cmath.`, `math.` or `numpy.` calls fail to load. |
| Scientific notation | rejected | a decimal mantissa with an exponent such as `1.5e-3` is mis-parsed by the loader; write `0.0015`, `15e-4` or a fraction. |
| Rational exponents | supported | `x**0.5` is evaluated exactly as `x^(1/2)`. |
| Parameter dependencies | supported | internal parameters may reference each other in any order; cycles are rejected. |
| Function registry | conditional | one-argument `Theta`, `abs`, `acos`, `acosh`, `acsc`, `asec`, `asin`, `asinh`, `atan`, `atanh`, `complexconjugate`, `conj`, `cos`, `cosh`, `csc`, `exp`, `im`, `log`, `log10`, `re`, `reglog`, `reglogm`, `reglogp`, `sec`, `sin`, `sinh`, `sqrt`, `tan`, `tanh`; two-argument `complex`, `pow`; three-argument `cond`, `if`. Anything else is `unknown-functions`. |
| Evaluable subset | conditional | `re`, `im`, `log10`, `pow`, `reglogp` and `reglogm` pass preflight but have no runtime evaluator and fail when a parameter that uses them is evaluated; all other registry functions compute. |
| Declared functions | conditional | a declared `Function` is accepted only if it is a registry name with the registry arity; its body is ignored in favour of the built-in meaning (`function-arity` otherwise). |
| Constants | conditional | `pi` stays exact until numerical evaluation and `complex(0, 1)` is the exact imaginary unit; a bare `I` symbol is not supported. |
| Epsilon-expanded values | ignored | dictionary-valued parameters and couplings keep only the order-0 entry, with one loader warning. |
| Counterterms and decays | ignored | counterterm parameters are ordinary parameters, `CTVertex` objects are dropped, the decays block is never read. |
| Coupling records | supported | any expression in the parameters, any number of coupling-order names per coupling. |
| Coupling matrix | supported | one row per colour structure and one column per Lorentz structure; `None` entries are skipped; each cell becomes its own term and pieces with different orders are counted separately. |
| Coupling orders | conditional | `hierarchy` weights the minimal coupling-order policy; `expansion_order` is stored but has no effect; undeclared order names default to weight 1. |
| Coupling-order policies | supported | `minimal` (default) keeps the lowest hierarchy-weighted order; `explicit` applies only `--max-coupling-order NAME=N`, and misspelled names are not detected. |
| Runtime parameter cards | supported | flat JSON objects of finite real or `[re, im]` values via `--model-parameters`, `Runtime.load(model_parameters=...)` or `set_model_parameters`; an invalid batch leaves the runtime unchanged. |
| Runtime mutability | conditional | only surviving external parameters are mutable; derived parameters and couplings are recomputed and read-only. In recurrence mode an external parameter that the process never uses may be rejected as not used. |
| Masses and widths at runtime | conditional | external mass and width parameters can be varied, but a particle cannot switch between the massless and massive class without regenerating. |
| Numerical precision | supported | model constants are kept exact in the compiled model; only user-supplied values are binary64. |

## Lorentz structures and vertices

| Feature | Status | Condition and consequence |
| --- | --- | --- |
| Normalized tensor heads | supported | `Identity(i, j)`, `Gamma(mu, i, j)`, `Gamma5(i, j)`, `ProjM(i, j)`, `ProjP(i, j)`, `Sigma(mu, nu, i, j)`, `Metric(mu, nu)`, `P(mu, k)` with the standard UFO argument conventions; `PSlash(i, j)` only in its two-argument propagator form. |
| Levi-Civita tensor | conditional | `Epsilon(mu, nu, rho, sigma)` in Lorentz structures lowers to the dense four-index Levi-Civita tensor with ALOHA's components (contravariant, epsilon^{0123} = -1), in every vertex of every model. Validated: the tree kernels of loop_sm's axial R2 vertices `R2_GGZ` and `R2_GGGVa` equal compiled ALOHA routines exactly, and the R2 and bare virtual of u u~ > e- e+ g g [QCD] with `particle_veto = [d, s, b]` equal MadLoop to 1e-12. Ordinary tree vertices with `Epsilon` (e.g. a pseudoscalar coupling to two gluons) take the same lowering but have no reference check: experimental. |
| Other tensor heads | rejected | `EpsilonBar`, `C`, `IdentityL`, three-argument `PSlash` and any accepted head with a wrong arity fail model compilation with `ValueError: ... contains unsupported UFO tensors`. |
| Form factors | rejected | `form-factors`. |
| Sigma normalisation | experimental | expanded as (i/2)(gamma^mu gamma^nu - gamma^nu gamma^mu); vertices using `Sigma` have not been validated against a reference implementation. |
| Chiral structures | supported | `ProjM`, `ProjP` and `Gamma5` are validated through the packaged Standard Model. |
| Index conventions | conditional | 1-based leg indices, the `1000*component + leg` notation for spin-2 fields and negative dummy indices are supported; inconsistent indices are a `ValueError`. |
| Several structures per vertex | supported | any number of colour and Lorentz structures through the coupling matrix. |
| Three-point vertices | supported | any combination of scalars, Dirac fermions, vectors and spin-2 fields, momentum-dependent or not, with colour 1, delta, T^a or f^abc. |
| One- and two-point terms | ignored | tadpole and two-point mixing vertices are dropped without a warning. |
| Four-point colour-singlet contacts | supported | momentum-independent ones through the exact component-basis split; momentum-dependent ones through balanced contact trees. |
| Four-point two-f gauge contacts | conditional | accepted when written as a numeric factor times `f(a, b, x) * f(x, c, d)` with one shared adjoint index and no momentum dependence; otherwise `unsupported-contact-color-lowering`. |
| Other coloured four-point contacts | rejected | four-fermion operators, delta couplings to singlets, `d^abc` and single-f shapes outside the HEFT family give `unsupported-contact-color-lowering`. |
| Higher-point colour-singlet contacts | supported | any valence, momentum-dependent or not, tested up to ten scalars and five-point spin-2 vertices. |
| Scalar-HEFT contacts | conditional | four-point (8, 8, 8, 1) with colour exactly `f(i, j, k)` and five-point (8, 8, 8, 8, 1) with colour exactly `f(...) * f(...)` sharing one index, without a numeric prefactor in the colour string. |
| Other coloured contacts with five or more legs | rejected | `unsupported-contact-color-lowering`. |
| Four-fermion contacts | experimental | not gated at compile time but not validated; treat as unsupported. |
| Counterterm and loop vertices | ignored | only tree vertices are used; `type` and `loop_particles` attributes are never read. |
| Ghost vertices | ignored | compiled but never used. |
| Custom propagator tensors | conditional | the vertex tensor heads plus `PSlash(i, j)`, `P(1)` and `P(1)^2` for the propagating momentum. |

## Colour structures and colour accuracy

| Feature | Status | Condition and consequence |
| --- | --- | --- |
| Colour group | rejected | one SU(3) with N_c = 3 only; a second group or SU(N) with N other than 3 is not representable. |
| `1` | conditional | accepted for any valence when every leg is colour-neutral; on a vertex with coloured legs it is rejected. |
| `Identity(i, j)` | conditional | both indices must point at coloured legs of dual (or equal adjoint) representations. |
| `T(a, i, j)` | supported | adjoint, fundamental, antifundamental index order. |
| `f(a, b, c)` | supported | three adjoint legs or dummies; a product of two `f` is the only coloured four-point structure. |
| `d(a, b, c)` | rejected | recognised, then refused at compilation as an unsupported trilinear tensor or contact lowering. |
| `Epsilon`, `EpsilonBar`, `K6`, `K6Bar`, `T6` | rejected | pass the preflight name check but fail model compilation with `ValueError: color expression contains unsupported UFO tensors`. |
| Three-point projection | conditional | every three-point colour structure must be exactly proportional to 1, delta, T^a or f^abc with matching leg representations; the constant may be any number. |
| Colour accuracies | supported | `lc`, `nlc` and `full` for any accepted model; the choice is a run-card setting. |
| NLC and full colour weights | supported | exact for any number of quark lines; pure-gluon processes are exact up to 40 gluons. |
| Colour-charge balance | conditional | after crossing, fundamental and antifundamental legs must balance; otherwise the process has no colour plan. |
| Sector truncation | conditional | `process.max_color_sectors` produces a partial sum marked `selected` with direct contraction and is refused by on-the-fly, FFT and correlators. |
| Colour-flow selectors | conditional | LC outputs only; NLC and full outputs return one contracted entry per helicity. |
| `all-flow-union` layout | conditional | LC only, with complete coverage. |
| FFT contraction | conditional | Default `contraction="auto"` tries FFT for `nlc` or `full` with `recurrence` or `on-the-fly`, falling back to direct contraction for unsupported plans. FFT needs at most ten permutable gluons whose colour sectors form complete permutation orbits. LC, compiled/eager execution, and correlators retain direct/trace; `--color-contraction direct` opts out. Explicit `--fft trace` or `--fft adjoint` forces FFT and retains unsupported-plan errors. |
| Adjoint FFT selection | conditional | Default `fft_basis="adjoint"` adaptively uses DDM for certified Yang–Mills or single-insertion scalar HEFT (`HIG = 1`), fundamental chains/products for quarks, and trace tensors otherwise. Saved `fft_basis_selection` reports the actual representation; no universal DDM or speedup claim. Forced adjoint FFT with correlators is rejected; direct contraction ignores the FFT basis. |
| Shared-trace optimisation | conditional | applied only to pure gauge-boson processes whose Yang-Mills structure the model certificate proves; results are exact either way. |
| Correlators | conditional | representations 1, 3, -3, 8 from complete full-colour plans in compiled execution. |

## Propagators and gauge

| Feature | Status | Condition and consequence |
| --- | --- | --- |
| Default versus custom | supported | a UFO propagator counts as custom only if its expression differs from the loader's Feynman default for that particle. |
| Massless vectors | supported | Feynman gauge -i g/p^2, fixed. |
| Massive vectors | supported | unitary gauge with a fixed-width denominator; matched Goldstones are absorbed. |
| Feynman-gauge massive vectors | not applicable | no such kernel exists; Goldstone exchange appears only if the UFO supplies a custom massive-vector propagator, which is then used verbatim. |
| Massless spin-2 | supported | de Donder propagator with dimension parameter `dim` (default 4). |
| Massive spin-2 | experimental | Fierz-Pauli propagator behind `experimental-massive-spin-2`. |
| Widths | supported | fixed-width Breit-Wigner denominators only; zero widths are fine; complex masses are rejected. |
| Mass class | conditional | fixed by the restriction used at generation; runtime cards change values within that form, and setting a massive particle's mass to zero at runtime is not guarded. |
| Custom propagators | conditional | numerator and denominator must both be defined, use only the accepted tensors and registered functions, and lower exactly to the particle's component count; no gauge conversion is applied. |
| Disabled shortcuts | conditional | a custom propagator switches off the Weyl projection, transverse Yang-Mills lowering, Fierz auxiliary currents, reflection certificates and Goldstone removal for that particle. |
| Custom propagator coverage | conditional | plumbed through all execution modes but not exercised end to end by the test suite. |
| Ghost propagators | ignored | none; ghosts never propagate in tree amplitudes. |
| Non-propagating particles | conditional | excluded from external states but not from internal lines. |
| Gauge selection | not applicable | no option selects a gauge; the UFO must supply its own propagator for another choice. |
| Contact auxiliaries | supported | synthesized auxiliary currents carry no propagator and introduce no pole. |

## Process-level scope with a generic model

| Feature | Status | Condition and consequence |
| --- | --- | --- |
| Model inputs at generation | supported | any accepted source kind; restriction and simplification choices apply only when compiling from UFO or JSON. |
| Preflight gate | rejected | a model with any error code cannot generate or enumerate processes. |
| Process shape | supported | `a b > c d ...` with exactly two incoming and at least one outgoing particle; several requests may be joined with `\|`. |
| Loop-induced channels | rejected | tree level only; a channel without tree amplitudes is skipped with a warning inside a multiparticle request and is an error when requested alone. |
| Multiparticle labels | conditional | `all` always exists; `p` and `j` exist only when the model has massless triplet fermions or octet vectors; define your own labels otherwise. |
| Repetition syntax | conditional | write `3*g`; the built-in shorthands `3g` and `[d g]` are not understood by generic expansion. |
| `flavor_scheme` | ignored | no effect on generic models; use multiparticle labels instead. |
| `max_quark_lines` | conditional | bounds the number of external quark-antiquark pairs per subprocess; channels above it are dropped like loop-induced ones. |
| Permutation representatives | supported | one generated representative serves every reordering within the incoming and within the outgoing side. |
| Basis reductions | conditional | trace-reflection folding and the shared single-trace basis are applied only to certified pure Yang-Mills gluon processes; the unreduced exact basis is used otherwise. |
| Execution modes | conditional | `compiled` works from a raw UFO or JSON model; `recurrence` (default), `eager` and `on-the-fly` need a prepared bundle: `pyamplicol model compile MODEL out.pyamplicol-model --backend jit`. |
| Prepared-bundle clamps | conditional | the bundle fixes the backend and code-shaping settings; differing run-card values are overridden with a warning. |
| Backends | supported | `jit`, `asm`, `cpp` for compiled mode; only JIT artifacts support arbitrary precision. |
| Precision | conditional | `precision=16` needs no Symbolica; higher precision needs the Symbolica package and a JIT-backed recurrence, compiled or eager artifact, never on-the-fly. |
| Symbolica at generation | supported | generation from any model needs Symbolica; double-precision evaluation of a finished artifact does not. |
| Helicity selectors | supported | any generated helicity, by stable ID, for every accuracy and mode. |
| On-the-fly | conditional | keeps every selector for runtime, double precision only, one-point warm-up required. |
| Correlated Born evaluations | conditional | compiled full-colour generation with complete coverage; spin replacements on vector legs only. |
| Numerical current reuse | conditional | off by default; `--numerical-current-reuse` enables the certified search for compiled, eager and recurrence generation. |

## One-loop (v2) generation refusals

A one-loop v2 request is refused at generation, with a typed error whose
message starts with the rule's key and names the rule and the remedy, rather
than stored with a value it does not compute exactly. `generate --dry-run
--json` lists the keys (`estimated_coverage.loop_admission_refusals`,
`loop_configuration_refusals`, `loop_internal_refusals`,
`process_request_refusals` and `native_export_refusals`). MG5_aMC's loop_sm
ships with pyAmpliCol; see [One-loop QCD on loop_sm](loop-sm-and-madloop.md)
for its declaration, restriction recipes and the MadLoop syntax mapping, and
[One-loop QCD+QED on loop_qcd_qed_sm](electroweak-and-madloop.md) for the
shipped QCD+QED models and the configuration keys.

| Feature | Status | Condition and consequence |
| --- | --- | --- |
| LC/NLC of tree currents not exact in N_c | rejected | `colour-accuracy-nc-exact`: leading and next-to-leading colour of blocks whose trees mix U(1)-subtraction grades, right after the trees are built and again at every export. LC and NLC split such blocks by grade, so identical-flavour quark lines (from `u u~ > u u~ g` up) are exact and admitted, as are distinct flavours; the rule guards any request that would still mix grades. |
| LC/NLC with a counterterm outside the colour declaration | rejected | `colour-declaration-coverage`: a leading- or next-to-leading-colour request whose counterterm program can insert a counterterm vertex, or whose external legs take a field factor, that its formal-N_c colour declaration does not cover, at generation. pyAmpliCol ships the declaration of MG5_aMC's loop_sm (every counterterm vertex and field factor, matched per vertex by structure, so every restriction of the shipped copy or of MG5_aMC's directory is covered); a model's own declaration wins. An uncovered vertex no tree of the request can hold is left alone, as for unlowerable counterterms. Request full colour (the one-loop default), or supply a declaration that covers the vertex. |
| Particle veto of a loop species on unlabelled counterterms | rejected | `veto-unlabelled-counterterms`: a `particle_veto` removing a species that runs in the loops of a requested correction, on a model some of whose counterterm terms of that correction declare no loop content (`loop_particles = []`: MG5_aMC's `loop_qcd_qed_sm` and its G_mu variant, e.g. photon-only QED corrections with the Z, W, Higgs and Goldstones vetoed), named, before any tree is built: the veto acts on the counterterms through those labels only, which such a model writes nominally. Request the complete corrections; QCD vetoes on `loop_sm` (`particle_veto = ["t"]`) are unaffected. |
| Loop vertex the restricted tree model lacks | rejected | `loop-vertex-outside-tree-model`: a loop vertex at a coupling order the request's tree model does not hold, while its couplings there are not shown to vanish at the restriction, named with its particles and orders when a loop of the request uses it. Vertices whose couplings vanish identically at the restriction (decided from the coupling values, e.g. the off-diagonal CKM W vertices at `loop_qcd_qed_sm_Gmu`'s diagonal-CKM default) are no loop vertices and are not refused. |
| Forced seed FFT the FFT cannot serve | rejected | `forced-seed-fft-unserved`: `color.contraction = "symmetric-group-fft"` for a process whose seed table has no certified symmetric-group orbit pair of its external gluons (e.g. `u u~ > d d~ g`, one gluon), or with declared colour, spin or Ward observables (their columns are contracted directly), at generation; a forced FFT is never silently direct. `color.contraction = "auto"` serves them (the certified pairs by the FFT, the rest directly, the algorithm reported per evaluation). |
| Correlations with a leading-colour selection | rejected | `selected-correlators`: declared colour or spin correlations with any of the `[process]` helicity or colour selectors (`selected_source_helicities`, `selected_color_sector_ids`, `max_color_sectors`, `reference_color_order`) at leading colour, at generation before any tree, as the tree engine's correlated generation refuses them (a spin source on a leg of fixed helicity would ignore the selection); generate the correlations in an inclusive artifact. |
| Colour-flow selection below the leading colour | rejected | `subleading-colour-flows`: a generation-time colour-flow selection at leading colour (`selected_color_sector_ids`, or a truncating `max_color_sectors`) none of whose flows reaches the leading N_c power of the squared matrix element where its flows contribute (e.g. a flow of `u u~ > d d~ g` that joins each quark to its own antiquark), at generation: its leading-colour Born and virtual are zero. Select a leading flow with it (the subleading one then contributes exact zeros), or select the flows at evaluation (`Runtime.evaluate(..., color_flows=...)`, exactly zero). |
| Colour-flow selection interfering with an unselected flow | rejected | `interfering-colour-flows`: a generation-time colour-flow selection at leading colour one of whose flows interferes at the leading N_c power with an unselected flow, which happens between two Born orders (e.g. `u u~ > d d~` with QCD and photon Born amplitudes: the QCD flow (2,4,3,1) with the photon Born's (2,1,3,4) at QCD^2 QED^2), at generation: a generation-time selection restricts the Born itself and would drop that term. Select the interfering flows together, or select at evaluation, which keeps it. |
| Fermion line changing colour representation | rejected | `fermion-line-representation`: a vertex joining a coloured and a colour-singlet fermion (a leptoquark) that a tree or loop of the request can use, named, before any tree is built; veto one of its particles or bound its coupling order. |
| On-shell external photons with electroweak corrections | supported | per leg, as MG5_aMC's `!a!` tags (`[loop] external_photons`: `"final"` by default, `"all"`, `"none"` or a list of 1-based legs, initial legs allowed): a tagged photon couples with alpha(0) instead of the loops' coupling, with the matching charge-renormalisation term (MG5_aMC's alpha(0) formula), which needs a G_mu model with an adjustable 1/alpha(0) (MG5_aMC's `aEW0M1` at `SMINPUTS 1`, set by its `restrict_a0`) and the conversion counterterm `eCoup_DGmua0_UV_EW`, with the electromagnetic correction requested; stored with native point inputs (1/alpha(0) is a runtime parameter). An untagged photon (by default an initial-state, PDF photon) keeps the loops' coupling and its gamma_gamma pole and needs no model input. Tagged photons a model or request cannot convert are refused (`external-photon-scheme`, naming the reason and the remedy: untagged photons, or `restriction = "a0"`); a listed leg that is no on-shell external photon is refused (`external-photon-legs`). |
| Models outside 't Hooft-Feynman gauge | rejected | `feynman-gauge`: a source that declares no gauge 1, or an electroweak request on a bundle prepared in another gauge. |
| Anomalous vetoes | rejected | `anomalous-veto`: a `particle_veto` that breaks an anomaly-free multiplet whose Levi-Civita counterterm cell a tree of the request can hold; `generation.allow_anomalous_veto = true` lowers the cells anyway. |
| Multiparticle members without a Born | supported | skipped with a typed `loop-induced` record (dry run `skipped_members`, artifact extension `loop_skipped_members`); a request with no member left is refused. |
| Python-2 UFO | rejected | `ufo-python2`: a UFO whose modules are Python-2 code (the raw `loop_qcd_qed_sm_Gmu` download); the message names MG5_aMC's conversion and the shipped model. |
| Values written after their constructors | rejected for one-loop requests | `ufo-value-writes`: a UFO with module-level writes such as `X.value[k] += '...'` (the UFOmodeldb issue-1 patch of `loop_qcd_qed_sm_Gmu` as published); the model loader then certifies none of its values. `tools/developer/fold_ufo_value_appends.py` folds string appends into the constructors (the shipped copy is folded). |
| Decay requests | rejected | `decay-process`: a 1 -> n request (one initial particle), tree and one-loop alike. |
| MG5_aMC process syntax | rejected | `madloop-syntax`: NLO brackets (`[virt=QCD]`, `[QCD]`), coupling-order constraints (`QED=2`, `QCD^2==4`), `/`, `$`, decay chains and `@N` inside the expression; the message names the run-card setting (`process.loop_coupling_orders`, `process.max_coupling_orders`, `generation.particle_veto`). |
| Loop-induced processes | rejected | `loop-induced`: no Born amplitude within the coupling-order limits (`g g > h h`, `g g > z z`, `g g > a a`). |
| Coloured widths | rejected | `coloured-width`: a coloured particle with a nonzero width in a tree current (the top at every loop_sm restriction but `no_widths`); fix the width to zero in `[loop.preparation] fixed_parameters` (`WT = "0"`) or take `restriction = "no_widths"`. |
| Dirac-Weyl closures | supported | a fermion line joining massive and massless fermions through a chiral vertex whose massive current closes on a massless one (`b b~ > w+ w-`, `g b > t w-` on `loop_sm-no_b_mass`; W lines mixing massive and massless quarks at `zeromass_ckm`, `ckm`, `c_mass`): the one-loop Born closes at a vertex. Tree-level requests refuse the class with `dirac-weyl-closure` (no Dirac-Weyl closure contraction): the eager and recurrence lanes at generation, the on-the-fly lane at the first query whose anchor closes on such a line (`EvaluationError`; queries anchored elsewhere evaluate); some processes meet an older recurrence refusal first (`b u > d t`: "no exact closure for physical LC sector"), and requests whose coupling-order limits keep the chiral vertex off the line build (`t t~ > b b~` at QED 0). Otherwise keep the mass in the tree model and fix it to zero (`MB = "0"`, `ymb = "0"` at the default restriction). |
| Lorentz slot arity | conditional | a vertex referencing a constant scalar tensor with fewer slots than particles (loop_sm's `V_1001` with `SSS1`) is repaired by structure to the model's unique same-valued tensor of the right slots and recorded (`pyamplicol_source_repairs`); any other mismatch is `lorentz-slot-arity` when a request reaches the vertex. |
| Majorana fermions, multi-fermion contacts | rejected | `majorana-fermion`, `multi-fermion-contact`: at model preparation. |
| SymJIT plane check | conditional | `plane-mismatch`: a stored program whose binary64 SymJIT application disagrees with its exact evaluator at the generation's check is refused, naming the output (a model coupling by its UFO coupling and vertex, a counterterm coupling with its epsilon power, or a kernel). The check compares the two at complex points off the real axis, which finds a wrongly translated constant or function whatever the parameters' names and order, and at the request's parameter values, where a branch cut taken on different sides would make binary64 and double-double disagree (the translation makes the zero imaginary part of every branch-cut argument positive and evaluates every branch of a helper's conditional before selecting one: SymJIT alone gives a negated real number a negative zero imaginary part, and its common-subexpression elimination could leave a value shared with later code in a branch the parameters do not take; under the complex-mass scheme an argument real only when some widths are zero reads its real part when they are, in every precision); a program that reads a branch cut or a discontinuous helper of its inputs and has no such point is refused too. Every probe row also runs as a SIMD block of its own (the way a batch runs it), and a program whose optimised code fails the check is compiled again without optimisation and kept if that passes: SymJIT 2.27's SIMD complex product into aliased registers gave NaN in CM05's (`u u~ > e- e+ g [QCD QED]`, complex-mass scheme) parameter program, which is now stored and equals MadLoop. Counterterm couplings take MadLoop's side of the cuts their real arguments sit on at real masses, so electroweak requests on `loop_qcd_qed_sm_Gmu` (e.g. `g g > t t~` and `e- e+ > mu- mu+`) and `loop_qcd_qed_sm` are stored with native inputs. |

## Related pages

- [Models and Processes](models-and-processes.md) for the model workflow and
  the `model` commands.
- [Configuration](configuration.md) for the run-card fields referenced above.
- [Release and Support](release-and-support.md) for the validated release
  boundary.
