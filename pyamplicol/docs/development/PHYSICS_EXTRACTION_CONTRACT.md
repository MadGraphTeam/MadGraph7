# Physics Extraction Contract

This document records the first independent physics audit. It is normative
for the behavior-preserving extraction from source revision
`643bc6f99d7b2249af0a85204768df243e612411`.

## Port Boundary

Freeze numerical behavior before reorganizing:

- color contraction and planning;
- process IR, enumeration, and all-outgoing crossing;
- current identity and DAG construction;
- lowering, phase-space generation, stage compilation, and Symbolica
  evaluator construction;
- built-in model parameters and kernels;
- UFO model IR, tensor projection, contact decomposition, and propagators.

Rewrite the CLI, artifact writer, loader boundary, logging, reporting, model
asset access, generated API bundle, and Python runtime facade around those
locked contracts. Validation-only Fortran adapters and reference kernels stay
under developer tooling and are excluded from distributions.

The extraction initially preserves both Standard Model implementations. The
built-in path still uses its hard-coded production model, while UFO/JSON uses
the compiled external-model path. Converging those implementations is a
later, separately validated refactor.

## Required Invariants

- UFO and serialized JSON pass through the same `ufo-model-loader` model and
  produce identical restricted, wrapped-index, canonical, and compiled model
  representations.
- External filtering, particle/antiparticle crossing, external order,
  helicity bases, Weyl projection, propagators, spin-2 polarizations, and
  n-ary contact decomposition are preserved.
- A contact coupling is applied exactly once.
- `CurrentIndex` retains every physics identity field. Optimized and
  unoptimized DAGs agree point by point.
- LC flow, reflection, helicity, averaging, coupling, and identical-particle
  weights are each applied exactly once.
- NLC/full performs one coherent sparse color contraction, applies the
  off-diagonal factor two exactly once, and never reapplies an LC color
  factor.
- Inconsistent helicity weights in a contraction group are errors. They must
  never be silently averaged.
- Unsupported colored representations are errors. They must never be treated
  as singlets.
- Final-state permutation reuse is validated against fully materialized
  direct artifacts. It is not described as generic crossing reuse.
- Resolved evaluation executes one DAG. LC exposes physical
  `(point, helicity, flow)` values; NLC/full exposes one contracted color
  component per helicity. Structural zeros remain exact and resolved members
  sum to the compatibility total.

## Model Determinism

The historical scalar UFO modules read `UFO_SCALARS_MODEL_*` and
`UFO_GRAVITY_MODEL_*` environment variables without including them in the
compiled-model cache key. Standalone model loading must start from a sanitized
environment. Any retained model option becomes a typed input included in
canonical content, provenance, and cache identity. The historical undefined
`N_POINT_INTERACTIONS` gravity branch must be removed or corrected before the
asset is shipped.

## Reference Ladder

The compact baseline starts with:

- `d d~ > z` and `d d~ > z g` in LC/NLC/full;
- scalar contact `scalar_0 scalar_0 > scalar_0 scalar_0`;
- spin-2 `scalar_0 scalar_0 > graviton graviton`.

The built-in-SM `d d~ > z` and `d d~ > z g` rows are also replayed through
the pinned Fortran AmpliCol color probe in LC, NLC, and full color. This
independent check covers the summed matrix element and every recorded nonzero
physical-helicity component. The probe is developer-only, links no LHAPDF,
and is never imported or packaged by the installed distribution.

Before the model/generation milestone is accepted it expands to:

- process-set `d d~ > z g` and `d d~ > z g g`;
- `d d~ > t t~` and `g g > t t~ g`;
- `g g > g g`;
- charged-current and neutral leptonic examples;
- identical-vector/Higgs interference;
- multi-quark-line LC/NLC/full;
- scalar contacts with two, three, and five final scalars;
- scalar propagation with mass, width, and coupling variations;
- two and three final gravitons;
- reordered final states for permutation reuse.

Use at least three deterministic generic points and one unstable
high-precision point for each substantive row. Cross-language agreement
checks one Rust core; release physics additionally requires pinned Fortran,
analytic scalar, exact-color, and independent spin-2 oracles.

## Known Pre-Port Limitations

- Existing schema-v2 artifacts have unconfined paths, no payload hashes,
  ambiguous serialized-state fallbacks, nontransactional writes, and Python
  pickle diagnostics. They are rejected, not migrated.
- Resolved topology replay is not proven in the reference runtime. Schema v3
  must not claim that capability until direct expanded-versus-replayed
  resolved values pass.
- The old 180-case cache came from a dirty, older revision and contains
  absolute paths. It is not a release fixture and will be regenerated.
