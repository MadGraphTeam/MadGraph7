<!-- SPDX-License-Identifier: 0BSD -->
# Development Documentation

This directory separates stable project contracts from feature-specific design
and validation records.

## Stable contracts and release records

- [`API_CONTRACT.md`](API_CONTRACT.md)
- [`CONFIG_CONTRACT.md`](CONFIG_CONTRACT.md)
- [`PACKAGING_CONTRACT.md`](PACKAGING_CONTRACT.md)
- [`PHYSICS_EXTRACTION_CONTRACT.md`](PHYSICS_EXTRACTION_CONTRACT.md)
- [`ARCHITECTURE_DECISIONS.md`](ARCHITECTURE_DECISIONS.md)
- [`MODEL_ASSET_PROVENANCE.md`](MODEL_ASSET_PROVENANCE.md) and
  [`model-assets/`](model-assets/README.md)
- [`PORT_MANIFEST.toml`](PORT_MANIFEST.toml),
  [`SCHEMA_VERSIONS.toml`](SCHEMA_VERSIONS.toml), and
  [`SOURCE_BASELINE.toml`](SOURCE_BASELINE.toml)
- [`MILESTONES.md`](MILESTONES.md)

## Feature records

- [`arena/`](arena/README.md) contains eager/compiled Direct-Arena designs,
  milestone evidence, optimization records, and historical handoff material.
- [`recurrence/`](recurrence/README.md) contains recurrence ABIs, colour
  designs, acceptance evidence, and the ordered audit trail.

Feature records document the implementation history and validation rationale.
Current executable behavior is defined by the source, tests, and stable
contracts rather than by an old plan or session handoff.
