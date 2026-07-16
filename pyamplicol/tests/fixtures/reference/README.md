# Clean Reference Fixtures

`physics-v1.json` contains compact numerical and topology observations from
the clean source revision recorded in `docs/development/SOURCE_BASELINE.toml`.
It deliberately does not contain serialized evaluators, generated binaries,
or schema-v2 manifests.

Resolved values are stored sparsely by stable physical helicity and color
component ID. Missing entries are expected to be exact structural zeros.
Tests must compare totals as well as every listed component and must verify
that the generated metadata declares the recorded complete coverage.

These fixtures establish behavior for the standalone schema-v3 port. They are
not an excuse to preserve schema-v2 compatibility.

Public color-component identifiers are normalized to the schema-v3 spelling
(`color:contracted` and `flow:singlet`). This normalization changes no captured
matrix-element value, momentum, or topology count.

The `d d~ > z g` LC helicity labels were independently rechecked against the
recorded clean source revision with a fixed-helicity generation. This corrected
an earlier hand-assembled permutation of the LC labels; the component values,
their sum, topology, and the already-correct NLC/full mappings did not change.

`CAPTURE.toml` records the exact source revision, clean-path proof, dependency
revisions, toolchain, watchdog policy, command templates, and the two fixture
captures repeated during the standalone bootstrap. Paths under `/tmp` are
historical output labels only; no fixture depends on them.

`legacy-fortran-v1.json` records an independent replay through the pinned
Fortran AmpliCol color probe. It covers the built-in-SM `d d~ > z` and
`d d~ > z g` fixtures in LC, NLC, and full color. The developer-only
`tools/developer/legacy_amplicol.py` runner regenerates process files, maps
physical external-leg and helicity orderings explicitly, and checks both
summed and resolved values with the independent-oracle tolerance.
