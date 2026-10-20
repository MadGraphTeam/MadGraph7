# Completed variant controls

All twelve freshly prepared streams pass: rotation and changing mass/scale
inputs with all four backends, fresh versus retained NinjaRS contexts, and
near-singular points with CutTools, Ninja and NinjaRS. The three near-singular
streams contain ten genuine QP evaluations each. All eight complete comparisons
pass, including rotation against the refreshed unrotated ggHH controls.

The read-only verifier reparses every raw stream and recomputes all 1,600
coefficient comparisons with the retained comparison code. Its largest tolerance
fraction is 0.009236. Fresh and retained NinjaRS rows match exactly. Preparation
and execution preserve the original driver, cards, point streams, numerical
policy and independent DP/MP mass and scale updates.

Run `python -B docs/benchmarkfolder/policy-v2/final-variant-controls-abi3-tensor-factor-v1/verify.py`
from the repository root after sourcing `.ninjars-build/dev-env.sh`.
`SHA256SUMS.json` binds the complete retained namespace. This package performs
no full-process performance acceptance and does not complete the eight-case gate.
