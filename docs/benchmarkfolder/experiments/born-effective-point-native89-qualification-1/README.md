# Real-Z Born/effective-point controls with NinjaRS and TIRust

Both prepared backend8/9 stages completed. This package covers only real
`u u~ > z [virt=QCD]` with the same two saved ordinary points and repetition of
the first point. It adds no CMS, Gmu, retained-stress, full-campaign or performance
qualification. No amplitude was rerun while retaining this evidence.

| Stage | Completed evidence | Result |
|---|---|---|
| Ordinary | Eight runs, 24 observed calls, four arms × backends8/9 | Twelve status218 and twelve status219; all six original/other-arm comparisons exactly equal |
| Fixed QP | Sixteen runs, 48 observed calls, both integrand policies | All status120, explicitly **unchecked**; maximum comparison residual `1.2823900401285503e-16` |
| Full-arm Born/coordinate checks | Twelve fixed-QP observed rows | Published Born bits exactly match the requested actual view; DP view exactly rounds C |
| Unstable-contribution warnings | All 24 raw streams | No `##W03` warnings |

The maximum fixed-QP residual occurs in the published Born coefficient with the
full arm's genuine-MP integrand policy. It reflects the already documented
precision-policy change. Positive CT4/status120 and absence of W03 warnings are
not independent stability or coefficient-accuracy proofs. The fixed-QP evidence
establishes this narrow source/transport/Born-consistency result only.

The four arms are original, guarded MP mass, mass plus graph kinematics, and
the full coupled Born/effective-point implementation. The unchanged current
dependency is TIRust DSO `eb3d67fd…`; NinjaRS is statically linked and its archive
and executable hashes are bound. Actual ELF dependency listings and process
maps are retained. Both stages finished within their 300-second caps on CPU14,
with all source/input bindings unchanged. Their 52.43/119.73-second protocol
durations are not performance measurements.

The reviewed successor source index `55af12e1…`, controller `23a53516…`, and
prepared manifest `de50e4a9…` are preserved. The original unexecuted controller
`0d26c54a…` and manifest `3dc89b69…` remain in their historical copies. No physical
inputs, generated objects or numerical settings changed between preparations.
The successor rejects every W03 warning before comparisons and records requested
QP policy separately from the selected path:

- All 24 ordinary observations use DP tensors and the host DP integrand point P.
- Twenty-four fixed-QP observations use QP tensors with DP integrands at
  `REAL(C,8)`.
- Twenty-four use QP tensors and genuine MP integrands at C/MP_P.

These input-view labels are source-bound construction-path observations, not a
new independent per-helicity trace. The full arm's Born and C8 checks additionally
use the retained exact binary words. The raw original/full differences, flags,
accuracy fields, MP state and return codes remain visible.

## Portable verification

```sh
python3 -B /path/to/born-effective-point-native89-qualification-1/verify.py
```

The verifier needs only Python3 and this directory. It verifies exact file/blob
hashes, the reviewed source/manifest lineage, all prepared dependency identities,
actual run inputs, ELF mapping snapshots and stage closures. It then imports
only the retained pure-Python parser definitions and reparses all 24 raw logs,
including W03 rejection, selected precision/policy labels and exact full-arm
Born/C8 checks. It independently recomputes all 18 cross-arm comparisons. It
does not invoke a compiler, loader, reducer, amplitude or controller main.

`payload.json.gz` contains 1,744 exact UTF-8 source/evidence paths with 510 unique
contents. This includes the reviewed controller history, relevant implementation
and generated real-Z source, cards/points, every raw log/map/runtime report and
both completed stages. `payload-inventory.json` binds the payload. Other pinned
dependencies/artifacts are identities only in `other-bound-identities.json`;
no native bytes or symlink objects are shipped. Top-level summaries and source
manifests are exact copies with their original paths recorded.

The earlier [legacy milestone](../born-effective-point-native/README.md), index
`923f9028…`, remains unchanged. This separate sibling directory preserves its
exact recursive inventory. Relevant source bytes were copied from that frozen
milestone and checked against the new source bindings; this package can be
verified independently of the ignored build tree and prior package.
