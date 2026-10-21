# Main NinjaRS installation publication handoff

This helper prepares a byte-identical copy of the successful private actual-CLI
installation and preserves the entire old main `HEPTools/ninjars` tree. It does
not compile or execute any reduction engine, update dependencies, change MG7
configuration, delete old evidence, or clean any recovery directory.

The validated new archive is
`800a941faf3cdfdab764fdffe1d5ab73f84984a786aa194596b753af6b6f5d42`,
from actual installation revision `d05b59f3277ce0d727cb2816d8b7ad45107e58c4`.
The retained `installation.json` describes all ten artifacts, source inventory,
compiled Fortran smoke and Cargo/native-link metadata. The new copy must match
that complete tree, not merely an archive pathname. The original private prefix
may be named `ninjars-unavailable` by the relocation proof; exactly one ordinary
or hidden source tree with all expected hashes is accepted.

The main archive remains the historical
`1199bc84570bee76a6b162a86fd5c191479236b2c0617fe4bf250d5de0d9df32`
until root explicitly publishes after the private SA/NLO proof passes.

## Copy-only preparation

```sh
source .ninjars-build/dev-env.sh
python -B docs/benchmarkfolder/policy-v2/main-install-publication-v1/publish.py --prepare
python -B docs/benchmarkfolder/policy-v2/main-install-publication-v1/publish.py --verify-prepared
```

Preparation uses an exclusive new ignored workspace:
`.benchmark-work/policy-v2/main-install-publication-v1`. It copies the complete old
main prefix to `old-prefix-copy` and the exact validated private tree to
`transaction/install`, then records old/new full-tree file hashes, modes,
symlinks, sizes and identities plus the original common-prefix alias state.
It rechecks that main installation and aliases are unchanged. These private
archive/module/build bytes remain ignored and must not be pushed.

## Publication remains held

Only root may invoke `--publish`, after its private standalone/NLO verification
passes and root provides the literal handoff and final proof manifest digest:

```sh
python -B docs/benchmarkfolder/policy-v2/main-install-publication-v1/publish.py \
  --publish --root-handoff 'ROOT_EXPLICIT_PUBLICATION_HANDOFF' \
  --private-proof-manifest-sha256 FINAL_PRIVATE_SA_NLO_MANIFEST_SHA256
```

The helper refuses publication without that proof, checks its read-only `--live`
verifier and all standalone DP110/QP120/stable218 and twenty NLO pole controls,
then rechecks every prepared byte and alias before calling the unchanged tested
`ninjars_install._publish` transaction. It retains the transaction's `previous`
old-prefix tree and previous-alias journal as well as the independent old copy.
Regular user files/directories under the common library prefix are preserved.
No cleanup follows publication.

`transition.json` records the actual main archive, interface/module, Cargo and
link metadata digests and both complete old/new manifests. It explains that old
live proofs using the former main path remain historical and have not been
retargeted. Neither old proof seals nor benchmark generation identities are
rewritten. The helper copies a previously compiled and tested tree; it does not
substitute a new build or claim new numerical/timing measurements.
