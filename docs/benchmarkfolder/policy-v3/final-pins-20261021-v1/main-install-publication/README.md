# Final pinned installation publication

The root explicitly authorized copying the successful actual CLI
`install ninjars --update` image from revision
`856e76e669618ec8d0f799b9c8e6f86819dcd986` to `HEPTools/ninjars`.
The publication invokes the existing recoverable installer transaction and
performs no compilation or numerical evaluation.

All ten installed files, the manifest/lock/Git provenance, and all 45 final
bridge source files match the accepted installation record and source-v4
freeze. The new static archive is
`b0d69deb8c16deb13293176296afc5e18dbeeebe778a20fa39b61804fba89d4c`;
the new configuration is
`a2b427c4e1c766180cec068ca716e2ff3d035226b06864c50156202bcfa4617f`.

The entire previous `800a941f…` installation survives byte-for-byte, including
file modes, in an independent copy and the transaction's rename backup under
`.benchmark-work/policy-v3/final-pins-20261021-v1/main-install-publication/`.
The previous common-library aliases also have an independent copy and the
retained transaction journal. Earlier validation records and the private
validated image are preserved.

Run the read-only custody checks from the repository root:

```sh
python3 -B docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/main-install-publication/verify.py
python3 -B docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/main-install-publication/verify.py --live
```

The default checks retained records and source identities. `--live` additionally
rehashes the current main installation, both old backups, aliases and source
copies. It checks the original private installed image while visible; relocation
tests may subsequently hide that private prefix and the verification output
states whether it was present.

`publish.py` is a retained one-shot record and refuses to overwrite an earlier
attempt. Publication qualifies installed-image custody and reuses the actual
CLI smoke result. Standalone/NLO export, final scientific comparisons, stability
and timing gates are separate observations.
