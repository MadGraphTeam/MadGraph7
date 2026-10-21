# Exact electroweak model source for benchmark reproduction

Both benchmark roles import `loop_qcd_qed_sm_Gmu` from
`.benchmark-work/original/models/loop_qcd_qed_sm_Gmu`. This directory retains
the exact source used there: 176 files including the UFO modules, restriction
cards, auxiliary SMWidth source and available license/readme text. Python
bytecode, serialized import caches and the generated debug log are excluded.
No model code is imported by the restore helper.

Use `loop_qcd_qed_sm_Gmu-actual-source.tar.gz` for reproduction. Its SHA256 is
`495514a1f52f4d3b7445cbaac87dfe87d07d628a01680a3bf0457408b13fb7dc`.
The original fourteen-module checksum is
`5dd9a66ef0b501f643251a65d9fa21a8be744b04959e1fcecd13eea51476a0e1`.
That checksum concatenates each sorted top-level Python filename encoded as
UTF-8, one NUL byte, and that file's exact bytes. The historical
`._object_library.py` file is included in this inventory.

From the repository root, with the pinned development environment available:

```sh
source .ninjars-build/dev-env.sh
python -B docs/benchmarkfolder/policy-v2/ew-model-source-final-v1/restore.py \
  --target .benchmark-work/original/models/loop_qcd_qed_sm_Gmu
python -B tools/ninjars_setup_references.py --verify-model-only
```

The target must be absent or empty. An existing model can first be checked
with the last command; the helper never replaces a nonempty target. It
verifies the archive checksum, all 176 source-file hashes/modes, the exact
member inventory and the original Python aggregate before writing. Files
are restored through a private staging directory followed by a directory
rename. No download, build, model import or amplitude call occurs.

`original-download-candidate.tgz` preserves the older archive found in the
baseline model cache, SHA256
`7655cb636257b4c67db7e1e09f232e7c0358dd92b85373ee83bdb9e7686ff65f`.
Its `__init__.py`, `object_library.py` and `write_param_card.py` differ from
the actual model. The exact differences remain in
`original-to-actual-diffs/`. Its fourteen-file aggregate is `d80ca7d3...`,
so a fresh download from the historical provenance URL is not a substitute
for the checksum-bound actual-source archive.

The recorded URL is
`http://madgraph.phys.ucl.ac.be/Downloads/models/loop_qcd_qed_sm_Gmu.tgz`.
This offline retention audit establishes source identity and restoration
readiness. It does not establish HTTP authenticity, the historical
transformation command/time, a clean-cache model import or new physics
results.

`publication-inventory.json` binds the durable publication paths and source
members. The `historical-*.json` files preserve the earlier private-state
records byte for byte; their original paths have not been silently changed.
No build archive, live model directory or import cache was modified to
prepare these inputs.
