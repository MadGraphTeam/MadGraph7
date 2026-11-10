# Preserved 12-page intermediate report

This append-only archive preserves the report committed at
`bd8eb72a179a2a6a4385af07d13507f94c52df95`. The PDF, original input ledger,
context, renderer, validators, status and compact provenance files are copied
byte for byte. The canonical report, its current ledger, the prior eight-page
history and the NinjaRS report are unchanged.

The enclosing installer inventory later intentionally gained the `eb3d…`
installation package. Consequently its current bytes differ from the original
ledger. This archive retains the original `qualification/tirust-install/index.json`
from the report commit; it does not reseal the old report to the new inventory.
The PDF itself still has its original hash. The report remains an intermediate
assessment of the historical candidate, with its failed runtime gate and final
campaign pending.

The ledger binds 4,693 input files (287,811,678 bytes). To avoid committing a
second copy of that evidence, reconstruction reads the exact Git blobs from
the report commit and checks every byte count and SHA256 against the original
ledger. The Git history containing that commit is required. It never substitutes
the current working files. `archive-manifest.json` binds the compact copies;
`reconstruct.py` also pins the original commit, PDF and ledger hashes.

From the MadGraph repository root:

```sh
source tools/tirust_env.sh
PYTHONDONTWRITEBYTECODE=1 python3 docs/benchmarkfolder/history/report-before-split-dp-campaign/reconstruct.py
PYTHONDONTWRITEBYTECODE=1 python3 docs/benchmarkfolder/history/report-before-split-dp-campaign/reconstruct.py --destination .tirust-build/report-bd8-reconstruction
PYTHONDONTWRITEBYTECODE=1 python3 .tirust-build/report-bd8-reconstruction/build_report.py --verify-only
PYTHONDONTWRITEBYTECODE=1 python3 .tirust-build/report-bd8-reconstruction/build_report.py --output "$PWD/.tirust-build/report-bd8-reconstruction/recompiled.pdf"
```

The destination must be new, ignored and inside this repository. The final
command renders only the private reconstruction and never overwrites the
canonical PDF. Dependency verification alone is not a claim of bit-identical
PDF reproduction; the retained `verification.json` records whether rendering
was attempted and whether its resulting bytes match. No command launches
MadLoop, a numerical campaign or a native library build.
