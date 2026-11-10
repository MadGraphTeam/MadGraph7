The single actual seal is retained at `../provisional-retention-2/sealed-1/index.json`, SHA-256 `4ffb7643943f40acc6ee099e78287ef99f05e2415e828088a2e46bd8d4e5ff76`. The v1 proposal and reviewed additive v2 correction retain their original source indices. No reseal or native execution was performed during this delivery check.

`relocation.log` records the successful one-time offline verification from a temporary copied package, with subprocess creation forbidden. It reparsed exactly 29 completed results from all 66 closure rows; five deadline failures and 32 unrun rows remain unavailable. The exact returned result-map digest is recorded without duplicating the large numerical arrays. This verifies retained descriptive observations, not performance acceptance or overall goal completion.

The owned staging scope is exactly `provisional-retention-1`, `provisional-retention-2` (including its compact `sealed-1`), and this delivery directory. `stage-manifest.json` binds every staged file except itself by path, SHA-256, byte count and encoding. It also records the decompressed UTF-8 JSON payload hash/length. No native artifacts or original experiment tree are staged. The sealed payload retains 3,162 logical files and 214,424,480 original bytes in one 22,853,689-byte deterministic gzip file; its exact logical inventory remains in the sealed index.

Reproduce the same relocated verification from the repository root with Python 3; this executes only the frozen Python parsers:

```bash
python3 -B - <<'PY'
import importlib.util, shutil, tempfile
from pathlib import Path
from unittest.mock import patch
source = Path('docs/benchmarkfolder/campaigns/split-dp-terminal-1/provisional-retention-2/sealed-1')
with tempfile.TemporaryDirectory() as directory:
    copied = Path(shutil.copytree(source, Path(directory) / 'portable'))
    spec = importlib.util.spec_from_file_location('retained', copied / 'retain.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    with patch('subprocess.Popen', side_effect=AssertionError('No native execution')):
        results = helper.replay_completed(copied, '4ffb7643943f40acc6ee099e78287ef99f05e2415e828088a2e46bd8d4e5ff76')
    assert len(results) == 29
    print('Relocated frozen replay passed: 29 completed results; all 66 outcomes retained.')
PY
```
