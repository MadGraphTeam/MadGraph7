#!/usr/bin/env python3
"""Portable replay plus exact proof that only the original file reader changed."""

import difflib
import hashlib
import types
from pathlib import Path

from portable_inputs import file_record, local_path, strict_json, verify_payloads


HERE = Path(__file__).resolve().parent
ATTEMPT1_INDEX = "7f8661e635664374023e057dedf45b6232c13407d01f62ee10605e16bbebdcd0"
ORIGINAL_READ = '''        path = ROOT / name
        assert path.is_file() and not path.is_symlink()
        blob = path.read_bytes()
'''
PORTABLE_READ = '''        blob = read_input(HERE, name, PINS)
'''


def verify(root=HERE):
    root = Path(root)
    verify_payloads(root)
    old_index_bytes = local_path(root, "attempt1-index.json").read_bytes()
    assert hashlib.sha256(old_index_bytes).hexdigest() == ATTEMPT1_INDEX
    old_index = strict_json(old_index_bytes)
    copies = {"README.md": "attempt1-README.md", "analyze.py": "attempt1-analyze.py",
              "analysis.json": "analysis.json"}
    for original, copied in copies.items():
        assert file_record(local_path(root, copied).read_bytes()) == old_index["files"][original]
    original = local_path(root, "attempt1-analyze.py").read_text()
    assert original.count("from pathlib import Path\n") == 1
    assert original.count("ROOT = HERE.parents[4]\n") == 1
    assert original.count(ORIGINAL_READ) == 1
    adapted = original.replace("from pathlib import Path\n",
                               "from pathlib import Path\n\nfrom portable_inputs import read_input\n")
    adapted = adapted.replace("ROOT = HERE.parents[4]\n", "")
    adapted = adapted.replace(ORIGINAL_READ, PORTABLE_READ)
    assert local_path(root, "analyze.py").read_text() == adapted
    patch = "".join(difflib.unified_diff(original.splitlines(True), adapted.splitlines(True),
                                      fromfile="attempt1/analyze.py", tofile="portable-analysis2/analyze.py"))
    assert local_path(root, "reader-only.patch").read_text() == patch
    module = types.ModuleType("portable_closed_analysis")
    module.__file__ = str(root / "analyze.py")
    exec(compile(local_path(root, "analyze.py").read_bytes(), module.__file__, "exec"), module.__dict__)
    result = module.analyze()
    serialized = module.encoded(result).encode()
    assert serialized == local_path(root, "analysis.json").read_bytes()
    return {"passed": True, "exact_attempt1_analysis": True, "reader_only_delta": True,
            "attempt1_index_sha256": ATTEMPT1_INDEX,
            "analysis_sha256": hashlib.sha256(serialized).hexdigest(),
            "closed_input_count": len(module.PINS), "accepted_rows": 1200,
            "finite_only_target_points": 28, "new_native_evaluations": 0}


if __name__ == "__main__":
    print(verify())
