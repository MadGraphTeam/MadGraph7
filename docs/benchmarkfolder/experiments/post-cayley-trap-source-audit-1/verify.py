#!/usr/bin/env python3
"""Text-only source proposal verification: no native artifact is read/loaded."""
from hashlib import sha256
import difflib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def identity(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}


def main():
    index = json.loads((HERE / "index.json").read_text())
    for relative, wanted in index["files"].items():
        path = HERE / relative
        assert path.is_file() and not path.is_symlink(), relative
        assert identity(path) == wanted, relative
    proposal = json.loads((HERE / "proposal.json").read_text())
    for relative, wanted in proposal["inputs"].items():
        path = ROOT / relative
        assert path.is_file() and not path.is_symlink(), relative
        assert identity(path) == wanted, relative
    source = (ROOT / proposal["gram_source"]).read_text()
    old = proposal["old_expression"]
    assert source.count(old) == 1
    changed = source.replace(old, proposal["proposed_expression"])
    assert sha256(changed.encode()).hexdigest() == proposal["proposed_gram_sha256"]
    generated_patch = ''.join(difflib.unified_diff(
        source.splitlines(keepends=True), changed.splitlines(keepends=True),
        fromfile='a/crates/iregi-rust/src/direct/exec/gram.rs',
        tofile='b/crates/iregi-rust/src/direct/exec/gram.rs'))
    assert generated_patch == (HERE / "gram-zero-only.patch").read_text()
    evidence = json.loads((ROOT / proposal["localization_analysis"]).read_text())
    assert evidence["library_sha256"] == proposal["private_library_sha256"]
    assert [c["ELF_address"] for c in evidence["cases"]] == ['0x11fcecb', '0x2523cc3']
    assert all(not c["actual_register_operands_or_exception_subtype_available"]
               for c in evidence["cases"])
    for key in ['patch_applied', 'native_build', 'native_execution',
                'observed_zero_denominator', 'observed_dilog_overflow',
                'trap_fix_qualified', 'performance_acceptance']:
        assert proposal[key] is False, key
    print(json.dumps({"passed": True, "bound_inputs": len(proposal["inputs"]),
                      "native_execution": False, "patch_applied": False,
                      "trap_fix_qualified": False}, sort_keys=True))


if __name__ == '__main__':
    main()
