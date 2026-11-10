#!/usr/bin/env python3
"""Verify static source bindings only; no native artifacts or workers."""
from hashlib import sha256
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def identity(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": sha256(data).hexdigest()}


def main():
    index = json.loads((HERE / 'index.json').read_text())
    for name, expected in index['files'].items():
        path = HERE / name
        assert path.is_file() and not path.is_symlink(), name
        assert identity(path) == expected, name
    audit = json.loads((HERE / 'audit.json').read_text())
    for name, expected in audit['inputs'].items():
        path = ROOT / name
        assert path.is_file() and not path.is_symlink(), name
        assert identity(path) == expected, name
    for anchor in audit['anchors']:
        lines = (ROOT / anchor['path']).read_text().splitlines(keepends=True)
        assert ''.join(lines[anchor['first_line']-1:anchor['last_line']]) == anchor['text']
    proof = json.loads((ROOT / audit['closed_lifecycle_summary']).read_text())
    coverage = proof['coverage']
    assert coverage['measured_physical_calls_total'] == 76
    assert coverage['generated_propagator_max'] == 3
    assert coverage['generated_rank_max'] == 2
    assert audit['implementation'] is False and audit['native_execution'] is False
    assert audit['active_numerical_data_read'] is False and audit['performance_claim'] is False
    count = (6+1)*(6+2)*(6+3)*(6+4)//24
    assert count == audit['dimensional_example']['coefficients'] == 210
    output = 64*8*count
    allocation = 64*(4*6+6*6+2*6+8*count) + 16*8*count
    assert output == audit['dimensional_example']['qp_output_ascii_bytes'] == 107520
    assert allocation == audit['dimensional_example']['fortran_transport_payload_bytes'] == 139008
    print(json.dumps({'passed': True, 'inputs': len(audit['inputs']),
                      'native_execution': False, 'performance_claim': False}, sort_keys=True))


if __name__ == '__main__':
    main()
