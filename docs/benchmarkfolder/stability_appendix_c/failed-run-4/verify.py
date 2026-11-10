"""Verify closed setup-failure receipts using capsule-local files only."""
import hashlib
import json
from pathlib import Path

SOURCE_SHA = 'bbdb3d82350f79eeccc421bf1ca1fa143957fd466f7d101cf3ba33daa34b7c4d'
RUN_SHA = '307b8a2dc9f4d95ca636fbabf496b3d7cf17a7cafa11d153a3dd222c3c5f626e'
PREP_SHA = 'cea9798b3127023153d19f6533931ae26acd786a02c290e264ca9f576eba07ad'
CASES = ('qcd_n2_gg_ttx', 'qcd_n3_uux_zgg', 'ew_n2_epem_mupmum', 'qcd_n4_uux_zzgg')


def verify(root=Path(__file__).resolve().parent):
    root = Path(root)
    sha = lambda blob: hashlib.sha256(blob).hexdigest()
    read = lambda name: json.loads((root / name).read_bytes())
    index = read('index.json')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() or p.is_symlink()}
    assert actual == set(index['files']) | {'index.json'}
    for name, item in index['files'].items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        path = root / name
        assert path.is_file() and not path.is_symlink()
        parent = path.parent
        while parent != root:
            assert not parent.is_symlink(); parent = parent.parent
        blob = path.read_bytes()
        assert len(blob) == item['bytes'] and sha(blob) == item['sha256']
    result = read('evidence/run-terminal.json'); started = read('evidence/run-started.json')
    inherit = read('evidence/inherit-terminal.json'); inherit_start = read('evidence/inherit-started.json')
    assert sha((root / 'evidence/run-terminal.json').read_bytes()) == RUN_SHA
    assert result['source_index_sha256'] == started['source_index_sha256'] == inherit['source_index_sha256'] == inherit_start['source_index_sha256'] == SOURCE_SHA
    assert result['status'] == 'failed' and result['error'] is None and result['request_count'] == 20
    assert result['preparation_sha256'] == started['preparation_sha256'] == inherit['preparation_sha256'] == PREP_SHA
    assert inherit['status'] == 'passed' and inherit['error'] is None
    assert all(inherit[name] == 0 for name in ('commands_executed','compiler_invocations','loader_invocations','point_generation_calls','amplitude_evaluations'))
    expected = [(case, b) for backends in ((1, 6, 7), (8, 9)) for case in CASES for b in backends]
    assert [(r['request']['case'], r['request']['backend']) for r in result['requests']] == expected
    for row in result['requests']:
        rid = row['request']['id']
        assert read('evidence/requests/' + rid + '/result.json') == row
        assert row['state'] == 'unexecuted' and row['native_started'] is False and row['readonly_guard_passed'] is True
        assert 'native' not in row and 'lock' not in row and 'readonly_guard_error' not in row
        assert row['error']['class'] == 'FileNotFoundError' and row['error']['message'].endswith("/param_card.dat'")
        assert len(row['primary_call_outcomes']) == 300 and all(x['state'] == 'unexecuted' for x in row['primary_call_outcomes'])
        card = (root / ('written-cards/' + rid + '.dat')).read_bytes()
        assert row['effective_card']['settings']['MLReductionLib'] == str(row['request']['backend'])
        assert row['effective_card']['readback'] and all(x == row['effective_card']['settings'] for x in row['effective_card']['readback'].values())
        assert set(row['card_write_intent']) == set(row['effective_card']['files'])
        assert all(checksum == sha(card) for checksum in row['effective_card']['files'].values())
        assert card.decode().splitlines()[-1] == '/* End of param file */'
    retained = read('retention.json')
    assert retained['root_reported_inheritance_terminal'] == {'session_id': 54810, 'exit_code': 0}
    assert retained['root_reported_run_terminal'] == {'session_id': 9030, 'exit_code': 1}
    return {'verified': True, 'requests': 20, 'native_starts': 0, 'unexecuted_primary_calls': 6000,
        'readonly_guards_passed': 20, 'inherited_new_compiler_loader_point_amplitude_calls': 0}


if __name__ == '__main__':
    print(json.dumps(verify(), sort_keys=True))
