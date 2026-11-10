"""Pure replay of closed preparation failure; no original work paths opened."""
import hashlib
import json
from pathlib import Path


def verify(root=Path(__file__).resolve().parent):
    root = Path(root)
    index = json.loads((root / 'index.json').read_bytes())
    assert set(p.name for p in root.iterdir()) == set(index['files']) | {'index.json'}
    for name, item in index['files'].items():
        assert Path(name).name == name
        path = root / name
        assert path.is_file() and not path.is_symlink()
        blob = path.read_bytes()
        assert len(blob) == item['bytes'] and hashlib.sha256(blob).hexdigest() == item['sha256']
    terminal = json.loads((root / 'prepare-terminal.json').read_bytes())
    retained = json.loads((root / 'retention.json').read_bytes())
    assert terminal['source_index_sha256'] == retained['source_index_sha256']
    assert terminal['status'] == retained['status'] == 'failed'
    assert terminal['error'] == retained['failure']
    assert terminal['amplitude_evaluations'] == 0
    assert terminal['cases'] == {} and terminal['completed_cases'] == []
    assert len(terminal['requests']) == 20 and all(r['state'] == 'unexecuted' for r in terminal['requests'])
    assert retained['compiler_launches'] == retained['loader_query_launches'] == 0
    assert retained['root_reported_terminal']['session_id'] == 15086
    assert retained['root_reported_terminal']['exit_code'] == 1
    return {'verified': True, 'status': 'failed', 'compiler_launches': 0,
            'loader_query_launches': 0, 'amplitude_evaluations': 0,
            'unexecuted_requests': 20}


if __name__ == '__main__':
    print(json.dumps(verify(), sort_keys=True))
