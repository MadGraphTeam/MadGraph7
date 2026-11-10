"""Freeze private source-generation evidence; performs no build or execution."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report = json.loads((HERE / 'source-tests.json').read_text())
    assert report['successful'] and report['tests_run'] == 11
    assert report['consumer_sha256'] == sha(HERE / 'src/madgraph/loop/kinematic_consumer.py')
    assert report['test_sha256'] == sha(HERE / 'test_generation.py')
    for name, digest in report['protected_inputs'].items():
        assert sha(ROOT / name) == digest, name
    preparation = json.loads((HERE / 'preparation.json').read_text())
    assert preparation['derived']['consumer'] == report['consumer_sha256']
    assert preparation['derived']['exporter'] == sha(HERE / 'src/madgraph/loop/loop_exporters.py')
    assert preparation['derived']['patch'] == sha(HERE / 'consumer.patch')
    files = {str(p.relative_to(HERE)): {'sha256': sha(p), 'bytes': p.stat().st_size}
             for p in sorted(HERE.rglob('*'))
             if p.is_file() and p.name != 'source-index.json'}
    assert all('__pycache__' not in name and not name.endswith('.pyc') for name in files)
    index = {'scope': 'Private source-only QP8/9 consumer; no native or production qualification.',
             'tests_passed': 11, 'files': files,
             'protected_inputs': report['protected_inputs']}
    (HERE / 'source-index.json').write_text(json.dumps(index, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'files': len(files), 'source_index_sha256': sha(HERE / 'source-index.json')}))


if __name__ == '__main__':
    main()
