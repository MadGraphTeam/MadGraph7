"""File-only verification. Does not rerun generation, tests or native programs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def identity(path):
    data = path.read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index-sha256')
    p.add_argument('--live-parent-sources', action='store_true')
    p.add_argument('--check-generation-tree', action='store_true')
    a = p.parse_args()
    index_path = HERE / 'source-index.json'
    if a.index_sha256 and identity(index_path)['sha256'] != a.index_sha256:
        raise ValueError('source index digest mismatch')
    index = json.loads(index_path.read_bytes())
    for name, pin in index['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('unsafe payload path')
        path = HERE / relative
        if path.is_symlink() or identity(path) != pin:
            raise ValueError('changed payload: ' + name)
    sys.path.insert(0, str(HERE))
    from bindings import verify_parents
    parents = verify_parents(live=a.live_parent_sources)
    rows = json.loads((HERE / 'generation-results.json').read_bytes())
    if len(rows) != 7:
        raise ValueError('generation case count changed')
    counts = {True: 0, False: 0}
    for row in rows:
        if row['compiler_execution'] is not False or row['numerical_execution'] is not False:
            raise ValueError('unexpected native claim')
        contract = row['geometry_window_source_contract']
        if contract['runtime_abi_qualified'] is not False or contract['scientific_dispatch_enabled'] is not False:
            raise ValueError('unexpected runtime/scientific admission')
        for width in ('64', '128'):
            value = contract['slots'][width]['source_window_supported']
            if type(value) is not bool or value != row['has_born']:
                raise ValueError('changed source window classification')
            counts[value] += 1
        if a.check_generation_tree:
            root = HERE / 'generation-work' / row['case']
            for name, pin in row['generated'].items():
                if identity(root / name) != pin:
                    raise ValueError('changed generated source: ' + str(root / name))
    if counts != {True: 10, False: 4}:
        raise ValueError('source window counts changed')
    print(json.dumps(dict(status='pass', verified_payloads=len(index['files']),
                          parents=parents, historical_generated_cases=7,
                          supported_source_windows=10, unknown_source_windows=4,
                          generation_tree_checked=a.check_generation_tree,
                          new_compiler_or_scientific_native_executions=0,
                          runtime_abi_qualified=False, scientific_dispatch_enabled=False),
                     indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
