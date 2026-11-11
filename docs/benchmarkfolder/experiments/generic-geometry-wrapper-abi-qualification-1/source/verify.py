"""File/source-only controller verification; no compiler or native execution."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index-sha256', required=True)
    a = p.parse_args()
    if prepare.identity(HERE / 'index.json')['sha256'] != a.index_sha256:
        raise ValueError('index hash mismatch')
    index = json.loads((HERE / 'index.json').read_bytes())
    for name, pin in index['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or (HERE / name).is_symlink():
            raise ValueError('unsafe payload path')
        if prepare.identity(HERE / name) != pin:
            raise ValueError('changed source payload: ' + name)
    row = prepare.verify_source3()
    for path, pin in index['tools'].items():
        if prepare.identity(Path(path)) != pin:
            raise ValueError('changed tool')
    loader = index['expected_interpreter']
    if prepare.identity(Path(loader['path'])) != loader['identity']:
        raise ValueError('changed expected interpreter')
    cases = json.loads((HERE / 'cases.json').read_bytes())
    answers = json.loads((HERE / 'expected.json').read_bytes())
    if prepare.corpus() != cases or prepare.expected(cases) != answers:
        raise ValueError('changed deterministic corpus/reference')
    if prepare.render(cases) != (HERE / 'inputs.txt').read_text():
        raise ValueError('changed wire corpus')
    print(json.dumps(dict(status='pass', indexed_payloads=len(index['files']),
                          direct_wrapper_cases=240, expected_available=180, expected_rejected=60,
                          generated_case=row['case'], full_sloop_lifetime_coverage=False,
                          compiler_or_scientific_native_executions=0), sort_keys=True))


if __name__ == '__main__':
    main()
