"""Add host-affinity provenance to the unchanged fixed statistical analysis."""
import argparse
import json
from pathlib import Path

import analyze
from affinity_addendum import HERE, ROOT, sha, verify_addendum


def analyze_result(path):
    addendum = verify_addendum()
    raw = json.loads(path.read_text())
    assert raw['cpu_affinity'] == [234] and raw['thread_siblings'] == '234'
    assert analyze.bound(raw['execution_affinity']) == HERE / 'execution-affinity.json'
    assert raw['controller'] == addendum['bindings']['controller']
    result = analyze.analyze(path)
    result['execution_cpu'] = 234
    result['affinity_addendum_identity'] = addendum['identity']
    result['affinity_addendum_sha256'] = sha(HERE / 'execution-affinity.json')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    assert Path.cwd() == ROOT
    result = analyze_result(args.result.resolve())
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
