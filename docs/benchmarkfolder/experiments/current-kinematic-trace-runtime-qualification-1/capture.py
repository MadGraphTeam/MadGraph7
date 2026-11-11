"""Retain only explicitly closed ordinary/stress trace evidence; text only."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
HERE = Path(__file__).resolve().parent
WORK = ROOT / '.tirust-build/kinematic-provenance-current-executor-1'
BUILD = HERE.parent / 'current-kinematic-trace-build-qualification-1'
BUILD_SEAL = '89697913341ecc5fe765fb023cc09bced783eb3eedb85436afe1ca7d4dc6c033'
PINS = {
    'results/ordinary-terminal-closure.json': '56d4249509bbda4d1aa3c328b18b42176c7419f4805a1d4454a73a7fa3608fcf',
    'results/stress-terminal-closure.json': 'df0987b3f0c20dd26c4a903ff7c4acc7fb6cce4e433a50071a82918194553e9e',
    'results/root-ordinary-review.json': '4834a317fbe400d60a0515f184985dc07deb2107afccb7bbcc98015ec4a25da1',
    'results/root-stress-review.json': '2954d77a73bf5a31ff97b4e8b4f52b39989b5a62be17fd0e301c8a157c2b2484',
    'results/root-ordinary-decoder-review.json': '3940ef665c827a2bebcec72a3f5bc9b879432880be25540a62a33bd5c9e19d8c',
    'ordinary-admission-1/index.json': '325eb3205b7990e5e8ec367b25ef0af48b41d7271e227dbbd15bdc878be560cf',
    'stress-admission-1/index.json': '9c81ae01f16c4e9a02493a3fac49198c59f0540c2f079c0a4d2aafee2a004735',
}
ANALYSES = {
    'decoder': ('.tirust-build/kinematic-provenance-analysis-1', 'df0c1113bee8a88bc6e9a85b6770faa17820e66707ff9d33e45776e69a939040'),
    'ordinary': ('.tirust-build/kinematic-provenance-analysis-ordinary-1', '5842fea99e508654bf9f032f371530943cc80cb4cd188aab0bb914281537a453'),
    'stress': ('.tirust-build/kinematic-provenance-analysis-stress-1', '66fab0dde170020c83840364845d4413f7130f1c09dab1105fd2e95a0f13611d'),
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    if sys.argv[1:] != ['--execute'] or Path.cwd() != ROOT or (HERE / 'evidence-index.json').exists():
        raise ValueError('Explicit fresh retention from the repository directory required')
    if not all(pin for _, pin in ANALYSES.values()):
        raise ValueError('Stress analysis must be sealed before retention')
    build_raw = (BUILD / 'evidence-index.json').read_bytes()
    if sha(build_raw) != BUILD_SEAL:
        raise ValueError('Published sibling build packet changed')
    build_index = json.loads(build_raw)

    def build_data(name):
        pin = build_index['compressed_text'].get(name)
        raw = (BUILD / (pin['path'] if pin else name)).read_bytes()
        return gzip.decompress(raw) if pin else raw

    source = json.loads(build_data('source/index.json'))
    index = {'schema': 'current-kinematic-trace-runtime-retention-v1', 'files': {},
             'compressed_text': {}, 'origins': {}, 'build_dependency_sha256': BUILD_SEAL}

    def append(name, raw, origin=None, compress=False):
        raw.decode('utf-8')
        if b'\0' in raw or raw.startswith((b'\x7fELF', b'!<arch>')):
            raise ValueError('Non-text payload')
        logical = name
        if name in index['origins']:
            if index['origins'][name]['sha256'] != sha(raw):
                raise ValueError('Conflicting duplicate retained input')
            return
        if compress or (len(raw) > 65536 and Path(name).suffix != '.py'):
            name = 'compressed/' + name + '.gz'
            index['compressed_text'][logical] = {'path': name, 'raw_sha256': sha(raw), 'raw_bytes': len(raw)}
            stored = gzip.compress(raw, compresslevel=6, mtime=0)
        else:
            stored = raw
        path = HERE / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(stored)
        index['files'][name] = {'bytes': len(stored), 'sha256': sha(stored)}
        if origin is not None:
            index['origins'][logical] = {'path': str(origin.relative_to(ROOT)), 'bytes': len(raw), 'sha256': sha(raw)}

    def copy(origin, name, pin, compress=False):
        raw = origin.read_bytes()
        checksum = pin if isinstance(pin, str) else pin['sha256']
        if sha(raw) != checksum or (isinstance(pin, dict) and len(raw) != pin['bytes']):
            raise ValueError('Closed input changed: ' + str(origin))
        append(name, raw, origin, compress)
        return raw

    closures = {}
    for name, pin in PINS.items():
        raw = copy(WORK / name, name, pin)
        if name.endswith('-terminal-closure.json'):
            closures[name.split('/')[1].split('-')[0]] = json.loads(raw)
    for stage in ['ordinary', 'stress']:
        directory = stage + '-admission-1'
        packet = json.loads((WORK / directory / 'index.json').read_bytes())
        for name, pin in packet['files'].items():
            copy(WORK / directory / name, directory + '/' + name, pin)
        closed = closures[stage]
        for original, pins in closed['commands'].items():
            origin = ROOT / original
            logical = str(origin.relative_to(WORK))
            copy(origin, logical, pins['receipt'], True)
            copy(origin.with_suffix('.txt'), str(Path(logical).with_suffix('.txt')), pins['log'], True)
        for original, pin in closed['closed_receipts'].items():
            origin = ROOT / original
            copy(origin, str(origin.relative_to(WORK)), pin, True)
        for worker, record in closed['workers'].items():
            for original, pin in record['files'].items():
                origin = ROOT / original
                copy(origin, str(origin.relative_to(WORK)), pin, True)
            specification_name = 'results/runs/' + worker + '/specification.json'
            specification = json.loads((WORK / specification_name).read_bytes())
            request = specification['request']
            for key in ['numerical_card', 'physical_card', 'bootstrap', 'measurement']:
                name = 'inputs/' + request[key]
                raw = copy(WORK / 'source' / name, name, source['files'][name])
                spec_key = {'bootstrap': 'bootstrap_pin', 'measurement': 'point_pin'}.get(key, key)
                if {'bytes': len(raw), 'sha256': sha(raw)} != specification[spec_key]:
                    raise ValueError('Worker literal card/point differs from sealed input')
    for label, (directory, pin) in ANALYSES.items():
        origin = ROOT / directory
        packet = json.loads(copy(origin / 'index.json', 'analysis/' + label + '/index.json', pin))
        for name, item in packet['files'].items():
            copy(origin / name, 'analysis/' + label + '/' + name, item,
                 compress=Path(name).suffix in {'.json', '.log'})
    summary = {'schema': 'current-kinematic-trace-runtime-summary-v1', 'closed_inputs': PINS,
               'analysis_indices': {key: value[1] for key, value in ANALYSES.items()},
               'build_sibling': '../current-kinematic-trace-build-qualification-1',
               'build_sibling_sha256': BUILD_SEAL, 'ordinary_workers': 12, 'ordinary_outer_entries': 24,
               'ordinary_observation_parity_groups': 4, 'stress_workers': 18, 'stress_outer_entries': 36,
               'stress_terminal_exit_code': 1, 'stress_failed_contribution_groups': 6,
               'stress_equal_observation_bytes_promoted_to_success': False,
               'independent_accuracy_qualified': False, 'stability_qualified': False,
               'performance_qualified': False, 'capture_native_executions': 0,
               'external_binary_bytes_read_or_retained': False}
    append('qualification-summary.json', (json.dumps(summary, indent=2, sort_keys=True) + '\n').encode())
    for name in ['README.md', 'capture.py', 'verify.py', 'worker_checks.py']:
        raw = (HERE / name).read_bytes()
        index['files'][name] = {'bytes': len(raw), 'sha256': sha(raw)}
    with (HERE / 'evidence-index.json').open('x') as stream:
        json.dump(index, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'payloads': len(index['files']), 'stored_bytes': sum(x['bytes'] for x in index['files'].values()),
                      'raw_origin_bytes': sum(x['bytes'] for x in index['origins'].values()),
                      'native_executions': 0}, indent=2))


if __name__ == '__main__':
    main()
