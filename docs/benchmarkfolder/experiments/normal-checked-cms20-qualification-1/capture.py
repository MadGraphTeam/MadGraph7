"""Retain explicit closed text evidence; never build, load, or evaluate native code."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
HERE = Path(__file__).resolve().parent
SOURCES = {
    'science': ('.tirust-build/kinematic-normal-checked-source-1', '7dab8a04296a44e6eef1cf017bee751491220b77804ebe2a34faa328f85cb2c9'),
    'controller1': ('.tirust-build/kinematic-normal-checked-controller-1', 'fd482474221715397a0dff7b47244ef25a07f077e79b4b8a47c5e3beca398c03'),
    'controller2': ('.tirust-build/kinematic-normal-checked-controller-2', '88cc660d0deecc3b8f87eba23bc10deed729a048cb4bddfcbe3ca4265f8830da'),
    'plan': ('.tirust-build/kinematic-normal-checked-proposal-1', '9387a2858df75bf34d0940121da6e006fe7648a9d63d241276ba05f94487f7b4'),
}
WORK = {number: ROOT / ('.tirust-build/kinematic-normal-checked-executor-' + number) / 'results'
        for number in ('1', '2')}
CLOSURES = {
    'results1/build': '3071f3887244a34ebdf134e3867484cae6ddec9a6623309d3668d8817d26f09a',
    'results1/run': '754078acb9700ffa10b9c040bc6c0de86784e4da66adaaef0e8fdcb8441985af',
    'results2/run': '046d83232bd40235f2401341c0806fb9446599b321da23e72b663b9430d68aa1',
}
COMPARISON = '8ff34fa6a4648ebb84cab4b2cb8a0a1d2f3e25591c42b55aa8f398b308a27465'
DEPENDENCIES = {
    'kinematic-invariant-amplitude-qualification-1': 'c04a73c86f6aa3db59e6d5cafd2fb9edf92dd2dbd6d3e418cab462ebbb5327d4',
    'matched-native-input-qualification-1': 'eacced6cc887ee586d15c4ceade8ac1f089fec0a36b84708f12086d05ba2f8a0',
    'kinematic-legacy-reference-adjudication-1': '248bd61e987ac97c81e5397489eb4322ecfbf96cf85a42c84607aef2c013500e',
    'cms-universal-qed-pole-audit-1': '1565ac55fe6aeeefbd4f660455c422c4b954737a434d27138cbe8433d9534b77',
    'cms-universal-qed-single-pole-audit-1': '99603d41d662e90f071b8052eb4303bcba8af451c5c96654febc276e620ad11e',
}
EXTRAS = {
    'results1/build-terminal-review.json': '15f3c3da49b169468f79a8158e632001d736fa543fa09aed4175022753fd54b3',
    'results1/root-build-review.json': '6039f1d2313d9d774cc7ca714fabd62fa8adfb05213970acfedfa7efd0916867',
    'results1/run-terminal-review.json': '66935caf46ad48690f1454fb7c5e7e6398d9d2c1ebc0a897654919506e2a3eec',
    'results1/retrospective-input-lifecycle-diagnostic.json': 'd4cfd3362ce4cfac92d0f536220c1c0d38fe12e9a692730d36c3a6869940530a',
    'results1/retrospective-ordinary-control-pole-diagnostic.json': '96545b0fdfe1789b9ae1c93f4db5d04748e7596d8bc4994fd81b05b5a74408df',
    'results2/run-terminal-review.json': '27f25353bc8bfbdae0d9acb6bd2a2342f538991ca210302493ca4e5b9854c6c4',
    'results2/root-run-review.json': '226918bd2b26c740d493473aa12384fd9a5859da04c5e14a03e17b23ec954ff5',
}
PREFLIGHTS = {
    'kinematic-normal-checked-source-1-preflight.json': '1c2d151e9713a9798b42b9cbd1b6269f08895c3c249c897b2aa3e1404f95d4ca',
    'kinematic-normal-checked-controller-1-preflight.json': 'f29bda38d29d42f9051cb0279d5289796385ca5e257dc56d51dbddde39c3bf01',
    'kinematic-normal-checked-controller-1-cli-preflight.json': '69239d9d3adf02f807e2649c548cee8d117670545f330a80b59fbebfdd1c9b0e',
    'kinematic-normal-checked-controller-2-retrospective-preseal.json': 'f0e02a14b4b3f96c787d60d7ff6a7cf1d35741aeb3fd33a0042db88f2a69b3e0',
    'kinematic-normal-checked-controller-2-cli-preflight.json': '9ea02af36d9628706b7cef79ed12cf87ed4cb5ce5ce773e538ae96965c73f521',
    'kinematic-normal-checked-controller-2-parent-preflight.json': '9ea02af36d9628706b7cef79ed12cf87ed4cb5ce5ce773e538ae96965c73f521',
    'kinematic-normal-checked-controller-2-root-preflight.json': '9ea02af36d9628706b7cef79ed12cf87ed4cb5ce5ce773e538ae96965c73f521',
}
LOCAL_FILES = ('README.md', 'capture.py', 'verify.py', 'build_checks.py', 'worker_checks.py', 'test_worker_checks.py')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pin(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def safe(name):
    require(isinstance(name, str) and name and not Path(name).is_absolute() and
            '\\' not in name and all(p not in ('', '.', '..') for p in name.split('/')),
            'Unsafe retained logical path')
    return name


def collect():
    """Development/capture only: read exact sealed lists, returning text bytes."""
    index = {'schema': 'private-normal-checked-cms20-retention-v1',
             'source_indices': {k: v[1] for k, v in SOURCES.items()},
             'closed_stages': CLOSURES, 'dependencies': DEPENDENCIES,
             'files': {}, 'compressed_text': {}, 'origins': {}}
    payloads = {}

    def retain(path, logical, expected):
        safe(logical)
        raw = path.read_bytes(); actual = pin(raw)
        want = {'sha256': expected} if isinstance(expected, str) else expected
        require(all(actual[k] == v for k, v in want.items()), 'Changed closed text: ' + str(path))
        raw.decode('utf-8')
        require(b'\0' not in raw and not raw.startswith((b'\x7fELF', b'!<arch>')), 'Nontext payload')
        require(logical not in payloads or payloads[logical] == raw, 'Conflicting duplicate payload')
        payloads[logical] = raw
        index['origins'][logical] = {'path': str(path.relative_to(ROOT)), **actual}
        return raw

    for prefix, (origin, sha) in SOURCES.items():
        directory = ROOT / origin
        document = json.loads(retain(directory / 'index.json', prefix + '/index.json', sha))
        for name, identity in document['files'].items():
            retain(directory / safe(name), prefix + '/' + name, identity)
    for label, sha in CLOSURES.items():
        prefix, stage = label.split('/')
        directory = WORK[prefix[-1]]
        closed = json.loads(retain(directory / (stage + '-closure.json'), label + '-closure.json', sha))
        for name, identity in closed['payloads'].items():
            retain(directory / safe(name), prefix + '/' + name, identity)
    for name, sha in EXTRAS.items():
        prefix, tail = name.split('/', 1)
        retain(WORK[prefix[-1]] / tail, name, sha)
    for name, sha in PREFLIGHTS.items():
        retain(ROOT / '.tirust-build' / name, 'preflights/' + name, sha)
    comparison = payloads['results2/checked-comparison.json']
    require(pin(comparison)['sha256'] == COMPARISON, 'Closed scientific comparison differs')
    report = json.loads(comparison)
    helpers = report['phases']['0']['host_qualification']['geometry'][0]['proof']['source_evidence']
    require(len(helpers) == 14, 'Geometry helper set differs')
    for phase in report['phases'].values():
        for row in phase['host_qualification']['geometry']:
            require(row['proof']['source_evidence'] == helpers, 'Geometry ancestry differs across groups')
    for origin, identity in helpers.items():
        retain(ROOT / safe(origin), 'helpers/' + origin, identity)
    index['geometry_helper_origins'] = helpers
    for name, sha in DEPENDENCIES.items():
        require(pin((HERE.parent / name / 'evidence-index.json').read_bytes())['sha256'] == sha,
                'Required ancestor index differs: ' + name)
    return index, payloads


def main():
    require(sys.argv[1:] == ['--execute'] and Path.cwd() == ROOT, 'Explicit repository-root text capture required')
    require(not (HERE / 'evidence-index.json').exists(), 'Already sealed; never overwrite')
    index, payloads = collect()
    for logical, raw in sorted(payloads.items()):
        name = logical if logical.endswith('/index.json') else 'compressed/' + logical + '.gz'
        compressed = name != logical
        stored = gzip.compress(raw, compresslevel=6, mtime=0) if compressed else raw
        destination = HERE / name; destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(stored)
        index['files'][name] = pin(stored)
        if compressed:
            index['compressed_text'][logical] = {'path': name, 'raw_bytes': len(raw),
                                                'raw_sha256': pin(raw)['sha256']}
    for name in LOCAL_FILES:
        index['files'][name] = pin((HERE / name).read_bytes())
    with (HERE / 'evidence-index.json').open('x') as stream:
        json.dump(index, stream, sort_keys=True, indent=2); stream.write('\n')
    print(json.dumps({'payloads': len(index['files']), 'retained_origins': len(payloads),
                      'stored_bytes': sum(v['bytes'] for v in index['files'].values()),
                      'native_actions': 0}, sort_keys=True))


if __name__ == '__main__':
    main()
