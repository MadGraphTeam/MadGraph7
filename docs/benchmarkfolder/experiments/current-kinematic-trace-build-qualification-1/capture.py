"""Retain only explicitly closed trace-build text; no directory discovery/native calls."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
HERE = Path(__file__).resolve().parent
WORK = ROOT / '.tirust-build/kinematic-provenance-current-executor-1'
PINS = {
    'results/build-terminal-closure.json': '564688cfb10fa7a05f4349600209299751a91159632047377b59ebe1d7fe2185',
    'results/root-build-review.json': '36c3ec42425ee8cd96c72a8ecee9efffc95c00f61e7b0462bc2dbdc5e1204559',
    'results/build-postterminal-ownership.json': 'e4849a7dbce603f4659ce878865e4bc3bd8bb8b2c16a3bd7aa410af7c648770f',
    'source/index.json': '7ea709b48d8de462b754fabfb47dcbd5d21d7847458382f5844190028e8c1bed',
    'results/derived/index.json': '25c2114756ed3eb968c311945a27d175c5c5a0af54d4ae21158abc8ed42a4184',
    'build-admission-1/index.json': 'c30fefa33c13dc04cbeed5b12a6a44284d1fc4858c12aa569382ea8a8f8bf813',
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    if sys.argv[1:] != ['--execute']:
        raise SystemExit('Explicit text-retention invocation required: --execute')
    if Path.cwd() != ROOT or (HERE / 'evidence-index.json').exists():
        raise ValueError('Wrong working directory or already retained packet')
    index = {'schema': 'current-kinematic-trace-build-retention-v1', 'files': {},
             'compressed_text': {}, 'origins': {},
             'scope': 'Closed build only; explicit 258-command list; no ordinary/stress execution entries'}

    # Same deterministic UTF-8/gzip pattern as the existing portable preflight
    # packets, driven here exclusively by sealed file maps and closed receipts.
    def append(name, raw, origin=None, compress=False):
        raw.decode('utf-8')
        if b'\0' in raw or raw.startswith((b'\x7fELF', b'!<arch>')):
            raise ValueError('Non-text payload')
        original = name
        if compress or (len(raw) > 65536 and Path(name).suffix != '.py'):
            name = 'compressed/' + name + '.gz'
            index['compressed_text'][original] = {
                'path': name, 'raw_sha256': sha(raw), 'raw_bytes': len(raw)}
            stored = gzip.compress(raw, compresslevel=6, mtime=0)
        else:
            stored = raw
        target = HERE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(stored)
        index['files'][name] = {'sha256': sha(stored), 'bytes': len(stored)}
        if origin is not None:
            index['origins'][original] = {'path': str(origin.relative_to(ROOT)),
                                          'sha256': sha(raw), 'bytes': len(raw)}

    def copy(origin, name, pin=None, compress=False):
        raw = origin.read_bytes()
        if pin is not None:
            expected = pin if isinstance(pin, str) else pin['sha256']
            if sha(raw) != expected or (isinstance(pin, dict) and len(raw) != pin['bytes']):
                raise ValueError('Closed input changed: ' + str(origin))
        append(name, raw, origin, compress)
        return raw

    for name, pin in PINS.items():
        copy(WORK / name, name, pin)
    closure = json.loads((WORK / 'results/build-terminal-closure.json').read_bytes())
    if len(closure['commands']) != 258 or closure['amplitude_evaluations'] != 0:
        raise ValueError('Unexpected closed scope')
    for directory in ['source', 'results/derived', 'build-admission-1']:
        packet = json.loads((WORK / directory / 'index.json').read_bytes())
        for name, pin in packet['files'].items():
            copy(WORK / directory / name, directory + '/' + name, pin,
                 compress=Path(name).suffix == '.log')
    for receipt_name, pins in closure['commands'].items():
        # Never glob execution/ or inspect an entry not explicitly closed here.
        origin = ROOT / receipt_name
        relative = str(origin.relative_to(WORK))
        copy(origin, relative, pins['receipt'], compress=True)
        copy(origin.with_suffix('.txt'), str(Path(relative).with_suffix('.txt')),
             pins['log'], compress=True)
    for label, pin in closure['builds'].items():
        base = 'results/builds/' + label
        for name in ['before-compile.json', 'prelink.json', 'build.json']:
            copy(WORK / base / name, base + '/' + name,
                 pin['receipt'] if name == 'build.json' else None, compress=True)
        name = 'results/build-admission-1/' + label + '-runtime.json'
        copy(WORK / name, name)
    for name in ['results/build-admission-1/before.json', 'results/build-admission-1/after.json',
                 'results/build-admission-1/result.json', 'results/stage-build.json']:
        copy(WORK / name, name)
    derived = json.loads((WORK / 'results/derived/index.json').read_bytes())
    copy(WORK / 'results/generation.json', 'results/generation.json', derived['generated_source_receipt'])
    helpers = json.loads((WORK / 'source/helper-bindings.json').read_bytes())
    for name, pin in helpers['files'].items():
        copy(ROOT / name, 'helpers/' + name, pin)
    bindings = json.loads((WORK / 'build-admission-1/bindings.json').read_bytes())
    for name, pin in bindings['evidence'].items():
        copy(ROOT / name, 'provenance/' + name, pin)
    context_path = ROOT / '.tirust-build/kinematic-provenance-current-proposal-1/proposal/context-bindings.json'
    context = json.loads(context_path.read_bytes())
    for key in ['ew_compilation', 'historical_packet', 'campaign_manifest']:
        pin = context[key]
        copy(ROOT / pin['path'], 'provenance/' + pin['path'], pin)
    parent = ROOT / '.tirust-build/kinematic-provenance-current-proposal-1/proposal/index.json'
    source = json.loads((WORK / 'source/index.json').read_bytes())
    copy(parent, 'ancestry/parent-proposal-index.json', source['parent_proposal_index_sha256'])
    summary = {'schema': 'current-kinematic-trace-build-summary-v1',
               'command_count': 258, 'built_executables': sorted(closure['builds']),
               'actual_loader_query_count': 4, 'providers_per_executable': 15,
               'amplitude_evaluations': 0, 'ordinary_or_stress_evidence_retained': False,
               'observational_parity_qualified': False, 'accuracy_qualified': False,
               'performance_qualified': False, 'external_binary_bytes_retained': False,
               'external_binary_bytes_rehashed_by_capture': False,
               'closed_input_pins': PINS, 'source_files': len(source['files']),
               'derived_files': len(derived['files']), 'helper_files': len(helpers['files']),
               'selection': 'Only fixed source/derived/guard maps, closure commands, four closed build records and named provenance records'}
    append('qualification-summary.json', (json.dumps(summary, indent=2, sort_keys=True) + '\n').encode())
    for name in ['README.md', 'capture.py', 'verify.py']:
        raw = (HERE / name).read_bytes()
        index['files'][name] = {'sha256': sha(raw), 'bytes': len(raw)}
    with (HERE / 'evidence-index.json').open('x') as stream:
        json.dump(index, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'retained_payloads': len(index['files']),
                      'stored_bytes': sum(v['bytes'] for v in index['files'].values()),
                      'raw_origin_bytes': sum(v['bytes'] for v in index['origins'].values()),
                      'native_invocations': 0, 'active_execution_entries_inspected': 0}, indent=2))


if __name__ == '__main__':
    main()
