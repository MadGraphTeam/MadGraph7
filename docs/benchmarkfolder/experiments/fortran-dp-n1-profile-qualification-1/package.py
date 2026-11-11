"""Prepare a private text-only capsule; no native or public publication work."""
import difflib
import gzip
import hashlib
import json
from pathlib import Path
import re

ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
BASE = ROOT / 'docs/benchmarkfolder/experiments'
EXECUTION = BASE / 'fortran-dp-n1-profile-execution-2'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    payloads = {}
    origins = {}

    def add(name, raw, origin):
        raw.decode('utf8')
        if b'\0' in raw or raw.startswith((b'\x7fELF', b'!<arch>')):
            raise ValueError('Native/nontext payload: ' + name)
        if name in payloads and payloads[name] != raw:
            raise ValueError('Payload collision: ' + name)
        payloads[name] = raw
        origins[name] = {'recorded_origin': origin, 'sha256': sha(raw), 'bytes': len(raw)}

    for number in (1, 2):
        source = BASE / ('fortran-dp-n1-profile-proposal-' + str(number))
        for path in sorted(source.rglob('*')):
            if path.is_file():
                add('source/source' + str(number) + '/' + str(path.relative_to(source)),
                    path.read_bytes(), str(path.relative_to(ROOT)))
        for relative in json.loads((source / 'source-index.json').read_text())['support']:
            add('support/' + relative, (ROOT / relative).read_bytes(), relative)
    for prefix, source in [('results/failed-build1', BASE / 'fortran-dp-n1-profile-execution-1'),
                           *[('results/' + name, EXECUTION / name) for name in ('build', 'parity', 'profile')]]:
        for path in sorted(source.rglob('*')):
            if path.is_file():
                add(prefix + '/' + str(path.relative_to(source)), path.read_bytes(), str(path.relative_to(ROOT)))
    pins = json.loads(payloads['source/source2/input-pins.json'])
    original = Path(pins['output'])
    for relative in pins['generated_sources']:
        path = original / relative
        add('donor/' + relative, path.read_bytes(), str(path))
    # Retain textual numerical inputs, never the executable/static/shared objects.
    for relative in ('generation.json', 'compilation.json', 'Cards/param_card.dat', *pins['numerical_cards']):
        path = original / relative
        add('donor/' + relative, path.read_bytes(), str(path))
    closed = Path(pins['closed_observation']).parent
    for logical, path in [('inputs/closed-observation.json', closed / 'observation.json'),
                          ('inputs/closed-warmup.dat', closed / 'warmup.dat'),
                          ('inputs/closed-input.json', Path(pins['closed_input'])),
                          ('inputs/points.dat', Path(pins['points'])),
                          ('inputs/points.json', Path(pins['points']).with_suffix('.json'))]:
        add(logical, path.read_bytes(), str(path))
    for phase in ('parity', 'profile'):
        for lane in ('baseline', 'observed-0', 'observed-1', 'observed-2'):
            name = 'results/' + phase + '/' + lane + '/warmup.json'
            raw = json.loads(payloads[name])['raw_outputs']['warmup.dat'].encode()
            add('learning/' + phase + '/' + lane + '.dat', raw, name + '#raw_outputs.warmup.dat')
    tools = json.loads(payloads['source/source2/toolchain-pins.json'])
    for recorded in tools['donor_lineage']:
        path = ROOT / recorded
        add('environment-donor/' + recorded, path.read_bytes(), recorded)
    for number, folder in [('build-parity', 'fortran-dp-n1-profile-independent-audit-2'),
                           ('profile', 'fortran-dp-n1-profile-independent-profile-audit-2')]:
        source = ROOT / '.tirust-build' / folder
        for path in sorted(source.rglob('*')):
            if path.is_file():
                add('audits/' + number + '/' + str(path.relative_to(source)),
                    path.read_bytes(), str(path.relative_to(ROOT)))
    # Capture symlink resolution once as provenance text. The verifier never
    # resolves these historical names or opens the referenced native inputs.
    resolutions = {}
    build = json.loads(payloads['results/build/result.json'])
    for variant, row in build['variants'].items():
        process = Path(row['output']) / 'SubProcesses/P0_uux_z'
        for name in ('makefile', 'private-link.map', 'check_sa.f', 'TIR_interface.f',
                     'TIRUST_interface.f', 'mtirust.f90', 'qp_profile.f90'):
            path = process / name
            add('compiled/' + variant + '/' + name, path.read_bytes(), str(path))
        aliases = []
        for spelling in re.findall(r'^LOAD\s+(.+)$', (process / 'private-link.map').read_text(), re.M):
            resolved = str((process / spelling).resolve())
            aliases.append({'spelling': spelling, 'resolved': resolved,
                            'sha256': row['actual_link_loads'][resolved]})
        runtime = []
        for line in row['runtime']['listing'].splitlines():
            line = line.strip()
            if not line or line.startswith('linux-vdso.so.1 '):
                continue
            match = re.fullmatch(r'(?:\S+\s+=>\s+)?(/[^\s()]+)\s+\(0x[0-9a-fA-F]+\)', line)
            if match is None:
                raise ValueError('Unexpected runtime listing')
            spelling = match[1]
            resolved = str(Path(spelling).resolve())
            runtime.append({'spelling': spelling, 'resolved': resolved,
                            'sha256': row['runtime']['files'][resolved]})
        resolutions[variant] = {'map_aliases': aliases, 'runtime_aliases': runtime}
    add('link-runtime-resolution.json', (json.dumps(resolutions, indent=2, sort_keys=True) + '\n').encode(),
        'Pure preparation of retained link/runtime spelling aliases; native bytes not copied')
    replay = BASE / 'fortran-qp-output-bulk-physical-qualification-1/replay_warmup.py'
    add('replay_warmup.py', replay.read_bytes(), str(replay.relative_to(ROOT)))
    retention = {'schema': 'n1-dp-profile-private-retention-v1', 'scope': 'Private review only; not published.',
                 'intended_future_public_target': 'docs/benchmarkfolder/experiments/fortran-dp-n1-profile-qualification-1',
                 'binary_bytes_retained': False, 'native_executions_during_capture': 0,
                 'performance_acceptance': False, 'speedup_claimed': False,
                 'current_NinjaRS_comparison': False, 'origins': origins,
                 'query_count_scope': {'recorded_BUILD_actions': 122,
                     'additional_compiler_policy_captures': 4,
                     'implied_additional_make_dryruns': 12, 'implied_cached_FC_version_queries': 4,
                     'all_make_internal_children_individually_recorded': False}}
    add('retention.json', (json.dumps(retention, indent=2, sort_keys=True) + '\n').encode(), 'Private packaging metadata')
    # Verifier/tests/package source are indexed too. No self-indexing index.
    for name in ('verify_evidence.py', 'test_verify.py', 'package.py', 'README.md'):
        path = HERE / name
        if path.is_file():
            add(name, path.read_bytes(), 'Private capsule helper ' + name)
    files = {}
    logical = {}
    for name, raw in sorted(payloads.items()):
        packed = len(raw) > 4096 and name not in ('verify_evidence.py', 'test_verify.py', 'package.py', 'README.md', 'replay_warmup.py')
        physical = 'compressed/' + name + '.gz' if packed else name
        stored = gzip.compress(raw, compresslevel=9, mtime=0) if packed else raw
        path = HERE / physical
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(stored)
        files[physical] = {'sha256': sha(stored), 'bytes': len(stored)}
        logical[name] = {'path': physical, 'sha256': sha(raw), 'bytes': len(raw),
                         'encoding': 'gzip+utf8' if packed else 'utf8'}
    index = {'schema': 'n1-dp-profile-portable-evidence-v1', 'scope': 'Private text-only retained experiment.',
             'native_bytes_retained': False, 'performance_acceptance': False, 'speedup_claimed': False,
             'current_NinjaRS_comparison': False, 'files': files, 'payloads': logical}
    (HERE / 'evidence-index.json').write_text(json.dumps(index, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'logical_text_files': len(logical), 'stored_bytes': sum(v['bytes'] for v in files.values()),
                      'raw_bytes': sum(v['bytes'] for v in logical.values()),
                      'index_sha256': sha((HERE / 'evidence-index.json').read_bytes())}))


if __name__ == '__main__':
    main()
