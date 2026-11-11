"""Private text-only retention of the closed O3 experiment; no native work."""
import gzip
import hashlib
import json
from pathlib import Path
import re

ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
WORK = ROOT / '.tirust-build'
EXECUTION = WORK / 'fortran-tirust-interface-o3-execution-2'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    payloads, origins = {}, {}
    def add(name, raw, origin):
        raw.decode('utf8')
        if b'\0' in raw or raw.startswith((b'\x7fELF', b'!<arch>')):
            raise ValueError('Native/nontext payload: ' + name)
        if name in payloads and payloads[name] != raw:
            raise ValueError('Payload collision: ' + name)
        payloads[name] = raw
        origins[name] = {'recorded_origin': origin, 'sha256': sha(raw), 'bytes': len(raw)}
    for number in (1, 2):
        source = WORK / ('fortran-tirust-interface-o3-proposal-' + str(number))
        for path in sorted(source.rglob('*')):
            if path.is_file():
                add('source/source' + str(number) + '/' + str(path.relative_to(source)),
                    path.read_bytes(), str(path.relative_to(ROOT)))
        for relative in json.loads((source / 'source-index.json').read_text())['support']:
            add('support/' + relative, (ROOT / relative).read_bytes(), relative)
    for name in ('build', 'n1parity', 'n1timing', 'n3parity', 'n3anchortiming'):
        folder = EXECUTION / name
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                add('results/' + name + '/' + str(path.relative_to(folder)),
                    path.read_bytes(), str(path.relative_to(ROOT)))
    pins = json.loads(payloads['source/source2/input-pins.json'])
    for case, row in pins['cases'].items():
        output = Path(row['output'])
        for name in row['generated_sources']:
            add('donor/' + case + '/' + name, (output / name).read_bytes(), str(output / name))
        for name in ('generation.json', 'compilation.json', 'Cards/param_card.dat', *row['numerical_cards']):
            add('donor/' + case + '/' + name, (output / name).read_bytes(), str(output / name))
        for name, path in [('closed-observation.json', Path(row['closed_observation'])),
                           ('closed-warmup.dat', Path(row['closed_warmup'])),
                           ('closed-input.json', Path(row['closed_input'])),
                           ('points.dat', Path(row['points'])),
                           ('points.json', Path(row['points']).with_suffix('.json'))]:
            add('inputs/' + case + '/' + name, path.read_bytes(), str(path))
    add('inputs/stock-mtirust.f90', (ROOT / 'vendor/tirust/mtirust.f90').read_bytes(), 'vendor/tirust/mtirust.f90')
    toolpins = json.loads(payloads['source/source2/toolchain-pins.json'])
    for relative in set(pins['ancestry']) | set(toolpins['donor_lineage']):
        add('ancestry/' + relative, (ROOT / relative).read_bytes(), relative)
    for name in ('build', 'n1parity', 'n1timing', 'n3parity', 'n3anchortiming'):
        folder = WORK / ('fortran-tirust-interface-o3-independent-' + name + '-audit-2')
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                add('audits/' + name + '/' + str(path.relative_to(folder)),
                    path.read_bytes(), str(path.relative_to(ROOT)))
    for name in ('build', 'n1parity', 'n1timing', 'n3anchor'):
        folder = WORK / ('fortran-tirust-interface-o3-' + name + '-independent-audit-2')
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                add('primary-audits/' + name + '/' + str(path.relative_to(folder)),
                    path.read_bytes(), str(path.relative_to(ROOT)))
    preflight = WORK / 'fortran-tirust-interface-o3-root-preflight-2.json'
    add('primary-audits/root-preflight-2.json', preflight.read_bytes(), str(preflight.relative_to(ROOT)))
    delivery = WORK / 'fortran-tirust-interface-o3-delivery-proposal-1'
    delivery_index = json.loads((delivery / 'index.json').read_text())
    for name in (*delivery_index['files'], 'index.json'):
        add('prospective/delivery/' + name, (delivery / name).read_bytes(), str((delivery / name).relative_to(ROOT)))
    resolutions = {}
    build = json.loads(payloads['results/build/result.json'])
    for key, row in build['variants'].items():
        process = Path(row['output']) / 'SubProcesses' / pins['cases'][row['case']]['subprocess']
        for name in ('makefile', 'private-link.map', 'check_sa.f', 'TIRUST_interface.f'):
            add('compiled/' + key + '/' + name, (process / name).read_bytes(), str(process / name))
        maps = []
        for spelling in re.findall(r'^LOAD\s+(.+)$', (process / 'private-link.map').read_text(), re.M):
            resolved = str((process / spelling).resolve())
            maps.append({'spelling': spelling, 'resolved': resolved, 'sha256': row['actual_link_loads'][resolved]})
        runtime = []
        for line in row['runtime']['listing'].splitlines():
            line = line.strip()
            if not line or line.startswith('linux-vdso.so.1 '):
                continue
            match = re.fullmatch(r'(?:\S+\s+=>\s+)?(/[^\s()]+)\s+\(0x[0-9a-fA-F]+\)', line)
            if match is None:
                raise ValueError('Unexpected retained runtime listing')
            spelling = match[1]
            resolved = str(Path(spelling).resolve())
            runtime.append({'spelling': spelling, 'resolved': resolved, 'sha256': row['runtime']['files'][resolved]})
        resolutions[key] = {'map_aliases': maps, 'runtime_aliases': runtime}
    add('link-runtime-resolution.json', (json.dumps(resolutions, indent=2, sort_keys=True) + '\n').encode(),
        'Pure recorded spelling resolution; native bytes not retained')
    warmup = ROOT / 'docs/benchmarkfolder/experiments/fortran-qp-output-bulk-physical-qualification-1/replay_warmup.py'
    add('replay_warmup.py', warmup.read_bytes(), str(warmup.relative_to(ROOT)))
    retention = {'schema': 'o3-private-experiment-retention-v1', 'scope': 'PRIVATE complete experiment; unpromoted.',
        'future_public_target': 'docs/benchmarkfolder/experiments/fortran-tirust-interface-o3-qualification-1',
        'production_changes': False, 'public_writes': False, 'patches_applied': 0,
        'native_bytes_retained': False, 'native_evaluations_during_capture': 0,
        'candidate_promoted': False, 'performance_acceptance': False, 'optimization_benefit_claimed': False,
        'prospective_delivery_is_separate': True, 'origins': origins,
        'query_count_scope': {'recorded_BUILD_actions': 260, 'recorded_object_compiles': 8,
            'recorded_links': 4, 'recorded_prelink_queries': 204, 'recorded_make_queries': 28,
            'recorded_version_queries': 8, 'recorded_nm_queries': 4, 'recorded_BUILD_loader_queries': 4,
            'recorded_physics_loader_queries': 16, 'all_make_internal_children_individually_traced': False}}
    add('retention.json', (json.dumps(retention, indent=2, sort_keys=True) + '\n').encode(), 'Private retention metadata')
    for name in ('verify_evidence.py', 'test_verify.py', 'package.py', 'README.md'):
        add(name, (HERE / name).read_bytes(), 'Private capsule helper ' + name)
    files, logical = {}, {}
    for name, raw in sorted(payloads.items()):
        packed = len(raw) > 4096 and name not in ('verify_evidence.py', 'test_verify.py', 'package.py', 'README.md', 'replay_warmup.py')
        physical = 'compressed/' + name + '.gz' if packed else name
        stored = gzip.compress(raw, compresslevel=9, mtime=0) if packed else raw
        path = HERE / physical
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(stored)
        files[physical] = {'sha256': sha(stored), 'bytes': len(stored)}
        logical[name] = {'path': physical, 'sha256': sha(raw), 'bytes': len(raw), 'encoding': 'gzip+utf8' if packed else 'utf8'}
    index = {'schema': 'o3-portable-experiment-evidence-v1', 'scope': 'Private complete O3 experiment; unpromoted.',
        'native_bytes_retained': False, 'performance_acceptance': False, 'candidate_promoted': False,
        'optimization_benefit_claimed': False, 'patches_applied': 0, 'files': files, 'payloads': logical}
    (HERE / 'evidence-index.json').write_text(json.dumps(index, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'payloads': len(logical), 'stored_bytes': sum(v['bytes'] for v in files.values()),
        'decoded_bytes': sum(v['bytes'] for v in logical.values()), 'index_sha256': sha((HERE / 'evidence-index.json').read_bytes())}))


if __name__ == '__main__':
    main()
