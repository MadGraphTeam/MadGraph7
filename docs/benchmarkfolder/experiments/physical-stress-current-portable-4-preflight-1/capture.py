#!/usr/bin/env python3
"""Capture only root-confirmed closed preparation4/preflight4; never run codegen/native."""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
HERE = Path(__file__).resolve().parent
BASE = ROOT / 'docs/benchmarkfolder/experiments'
REPORT = BASE / 'physical-stress-current-qualification-4'
WORK = ROOT / '.tirust-build/physical-stress-current-4'
SOURCE = BASE / 'physical-stress-current-supervisor-4'
PROPOSAL = BASE / 'physical-stress-current-source-proposal-1'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'capture': False, 'native_execution': False, 'requires_root_terminal_confirmation': True}))
        return
    assert Path.cwd() == ROOT and not (HERE / 'evidence-index.json').exists()
    assert not (REPORT / 'stage-run.json').exists() and not (REPORT / 'runs').exists()
    for stage in ('prepare', 'preflight'):
        record = json.loads((REPORT / ('stage-' + stage + '.json')).read_bytes())
        assert record['status'] == 'passed' and record['exception'] is None
    initial = {str(p.relative_to(REPORT)): {'sha256': sha(p.read_bytes()), 'size': p.stat().st_size}
               for p in sorted(REPORT.rglob('*')) if p.is_file()}
    assert not any(p.is_symlink() for p in REPORT.rglob('*'))
    index = {'schema_version': 1, 'files': {}, 'compressed_text': {},
             'scope': 'Closed current-source preparation and preflight only; amplitudes pending.'}
    for name in ('capture.py', 'verify_evidence.py', 'README.md'):
        raw = (HERE / name).read_bytes()
        index['files'][name] = {'kind': 'file', 'sha256': sha(raw), 'size': len(raw)}

    def append(name, raw):
        raw.decode('utf-8'); assert b'\0' not in raw
        original_name = name
        if len(raw) > 65536 and Path(name).suffix not in ('.py',):
            name = 'compressed/' + name + '.gz'
            index['compressed_text'][original_name] = {'path': name, 'raw_sha256': sha(raw), 'raw_bytes': len(raw)}
            raw = gzip.compress(raw, compresslevel=6, mtime=0)
        path = HERE / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
        index['files'][name] = {'kind': 'file', 'sha256': sha(raw), 'size': len(raw)}

    def save(name, data):
        append(name, (json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())

    for name, item in initial.items():
        raw = (REPORT / name).read_bytes(); assert sha(raw) == item['sha256']
        append('closed/' + name, raw)
    for prefix, directory in (('source', SOURCE), ('proposal', PROPOSAL)):
        original = json.loads((directory / 'source-index.json').read_bytes())
        append(prefix + '/source-index.json', (directory / 'source-index.json').read_bytes())
        for name, item in original['files'].items():
            raw = (directory / name).read_bytes()
            assert len(raw) == item['size'] and sha(raw) == item['sha256']
            append(prefix + '/' + name, raw)
    for name, checksum in json.loads((PROPOSAL / 'input-bindings.json').read_bytes()).items():
        raw = (ROOT / name).read_bytes(); assert sha(raw) == checksum
        append('inputs/' + name, raw)
    for source in ('baseline', 'candidate'):
        receipt = json.loads((REPORT / ('materialized-' + source + '.json')).read_bytes())
        raw = (WORK / ('mg-' + source) / 'bin/madgraph').read_bytes()
        assert sha(raw) == receipt['files']['bin/madgraph']['sha256']
        append('entrypoints/' + source + '-madgraph.py', raw)
    plan = json.loads((PROPOSAL / 'proposal.json').read_bytes())
    retained = {}; maps = {}
    for case in plan['cases']:
        cid = case['case']['id']
        for source in ('baseline', 'candidate'):
            label = source + '-' + cid
            directory = REPORT / 'builds' / label
            generation = json.loads((directory / 'generation.json').read_bytes())
            build = json.loads((directory / 'build.json').read_bytes())
            output = WORK / 'exports' / source / cid
            process = output / case['process_subdirectory']
            retained[label] = {}
            wanted = dict(generation['scientific_sources'])
            for name in ('Cards/param_card.dat', 'Cards/MadLoopParams.dat', 'Cards/proc_card_mg5.dat'):
                if name in generation['all_generation_files']:
                    wanted[name] = generation['all_generation_files'][name]['sha256']
            for name, checksum in sorted(wanted.items()):
                path = output / name; raw = path.read_bytes(); assert sha(raw) == checksum
                retained_name = 'generated/' + label + '/' + name
                append(retained_name, raw)
                retained[label][name] = {'sha256': checksum, 'retained': retained_name,
                                        'symlink_target_metadata': str(path.readlink()) if path.is_symlink() else None}
            append('masses/' + label + '.dat', (process / 'masses.dat').read_bytes())
            map_raw = (directory / 'stress.map').read_bytes(); loads = []
            for line in map_raw.decode().splitlines():
                if not line.startswith('LOAD '):
                    continue
                spelling = line[5:]; path = Path(spelling)
                if not path.is_absolute(): path = process / path
                resolved = str(path.resolve())
                assert resolved in build['actual_link_loads'] and build['bound_inputs_before_after'][resolved] == build['actual_link_loads'][resolved]
                loads.append({'spelling': spelling, 'resolved': resolved, 'sha256': build['actual_link_loads'][resolved]})
            maps[label] = {'map_sha256': sha(map_raw), 'loads': loads,
                           'scope': 'Resolve recorded LOAD spellings; binary bytes remain bound by original build/preflight receipts.'}
    save('retained-generated.json', retained)
    save('link-map-resolution.json', maps)
    save('retention.json', {'scope': 'preparation-and-preflight-only', 'amplitude_evaluations': 0,
                           'capture_sha256': sha(Path(__file__).read_bytes()), 'closed_files': initial,
                           'source_index_sha256': sha((SOURCE / 'source-index.json').read_bytes()),
                           'root_terminal_admission': 'Root observed preparation25421 and preflight95469 terminal0 and admitted this capture.',
                           'retention_plan_index_sha256': '8acb41e2c90b79a4aae34468b75aca96c08d23a05d80326da0e7ea161bb6ee76',
                           'failed_predecessors_remain_separate': ['physical-stress-current-preparation-failure-1',
                                                                  'physical-stress-current-preparation-failure-2',
                                                                  'physical-stress-current-preparation-failure-3']})
    for name, item in initial.items():
        assert sha((REPORT / name).read_bytes()) == item['sha256'], 'Original closed output changed during retention'
    spec = importlib.util.spec_from_file_location('portable_stress_preflight_verifier', HERE / 'verify_evidence.py')
    verifier = importlib.util.module_from_spec(spec); spec.loader.exec_module(verifier)

    def reader(name):
        if name == 'evidence-index.json': return (json.dumps(index, sort_keys=True, indent=2) + '\n').encode()
        return (HERE / name).read_bytes()

    proof = verifier.verify(read_bytes=reader, derive_summary=True)
    save('qualification-summary.json', proof['summary'])
    verifier.verify(read_bytes=reader)
    # Index is the final append-only publication point.
    with (HERE / 'evidence-index.json').open('xb') as stream:
        stream.write((json.dumps(index, sort_keys=True, indent=2) + '\n').encode())
    print(json.dumps(verifier.verify(HERE), sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
