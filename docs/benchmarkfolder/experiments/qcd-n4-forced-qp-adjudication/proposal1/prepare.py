"""Source-only selection/binding. No compiler, loader or native imports."""
import hashlib
import json
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
REPORT = ROOT / 'docs/benchmarkfolder/campaigns/split-dp-terminal-1'
TRIAGE = REPORT / 'experiments/qcd-n4-target-triage-1'
IDS = [1, 61, 89, 93]


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def prepare():
    assert Path.cwd() == ROOT
    assert not (HERE / 'source-index.json').exists(), 'Frozen proposal is immutable'
    index = json.loads((TRIAGE / 'index.json').read_text())
    for name, identity in index['files'].items():
        assert sha(TRIAGE / name) == identity['sha256'], name
    lines = (REPORT / 'points/qcd_n4_uux_zzgg.dat').read_text().splitlines()
    assert lines[0] == '100' and len(lines) == 601
    blocks = {i: lines[1 + 6 * (i - 1):1 + 6 * i] for i in IDS}
    (HERE / 'points.dat').write_text('4\n' + ''.join('\n'.join(blocks[i]) + '\n' for i in IDS))
    analysis = json.loads((TRIAGE / 'analysis.json').read_text())
    selected = [row for row in analysis['target_point_results'] if row['point'] in IDS[1:]]
    assert [row['point'] for row in selected] == IDS[1:]
    save(HERE / 'selected-evidence.json', {
        'schema': 1, 'triage_index_sha256': sha(TRIAGE / 'index.json'),
        'triage_analysis_sha256': sha(TRIAGE / 'analysis.json'), 'bootstrap_point': 1,
        'selected_point_ids': IDS[1:], 'points': selected,
        'point_text_sha256': sha(REPORT / 'points/qcd_n4_uux_zzgg.dat'),
        'point_binary64_hex': {str(i): [[struct.pack('>d', float(x)).hex() for x in line.split()]
                                      for line in blocks[i]] for i in IDS}})
    lane = REPORT / 'experiments/numerical/full100-warmup-qcd_n4_uux_zzgg-1/candidate-9'
    inputs = json.loads((lane / 'candidate-inputs.json').read_text())
    export = ROOT / '.benchmark-work/tirust/campaigns/split-dp-terminal-1/outputs/candidate/qcd_n4_uux_zzgg'
    sub = export / 'SubProcesses/P0_uux_zzgg'
    sources = {}
    for name, expected in inputs['compilation']['sources'].items():
        path = export / name
        assert sha(path) == expected, name
        sources[str(path.relative_to(ROOT))] = expected
    objects = {str(path.relative_to(ROOT)): {'sha256': sha(path), 'bytes': path.stat().st_size,
               'binding': 'Current closed generated object; original compilation record binds source, not per-object bytes'}
               for path in sorted(sub.glob('*.o')) if path.name != 'check_sa.o'}
    archives = inputs['runtime']['archives']
    for name, digest in archives.items():
        assert sha(Path(name)) == digest, name
    mlcard = export / 'Cards/MadLoopParams.dat'
    assert sha(mlcard) == inputs['numerical_cards']['Cards/MadLoopParams.dat']
    bindings = {str(path.relative_to(ROOT)): sha(path) for path in (
        TRIAGE / 'index.json', TRIAGE / 'analysis.json', TRIAGE / 'inputs.json',
        REPORT / 'numerical-checkpoints/qcd-n4-1/index.json',
        REPORT / 'points/qcd_n4_uux_zzgg.dat', REPORT / 'points/qcd_n4_uux_zzgg.json',
        lane / 'candidate-inputs.json', mlcard, TRIAGE / 'source/param_card.dat')}
    save(HERE / 'dependencies.json', {
        'schema': 1,
        'binding_scope': 'Closed source and archive provenance verified; original generated object bytes newly snapshotted before any new build, not claimed historically object-hashed',
        'closed_evidence': bindings, 'generated_sources': sources, 'generated_objects': objects,
        'archives': archives, 'runtime_expected_from_closed_record': inputs['runtime']['libraries'],
        'settings': inputs['settings'],
        'recorded_process_build_command': inputs['compilation']['process_build_command']})
    (HERE / 'MadLoopParams.original.dat').write_bytes(mlcard.read_bytes())
    (HERE / 'param_card.original.dat').write_bytes((TRIAGE / 'source/param_card.dat').read_bytes())
    print(json.dumps({'selected': len(selected), 'sources_verified': len(sources),
                      'objects_snapshotted': len(objects), 'archives_verified': len(archives),
                      'triage_index': sha(TRIAGE / 'index.json'),
                      'triage_analysis': sha(TRIAGE / 'analysis.json')}, sort_keys=True))


if __name__ == '__main__':
    prepare()
