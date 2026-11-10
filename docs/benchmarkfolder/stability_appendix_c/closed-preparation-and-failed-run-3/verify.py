"""Portable closed-receipt verification; never follows recorded host paths."""
import hashlib
import json
import math
from pathlib import Path

CASES = ('qcd_n2_gg_ttx', 'qcd_n3_uux_zgg', 'ew_n2_epem_mupmum', 'qcd_n4_uux_zzgg')
BACKENDS = (1, 6, 7, 8, 9)
PREP_SHA = '4d69eebf9fb0f9d2a3cb4108ff6a98dbef5343eb760a3b9a81b066e7f6e2756d'
RUN_SHA = 'e0ca85280c994257448eee59adc411657b3c521f27c8d022ae7da284a22156c8'
SOURCE_SHA = '1a98e885702ffa1704cf5c8a025d03488e80ae00b590de6ffbfadad8157f8c93'
PROTOCOL_SHA = '77c38366107a955000dd67c2acf89f1e2f683b5155beee83c32ac623f291f9e2'


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def physical_points(blob, metadata, compiled_masses):
    """Recheck saved vectors, physical conservation/masses and strict cuts."""
    inputs = metadata['inputs']
    masses = inputs['incoming_masses'] + inputs['masses']
    assert masses == compiled_masses and inputs['count'] == 100
    assert inputs['energy'] == 1000 and inputs['pt_min_GeV'] == 50
    assert inputs['delta_R_min'] == .5 and inputs['strict_cut_boundary'] is True
    assert digest(blob) == metadata['sha256']
    rows = blob.decode().splitlines()
    assert rows[0] == '100' and len(rows) == 1 + 100 * len(masses)
    vectors = [list(map(float, row.split())) for row in rows[1:]]
    assert len(metadata['points']) == 100
    minimum_pt = minimum_dr = math.inf
    for i, point in enumerate(metadata['points']):
        momenta = vectors[i * len(masses):(i + 1) * len(masses)]
        assert point['id'] == 'stability-%06d' % i
        assert point['momenta'] == momenta and point['sqrt_s'] == 1000
        assert point['masses'] == inputs['masses'] and point['incoming_masses'] == inputs['incoming_masses']
        assert all(len(p) == 4 and p[0] > 0 and all(math.isfinite(x) for x in p) for p in momenta)
        total = [math.fsum(p[k] for p in momenta[:2]) for k in range(4)]
        assert all(abs(total[k] - math.fsum(p[k] for p in momenta[2:])) <= 1e-10 * max(1, total[0]) for k in range(4))
        for mass, p in zip(masses, momenta):
            assert abs(p[0] ** 2 - math.fsum(x*x for x in p[1:]) - mass*mass) <= 1e-10 * max(1, p[0]**2, mass*mass)
        assert all(abs(p[k]) <= 1e-7 for p in momenta[:2] for k in (1, 2))
        assert momenta[0][3] > 0 > momenta[1][3]
        assert abs(total[0]**2 - math.fsum(x*x for x in total[1:]) - 1e6) <= 1e-4
        for j, left in enumerate(momenta[2:]):
            pt = math.hypot(left[1], left[2]); minimum_pt = min(minimum_pt, pt)
            assert pt > 50
            for right in momenta[3+j:]:
                rpt = math.hypot(right[1], right[2])
                dphi = math.remainder(math.atan2(left[2], left[1]) - math.atan2(right[2], right[1]), 2*math.pi)
                dr = math.hypot(math.asinh(left[3]/pt) - math.asinh(right[3]/rpt), dphi)
                minimum_dr = min(minimum_dr, dr); assert dr > .5
    assert metadata['draw_count'] - metadata['rejected_draws'] == 100
    return {'count': 100, 'min_pt_GeV': minimum_pt, 'min_delta_R': minimum_dr}


def verify(root=Path(__file__).resolve().parent):
    root = Path(root)
    index = json.loads((root / 'index.json').read_bytes())
    expected = set(index['files']) | {'index.json'}
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() or p.is_symlink()}
    assert actual == expected
    for name, item in index['files'].items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        path = root / name
        assert path.is_file() and not path.is_symlink()
        parent = path.parent
        while parent != root:
            assert not parent.is_symlink()
            parent = parent.parent
        blob = path.read_bytes()
        assert len(blob) == item['bytes'] and digest(blob) == item['sha256']
    read = lambda name: json.loads((root / name).read_bytes())
    prep = read('evidence/prepare-terminal.json'); start = read('evidence/prepare-started.json')
    failed = read('evidence/run-terminal.json'); runstart = read('evidence/run-started.json')
    assert digest((root / 'evidence/prepare-terminal.json').read_bytes()) == PREP_SHA
    assert digest((root / 'evidence/run-terminal.json').read_bytes()) == RUN_SHA
    assert prep['source_index_sha256'] == start['source_index_sha256'] == failed['source_index_sha256'] == runstart['source_index_sha256'] == SOURCE_SHA
    assert prep['protocol_sha256'] == start['protocol_sha256'] == PROTOCOL_SHA
    assert prep['status'] == 'passed' and prep['error'] is None and prep['amplitude_evaluations'] == 0
    assert prep['completed_cases'] == list(CASES) and set(prep['cases']) == set(CASES)
    schedule = [(case, backend) for case in CASES for backend in BACKENDS]
    assert [(r['case'], r['backend']) for r in prep['requests']] == schedule
    assert all(r['state'] == 'unexecuted' for r in prep['requests'])
    points = {}; commands = maps = 0
    for case, seed in zip(CASES, (20261121, 20261122, 20261123, 20261124)):
        checkpoint = read('evidence/prepare/' + case + '/checkpoint.json')
        assert checkpoint == prep['cases'][case]
        for directory, key in (('prepare', 'command'), ('prepare-loader', 'runtime_command')):
            prefix = 'evidence/' + directory + '/' + case + '/'
            command = read(prefix + 'command.json'); assert command == checkpoint[key]
            assert command['error'] is None and command['returncode'] == 0 and command['process_started'] is True
            assert command['cleanup']['survivors'] == []
            assert command['elapsed_seconds'] <= command['wall_cap_seconds']
            assert digest((root / (prefix + 'stdout.txt')).read_bytes()) == command['stdout_sha256']
            assert command['cpu_exclusive'] is False and command['cpu'] == 12
            for sample in command['runtime_map_samples']:
                assert Path(sample['path']).name == sample['path']
                assert digest((root / (prefix + sample['path'])).read_bytes()) == sample['sha256']; maps += 1
            if key == 'command':
                assert command['target_argv'][:5] == ['make', '--no-print-directory', '-j1', '-f', 'pilot.mk']
                assert command['target_argv'][5] == 'pilot-build' and command['target_environment'] == {}
            else:
                closure = checkpoint['runtime_closure']
                assert command['target_argv'] == [closure['loader'], '--list', command['cwd'] + '/stability-check']
                assert (root / (prefix + 'stdout.txt')).read_text() == closure['listing']
                assert closure['files'][closure['loader']] == closure['loader_sha256']
            commands += 1
        donor = checkpoint['donor']; layout = checkpoint['group_layout']
        assert layout['common_declarations_and_static_initialization_verified'] is True
        assert layout['nloopgroups'] == len(layout['ranks']) == len(layout['ncoefs'])
        assert layout['ncoefs'] == [math.comb(rank+4, 4) for rank in layout['ranks']]
        assert max(layout['ncoefs']) <= layout['loopmaxcoefs'] and all(layout['source_prerequisites'].values())
        meta = read('points/' + case + '.json'); data = (root / ('points/' + case + '.dat')).read_bytes()
        assert digest(data) == donor['points_sha256']
        assert digest((root / ('points/' + case + '.json')).read_bytes()) == donor['point_metadata_sha256']
        assert meta['inputs']['resolved_physical_card_sha256'] == donor['physical_card_sha256']
        assert meta['inputs']['seed'] == seed
        points[case] = physical_points(data, meta, donor['compiled']['masses'])
    assert commands == 8 and maps == 12
    assert failed['status'] == 'failed' and failed['error'] is None and failed['request_count'] == 20
    assert failed['preparation_sha256'] == runstart['preparation_sha256'] == PREP_SHA
    assert [(r['request']['case'], r['request']['backend']) for r in failed['requests']] == schedule
    for record in failed['requests']:
        request = record['request']; rid = request['id']
        assert read('evidence/requests/' + rid + '/result.json') == record
        assert record['state'] == 'failed' and record['native_started'] is False
        assert 'native' not in record and 'lock' not in record
        assert record['error'] == {'class': 'ValueError', 'message': 'Unexpected extra numerical card value'}
        assert len(record['primary_call_outcomes']) == 300
        assert all(x['state'] == 'unexecuted' for x in record['primary_call_outcomes'])
        card = (root / ('written-cards/' + rid + '.dat')).read_text()
        assert card.splitlines()[-1] == '/* End of param file */'
    receipt = read('retention.json')
    assert receipt['root_reported_preparation_terminal'] == {'session_id': 48253, 'exit_code': 0}
    assert receipt['root_reported_run_terminal'] == {'session_id': 87730, 'exit_code': 1}
    return {'verified': True, 'preparation_driver_builds': 4, 'preparation_loader_queries': 4,
        'preparation_amplitudes': 0, 'failed_run_native_starts': 0,
        'failed_run_unexecuted_primary_calls': 6000, 'runtime_maps_retained': maps, 'point_streams': points}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2, sort_keys=True, allow_nan=False))
