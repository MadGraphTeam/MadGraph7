#!/usr/bin/env python3
"""Portable file-only replay. Historical host paths are strings, never inputs."""
import ast
from collections import defaultdict
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import sys
import types

SCHEMA = 'madloop-tirust-fortran-qp-profile-qualification-v1'
PINS = {
    'analyze.py': '8fa9a1710b456c3db834f67f715b6473d1fb1bc345e2408cb2b4acaefbe232c2',
    'science/index.json': '049c26fbf460e0102269bdd237a71fd8d2438160f0fa47afeb57be59aa218ba2',
    'controller1/source-index.json': '25ae013c98a9c0f30ef9e43c3aa9e1dee62ce97a76a44ac7f51b018d698702b7',
    'controller2/source-index.json': 'c5e6eeb2614ef2b0d99abc6621281479ac847add01e1019262b4c7fe97654f1d',
    'controller3/source-index.json': 'e70bf61cb3be793b1d66b8e010ad4e7c0aa98931ae795c1c9bda6adf1715406e',
    'failure-build1/failure-index.json': '8814dfb6b0d7e2c6dd0e3129af7ef4382acce4b43ca42b7846f7385d24229ab5',
    'failure-mode0/failure-index.json': 'a8d8e1c74ccdfdfcdbcf9d6977f0100d3102342c0ada59fbafe1e09bb58471da',
    'build2/build-index.json': 'd95af6d6e619b3f2d0ed8540abbe205925ca7649f41b0b2704b4c2f6eddf62b1',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def relative(name):
    p = PurePosixPath(name)
    require(isinstance(name, str) and name and not p.is_absolute() and '..' not in p.parts
            and str(p) == name and name != '.', 'Unsafe capsule member: ' + str(name))
    return p


def load_capsule(root):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), 'Capsule root must be a real directory')
    require(not (root / 'index.json').is_symlink(), 'Index symlink is forbidden')
    raw_index = (root / 'index.json').read_bytes(); index = json.loads(raw_index)
    require(index['schema'] == SCHEMA and index['performance_acceptance'] is False, 'Wrong capsule scope')
    members = index['files']; observed = set()
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(directory) / name
            require(not path.is_symlink(), 'Capsule symlink is forbidden: ' + str(path))
        observed.update((Path(directory) / name).relative_to(root).as_posix() for name in files)
    require(observed == set(members) | {'index.json'}, 'Capsule file membership differs')
    require(index['file_count'] == len(members), 'Capsule count differs')
    decoded = {}; total = 0
    for name, pin in members.items():
        path = root / relative(name)
        require(stat.S_ISREG(path.stat().st_mode), 'Nonregular capsule member')
        raw = path.read_bytes(); total += len(raw)
        require(type(pin['bytes']) is int and len(raw) == pin['bytes'] and sha(raw) == pin['sha256'],
                'Capsule member bytes differ: ' + name)
        if pin['kind'] == 'gzip-utf8-text':
            relative(pin['decoded_path'])
            require(pin['decoded_path'] not in members, 'Decoded path collides with physical member')
            require(type(pin['decoded_bytes']) is int and 0 <= pin['decoded_bytes'] <= 67108864,
                    'Unbounded decoded text')
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
                raw = stream.read(pin['decoded_bytes'] + 1)
            require(len(raw) == pin['decoded_bytes'] and sha(raw) == pin['decoded_sha256'],
                    'Decoded member bytes differ: ' + name)
        else:
            require(pin['kind'] == 'utf8-text', 'Unsupported member encoding')
        require(b'\0' not in raw, 'Binary NUL in typed text: ' + name)
        raw.decode('utf-8'); decoded[name] = raw
    require(total == index['total_bytes'], 'Capsule byte total differs')
    return index, decoded, sha(raw_index)


def snapshot_paths(files):
    """Give retained pure parsers an in-memory, read-only pathlib interface."""
    class SnapshotPath(PurePosixPath):
        def member(self):
            require(self.is_absolute() and self.is_relative_to('/capsule') and '..' not in self.parts,
                    'Parser attempted a read outside the capsule')
            return str(self.relative_to('/capsule'))

        def read_bytes(self):
            name = self.member(); require(name in files, 'Unindexed parser input: ' + name)
            return files[name]

        def read_text(self, encoding='utf-8', errors='strict'):
            return self.read_bytes().decode(encoding, errors)

        def is_file(self):
            return self.member() in files

        def open(self, mode='r', encoding='utf-8', errors='strict', newline=None):
            require(mode in ('r', 'rb'), 'Parser write is forbidden')
            return io.BytesIO(self.read_bytes()) if mode == 'rb' else io.StringIO(self.read_text(encoding, errors), newline=newline)
    return SnapshotPath


def pure(raw, names, namespace):
    tree = ast.parse(raw)
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            selected.append(node)
    found = {node.name for node in selected if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    found |= {t.id for node in selected if isinstance(node, ast.Assign) for t in node.targets if isinstance(t, ast.Name)}
    require(found == set(names), 'Missing retained pure definitions')
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<retained-pure-parser>', 'exec'), namespace)
    return types.SimpleNamespace(**namespace)


def verify(root):
    index, files, index_sha = load_capsule(root)
    document = lambda name: json.loads(files[name])
    pin = lambda name: {'bytes': len(files[name]), 'sha256': sha(files[name])}
    for name, checksum in PINS.items():
        require(sha(files[name]) == checksum, 'Historical source/closure index differs: ' + name)
    for prefix, index_name, extra in (
        ('science', 'index.json', set()), ('controller1', 'source-index.json', set()),
        ('controller2', 'source-index.json', set()), ('controller3', 'source-index.json', set()),
        ('failure-build1', 'failure-index.json', set()), ('failure-mode0', 'failure-index.json', set()),
        ('build2', 'build-index.json', {'private-link.map.gz', 'map-retention.json', 'root-build-review.json'}),
    ):
        nested = document(prefix + '/' + index_name)
        expected = set(nested['files']) | {index_name} | extra
        actual = {name[len(prefix)+1:] for name in files if name.startswith(prefix + '/')}
        require(actual == expected, 'Nested source/evidence membership differs: ' + prefix)
        for name, wanted in nested['files'].items():
            require(pin(prefix + '/' + str(relative(name))) == wanted, 'Nested member differs: ' + name)
    science = document('science/index.json'); c2 = document('controller2/source-index.json')
    c3 = document('controller3/source-index.json'); build = document('build2/build-result.json')
    bt = document('build2/terminal.json'); bi = document('build2/build-index.json')
    for number in (1, 2, 3):
        source = document(f'controller{number}/source-index.json')
        require(source['parent_proposal_index_sha256'] == PINS['science/index.json'], 'Scientific source join differs')
    for source, joins in [('controller2', {'failure_index':'failure-build1/failure-index.json',
            'source_index':'controller1/source-index.json','terminal':'failure-build1/terminal.json'}),
            ('controller3', {'build_result':'build2/build-result.json','build_terminal':'build2/terminal.json',
            'failure_index':'failure-mode0/failure-index.json','failure_terminal':'failure-mode0/terminal.json',
            'source_index':'controller2/source-index.json'})]:
        predecessor = document(source+'/source-index.json')['predecessor']
        for key, name in joins.items():
            require(predecessor[key]['sha256'] == sha(files[name]), 'Predecessor join differs')
    require(build['status'] == bt['status'] == bi['status'] == 'passed' and bt['returncode'] == 0
            and bt['cleanup']['survivors'] == [], 'Build is not closed successfully')
    require(build['source_index_sha256'] == bi['source_index_sha256'] == PINS['controller2/source-index.json'],
            'Donor build source identity differs')
    require(bi['reviewed_build_terminal_sha256'] == sha(files['build2/terminal.json']), 'Reviewed build terminal differs')
    require(build['native_executions'] == build['model_setup_calls'] == 0, 'Build executed numerical entrypoints')
    require(bi['compiled_units'] == 5 and bi['loader_queries'] == 1
            and bi['fortran_main_executions'] == bi['native_reductions'] == 0, 'Build scope differs')
    compat = document('controller3/source-compatibility.json')
    require(compat['recompiled'] is False and compat['build_source_identity_rewritten'] is False, 'Donor was relabeled')
    for key, name in [('donor_build_result_sha256','build2/build-result.json'),
                      ('donor_build_terminal_sha256','build2/terminal.json'),
                      ('donor_source_index_sha256','controller2/source-index.json'),
                      ('parent_scientific_source_index_sha256','science/index.json')]:
        require(compat[key] == sha(files[name]), 'Compatibility join differs: ' + key)
    for name, checksum in compat['byte_identical_helpers'].items():
        require(sha(files['controller2/' + name]) == sha(files['controller3/' + name]) == checksum,
                'Donor helper differs')
    for name, checksum in compat['compiled_fortran_sources'].items():
        require(build['sources']['SubProcesses/P0_uux_zgg/' + name] == checksum, 'Compiled source binding differs')
        if name not in ('check_sa.f',):
            source = 'science/qp_profile.f90' if name == 'qp_profile.f90' else 'science/source-preview/' + name
            require(sha(files[source]) == checksum, 'Scientific Fortran source differs')
    require(compat['compiled_executable_sha256'] == build['executable_sha256'], 'Compiled executable differs')
    source_inputs = document('controller3/input-pins.json')
    originals = [h for p,h in source_inputs['files'].items() if p.endswith('/SubProcesses/P0_uux_zgg/check')]
    require(len(originals) == 1 and build['protected'] == {
        'files_checked':len(source_inputs['files']), 'input_pins_sha256':sha(files['controller3/input-pins.json']),
        'original_executable_sha256':originals[0], 'original_objects':source_inputs['original_objects']},
        'Protected original identity differs from frozen source inputs')
    prelink = document('build2/prelink.json'); retained_map = document('build2/map-retention.json')
    require(sha(files['build2/prelink.json']) == build['prelink_sha256'] and prelink['completed_before_link'] is True,
            'Prelink binding differs')
    require(build['actual_link_loads'] and all(prelink['bound'].get(p) == h for p, h in build['actual_link_loads'].items()),
            'Recorded actual LOAD was not prebound')
    require(pin('build2/private-link.map.gz') == {k: retained_map['file'][k] for k in ('bytes','sha256')}
            and sha(files['build2/private-link.map.gz']) == build['link_map_sha256'], 'Retained linker map differs')
    for field, name in [('build_index_sha256','build2/build-index.json'), ('build_result_sha256','build2/build-result.json'),
                        ('build_terminal_sha256','build2/terminal.json')]:
        require(retained_map[field] == sha(files[name]), 'Map retention join differs')
    for prefix in ('failure-build1', 'failure-mode0'):
        terminal = document(prefix + '/terminal.json'); failed = document(prefix + '/failure-index.json')
        require(terminal['status'] == failed['status'] == 'failed' and terminal['returncode'] == 1
                and terminal['cleanup']['survivors'] == [] and failed['native_executions'] == 0,
                'Historical preparation failure was relabeled or unresolved')
    for alias, suffix in [('original-warmup-reference.json', 'candidate-9/raw/observation.json'),
                          ('original-candidate-inputs.json', 'candidate-9/candidate-inputs.json')]:
        matches = [item for item in document('science/input-pins.json')['files'] if item['path'].endswith(suffix)]
        require(len(matches) == 1 and pin(alias) == {k: matches[0][k] for k in ('bytes','sha256')},
                'Original comparison input differs')
    Snap = snapshot_paths(files); virtual = Snap('/capsule')
    env = dict(Path=Snap, math=math, json=json, struct=struct, re=re, hashlib=hashlib, PROPOSAL=virtual/'science')
    transform = pure(files['controller3/worker_flags.py'], {'BASE_WORKER_SHA256','FLAGS_FILENAME','FLAGS_COLUMNS',
        'DECLARATIONS','OPEN_FLAGS','ANCHOR','OBSERVED_ANCHOR','CAPTURE','_replace_once','instrument_worker'}, dict(env))
    require(sha(transform.instrument_worker(files['science/source-preview/check_sa.f'].decode()).encode())
            == build['private_driver_sha256'] == compat['compiled_fortran_sources']['check_sa.f'], 'Private worker transform differs')
    controller = pure(files['controller3/controller.py'], {'read','exact_rows','read_flags','validate_profile'}, dict(env))
    for name in ('tirust_benchmark_worker.py', 'tirust_benchmark_warmup.py'):
        require(pin('protocol/' + name) == c3['support']['tools/' + name], 'Retained parser source differs')
    worker = pure(files['protocol/tirust_benchmark_worker.py'], {'BACKEND_CODES','MeasurementError','validate_status'}, dict(env))
    warm = pure(files['protocol/tirust_benchmark_warmup.py'], {'_call_groups','parse_warmup'},
                dict(env, worker=worker, MeasurementError=worker.MeasurementError))
    shadow = pure(files['science/analyze_shadow.py'], {'inspect'}, dict(env))
    reference = document('original-warmup-reference.json'); original = document('original-candidate-inputs.json')
    for name, expected in [('points.dat',original['points_sha256']), ('points.json',original['points_metadata_sha256']),
                           ('param_card.dat',original['param_card_sha256']),
                           ('manifest.json',original['campaign_binding']['manifest_sha256']),
                           ('MadLoopParams.dat',original['numerical_cards']['Cards/MadLoopParams.dat'])]:
        require(sha(files['inputs/'+name]) == expected, 'Frozen physical input differs: ' + name)
    summaries = []; closure_modes = []; shared_rows = shared_flags = None
    card_paths = {'Cards/MadLoopParams.dat','SubProcesses/MadLoopParams.dat',
                  'SubProcesses/MadLoop5_resources/MadLoopParams.dat','SubProcesses/P0_uux_zgg/MadLoop5_resources/MadLoopParams.dat'}
    for mode in range(3):
        prefix = f'mode-{mode}'; raw = virtual/prefix/'raw'
        result = document(prefix+'/result.json'); terminal = document(prefix+'/terminal.json')
        admission = document(prefix+'/admission.json'); before = document(prefix+'/private-inputs-before.json')
        after = document(prefix+'/private-inputs-after.json'); ready = document(prefix+'/ready.json')
        require(result['status'] == terminal['status'] == 'passed' and result['mode'] == mode
                and terminal['returncode'] == 0 and terminal['cleanup']['survivors'] == [], 'Unclosed mode')
        require(terminal['result_sha256'] == sha(files[prefix+'/result.json']), 'Mode result receipt differs')
        require(admission['source_index_sha256'] == PINS['controller3/source-index.json']
                and admission['reviewed_build_sha256'] == sha(files['build2/terminal.json'])
                and admission['command'] == terminal['command'] and admission['stage'] == terminal['stage'] == prefix,
                'Mode admission differs')
        started = document(prefix+'/started.json'); limits = document(prefix+'/resource-limits.json')
        for key in ('stage','command','cap_seconds','cleanup_reserve_seconds','child_pid',
                    'deadline_monotonic','started_utc','resource_limits'):
            require(started[key] == terminal[key], 'Started/terminal identity differs')
        require(terminal['cap_seconds'] == admission['cap_seconds'] == c3['caps_seconds'][prefix]
                and terminal['resource_limits'] == admission['resource_limits'] == result['resource_limits'] == c3['resource_limits'],
                'Resource admission differs')
        require(limits['actual'] == {k:[v,v] for k,v in c3['resource_limits'].items()}, 'Actual resource limits differ')
        require(result['build_result_sha256'] == sha(files['build2/build-result.json'])
                and result['donor_source_index_sha256'] == PINS['controller2/source-index.json'], 'Mode donor differs')
        require(result['protected'] == build['protected'], 'Mode protected-original identity differs')
        require(result['performance_acceptance'] is False and result['optimized'] is False
                and result['warmup_points'] == 100 and result['actual_anchor_calls'] == 2, 'Mode scope differs')
        require(all(before[k] == after[k] for k in ('param_card_sha256','numerical_cards','executable_sha256'))
                and after['executable_sha256'] == build['executable_sha256'], 'Private scientific inputs changed')
        require(set(after['numerical_cards']) == card_paths and len(set(after['numerical_cards'].values())) == 1,
                'All four physical numerical cards must agree')
        require(after['param_card_sha256'] == original['param_card_sha256']
                and all(after['numerical_cards'][p] == h for p,h in original['numerical_cards'].items()), 'Original cards differ')
        require(result['private_runtime'] == after['runtime'], 'Result runtime join differs')
        for key in ('libraries','archives','metadata'):
            require(before['runtime'][key] == after['runtime'][key] == build['runtime'][key], 'Historical provider identity differs')
        for binding in (before, after):
            runtime = binding['runtime']['actual_pt_interp']; expected = build['runtime']['actual_pt_interp']
            require(runtime['files'] == expected['files'] and runtime['loader'] == expected['loader']
                    and runtime['files'][runtime['loader']] == runtime['loader_sha256'], 'Actual interpreter/provider binding differs')
        expected_maps = after['runtime']['actual_pt_interp']['files']
        for phase in ('before','after'):
            mapped = document(prefix+'/maps-'+phase+'.json'); paths = set()
            for line in mapped['maps'].splitlines():
                columns = line.split(None,5)
                if len(columns) == 6 and columns[5].startswith('/'):
                    require(not columns[5].endswith(' (deleted)'), 'Deleted mapped provider')
                    paths.add(columns[5])
            require(paths == set(mapped['files']) and all(mapped['files'].get(p) == h for p,h in expected_maps.items()),
                    'Raw process maps/provider table differs')
            extras = {p:h for p,h in mapped['files'].items() if p not in expected_maps}
            executables = {p:h for p,h in extras.items() if p.endswith(
                '/fortran-qp-transport-profile-controller-3/'+prefix+'/SubProcesses/P0_uux_zgg/check')}
            locale = '/nix/store/ig5ylfz7qcgpmvzip4pxfkp2pfi76s2j-glibc-locales-2.44-25/lib/locale/locale-archive'
            require(len(executables) == 1 and set(executables.values()) == {build['executable_sha256']}
                    and extras == {**executables,locale:'53851ab7caa5a0f893056762a6abff4e383ea06c0e6b276ee16876499e712d27'},
                    'Unexpected mapped executable/library beyond admitted providers and locale')
        parsed = warm.parse_warmup(raw, 9, 100, ready)
        require(parsed == document(prefix+'/warmup.json') and parsed['rows'] == reference['rows'], 'Warmup replay differs')
        require(all(parsed[k] == reference[k] for k in ('warmup_first_call','warmup_last_call','warmup_call_point_order')),
                'Warmup call window differs from the unchanged learning traversal')
        calls = {int(line.split()[0]) for line in (raw/'warmup.dat').read_text().splitlines()
                 if line.strip() and not line.startswith('#')}
        require(calls == set(range(101)) and len(calls) == result['warmup_public_calls'] == 101, 'Warmup call count/window differs')
        rows = controller.exact_rows(raw/'profile-values.dat'); flags = controller.read_flags(raw/'ieee-flags.dat', mode)
        require(rows == result['rows'] and flags == result['ieee_state'], 'Raw anchor/IEEE replay differs')
        require(all(row[2] in (1,2,3,4) and all(value in (0,1) for value in row[3:]) for row in flags),
                'Current IEEE qualification contains unknown observations')
        if mode == 0:
            shared_rows, shared_flags = rows, flags
        require(rows == shared_rows and flags == shared_flags, 'Actual numerical/IEEE parity differs')
        profile = {str(p): controller.validate_profile(raw/f'point-{p:02d}', mode) for p in (1,16)}
        require(profile == result['profile'], 'Raw clock/counter replay differs')
        traced = {str(p): shadow.inspect(raw/f'point-{p:02d}') for p in (1,16)} if mode == 2 else None
        require(traced == result['shadow'], 'Exact-key shadow replay differs')
        summaries.append({'mode':mode,'warmup_points':100,'warmup_public_calls':101,'anchors':2,
                          'tensor_calls':{p:v['tensor_calls'] for p,v in traced.items()} if traced else None})
        closure_modes.append({'mode':mode,'status':terminal['status'],'returncode':terminal['returncode'],
            'cleanup_survivors':terminal['cleanup']['survivors'],
            'observed_owned_identities':len(terminal['cleanup']['observed_owned_processes']),
            'result_sha256':sha(files[prefix+'/result.json']),'terminal_sha256':sha(files[prefix+'/terminal.json']),
            'steady_points':result['warmup_points'],'actual_anchor_calls':result['actual_anchor_calls'],
            'actual_anchor_statuses':[row['status'] for row in rows]})
    closure = document('run-closure.json')
    require(closure['schema'] == 'private-qp-profile-run-closure-v1' and closure['modes'] == closure_modes
            and closure['all_modes_passed'] is True and closure['native_stage_count'] == len(summaries)
            and closure['total_recorded_warmup_public_calls'] == sum(m['warmup_public_calls'] for m in summaries)
            and closure['full_steady_learning_points'] == sum(m['warmup_points'] for m in summaries)
            and closure['native_amplitude_anchors'] == sum(m['anchors'] for m in summaries)
            and closure['optimization_applied'] is False and closure['performance_acceptance'] is False
            and closure['failed_preparations_preserved'] == ['failure-build1/failure-index.json','failure-mode0/failure-index.json'],
            'Run closure differs from replayed stages')
    for key,name in [('donor_build_result_sha256','build2/build-result.json'),
                     ('donor_build_terminal_sha256','build2/terminal.json'),
                     ('donor_source_index_sha256','controller2/source-index.json'),
                     ('source_index_sha256','controller3/source-index.json')]:
        require(closure[key] == sha(files[name]), 'Run closure source/build join differs')
    analyzer = pure(files['analyze.py'], {'STAGES','COUNTERS','read','analyze'}, dict(env, defaultdict=defaultdict))
    analysis = json.loads(json.dumps(analyzer.analyze(virtual), allow_nan=False))
    require(analysis == document('analysis.json'), 'Derived descriptive analysis differs')
    return {'verified':True,'index_sha256':index_sha,'payloads':len(files),'modes':summaries,
            'numerical_and_actual_ieee_parity':True,'historical_failures_retained':2,
            'native_executions_by_verifier':0,'performance_acceptance':False}


if __name__ == '__main__':
    print(json.dumps(verify(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent), indent=2, sort_keys=True))
