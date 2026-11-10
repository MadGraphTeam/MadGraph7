#!/usr/bin/env python3
"""One closed current-source QCD n3 raw pilot; planning is inert.

No stage is admitted by writing this source. Root must review the exact seal
and actual read-only preflight before a separate compiler/native admission.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WORK = ROOT / '.tirust-build/current-source-qcd3-raw-pilot-1'
REPORT = WORK / 'evidence'
SOURCE5 = HERE.parent / 'source5'
PILOT_SHA = '22b06cda2303d9d2fd42aed418aa4518db0d9e834678862330ffa0d6600e9eef'
PILOT_INDEX_SHA = 'ffbc57948b062e6380f71e22c9f303b0e652bcd4060952b7bc0a4fe216191271'
CLASSIFIER = HERE.parent / 'current-source-classifier-proposal-1'
CLASSIFIER_INDEX_SHA = '7bc7a08c2a96483100ea47c2d652694a34a4f426af2b5ba9a8317798d6003032'
MANIFEST = ROOT / 'docs/benchmarkfolder/campaigns/entry-latch-cache-guard-1/manifest.json'
MANIFEST_SHA = '4a27dda7a0977cd1b0ed0d779b30c403da50968bf494aa96d31cf18fc9fe2076'
CASE = 'qcd_n3_uux_zgg'
BACKENDS = (1, 6, 7, 8, 9)
CPU = 13


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def module(path, checksum):
    path = Path(path)
    blob = path.read_bytes()
    require(hashlib.sha256(blob).hexdigest() == checksum, 'Bound Python source differs: ' + str(path))
    result = types.ModuleType('current_raw_' + path.stem)
    result.__file__ = str(path)
    exec(compile(blob, str(path), 'exec'), result.__dict__)
    return result


def protocol():
    value = read(HERE / 'protocol.json')
    require(value['case'] == CASE and tuple(value['backends']) == BACKENDS
            and value['campaign_manifest_sha256'] == MANIFEST_SHA,
            'Only the sealed current-source one-case five-backend pilot is supported')
    limits = value['limits']
    require(limits['cpu'] == CPU and limits['memory_bytes'] == 4294967296
            and limits['request_wall_seconds'] == 180 and limits['prepare_wall_seconds'] == 120
            and limits['loader_wall_seconds'] == 30 and limits['lock_wait_seconds'] == 1800
            and limits['common_lock'] == '.tirust-build/numerical-tests.lock', 'Execution limits differ')
    return value


def seal():
    index = read(HERE / 'source-index.json')
    require({'controller.py', 'protocol.json', 'closed-input.json', 'test_controller.py', 'verify_readiness.py'}
            .issubset(index['files']), 'Pilot source/admission seal is incomplete')
    for name, item in index['files'].items():
        require(Path(name).name == name and digest(HERE / name) == item['sha256'], 'Pilot source differs: ' + name)
    for name, checksum in index['inputs'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts
                and not name.startswith('.benchmark-work/'), 'Source seal cannot bypass the closed-export gate')
        require(digest(ROOT / name) == checksum, 'Pilot input differs: ' + name)
    return digest(HERE / 'source-index.json')


def components():
    require(digest(SOURCE5 / 'source-index.json') == PILOT_INDEX_SHA, 'Frozen source5 seal differs')
    p = module(SOURCE5 / 'pilot.py', PILOT_SHA)
    command_cpu(p, (SOURCE5 / 'pilot.py').read_bytes())
    helpers = p.load_helpers()
    require(digest(CLASSIFIER / 'source-index.json') == CLASSIFIER_INDEX_SHA, 'Classifier seal differs')
    classifier = module(CLASSIFIER / 'classifier.py', read(CLASSIFIER / 'source-index.json')['files']['classifier.py']['sha256'])
    # Only this private module's environment and postguard are adapted. None of
    # the old campaign/protocol/donor/inheritance/prepare/run functions is used.
    p.WORK, p.REPORT = WORK, REPORT
    p.verify_production = lambda unused=None: production(p)
    return p, helpers, classifier


def command_cpu(p, source):
    """Change only the pinned supervisor's taskset CPU and matching metadata."""
    require(hashlib.sha256(source).hexdigest() == PILOT_SHA, 'Pinned supervisor source differs')
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'command']
    require(len(nodes) == 1, 'Unique pinned command helper required')
    launchers = records = 0
    for node in ast.walk(nodes[0]):
        if isinstance(node, ast.List) and len(node.elts) >= 3:
            if all(isinstance(item, ast.Constant) for item in node.elts[:3]) and [item.value for item in node.elts[:3]] == ['taskset', '-c', '12']:
                node.elts[2].value = str(CPU); launchers += 1
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == 'cpu':
                    require(isinstance(value, ast.Constant) and type(value.value) is int and value.value == 12,
                            'Pinned command CPU metadata differs')
                    value.value = CPU; records += 1
    require((launchers, records) == (1, 1), 'Scheduling adaptation must change exactly two CPU constants')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 'pinned-source5-command-cpu13', 'exec'), p.__dict__)


def production(p):
    require(digest(MANIFEST) == MANIFEST_SHA, 'Current campaign manifest differs')
    value = read(MANIFEST)
    files = dict(value['candidate']['exporter_identity']['files'])
    files.update({name: item['sha256'] for name, item in value['candidate']['numerical_sources'].items()})
    p.verify_source_inventory(ROOT, files)
    native = value['candidate']['native']
    for name, checksum in native['files'].items():
        require(digest(Path(native['prefix']) / name) == checksum, 'Current native provider differs: ' + name)
    return value


def requests():
    return [{'id': CASE + '-backend' + str(backend), 'case': CASE, 'backend': backend,
             'lock_required': backend in (8, 9), 'primary_calls_expected': 300,
             'bootstrap_calls_maximum': 16} for backend in BACKENDS]


def verify_case_checkpoint(checkpoint, binding):
    require(checkpoint['schema'] == 'madloop-tirust-one-case-closure-v1'
            and checkpoint['campaign'] == 'entry-latch-cache-guard-1'
            and checkpoint['manifest_sha256'] == MANIFEST_SHA and checkpoint['case']['id'] == CASE,
            'Wrong closed-case checkpoint')
    result = checkpoint['numerical_result']
    require(result['case'] == CASE and result['status'] == 'passed' and result['returncode'] == 0
            and result['raw_replay_passed'] is True, 'Ordinary donor stage did not close successfully')
    require([(row['source'], row['backend']) for row in result['lanes']]
            == [('baseline', 1)] + [('candidate', backend) for backend in BACKENDS]
            and all(row['status'] == 'complete' and row['error'] is None
                    and row['matched_saved_points'] == 100 and row['numerical_agreement_passed'] is True
                    for row in result['lanes']), 'Six closed ordinary lanes are required')
    require(checkpoint['file_count'] == len(checkpoint['files']) and
            binding['case'] == CASE and binding['state'] == 'closed', 'Closed-case binding differs')


def closed_donor(value, p, helpers, classifier):
    binding = read(HERE / 'closed-input.json')
    require(binding['state'] == 'closed' and binding['case'] == CASE,
            'Closed owner checkpoint is required before any export read')
    checkpoint_path = ROOT / binding['checkpoint']['path']
    require(digest(checkpoint_path) == binding['checkpoint']['sha256'], 'Closed owner checkpoint differs')
    checkpoint = read(checkpoint_path)
    verify_case_checkpoint(checkpoint, binding)
    campaign_prefix = 'docs/benchmarkfolder/campaigns/entry-latch-cache-guard-1/'
    shared = {campaign_prefix + name for name in ('manifest.json', 'cases.json',
        'remaining-qualification-1/source-index.json', 'remaining-qualification-1/specification.json')}
    scoped = tuple(campaign_prefix + name for name in
        ('experiments/numerical/full100-warmup-' + CASE + '-1/', 'inputs/candidate/' + CASE + '/',
         'inputs/baseline/' + CASE + '/', 'remaining-qualification-1/' + CASE + '/'))
    for name, item in checkpoint['files'].items():
        require(name in shared or name.startswith(scoped)
                or name in {campaign_prefix + 'points/' + CASE + ending for ending in ('.dat', '.json')},
                'Checkpoint refers outside the explicitly closed case')
        path = ROOT / name
        require(path.stat().st_size == item['bytes'] and digest(path) == item['sha256'],
                'Closed checkpoint payload differs: ' + name)
    export = ROOT / binding['export']
    require(export.resolve() == export and export.is_dir(), 'Donor export traverses an alias or is missing')
    retained = MANIFEST.parent / 'inputs/candidate' / CASE
    for name in ('generation.json', 'compilation.json'):
        require(digest(export / name) == digest(retained / name) == binding[name + '_sha256'],
                'Current donor receipt differs: ' + name)
    generated, compiled = read(export / 'generation.json'), read(export / 'compilation.json')
    manifest = production(p)
    identity = manifest['candidate']['exporter_identity']
    require(generated['source'] == 'candidate' and generated['case']['id'] == CASE
            and generated['source_identity']['files'] == identity['files']
            and generated['source_identity']['files_sha256'] == identity['files_sha256']
            and generated['campaign_binding'] == compiled['campaign_binding']
            == {'id': 'entry-latch-cache-guard-1', 'manifest_sha256': MANIFEST_SHA},
            'Current generated/compiled content identity differs')
    require(compiled['full_numerical_rebuild_this_invocation'] is True
            and compiled['numerical_build_from_sources'] is True, 'Full current numerical source build required')
    for name, checksum in compiled['sources'].items():
        require(digest(export / name) == checksum, 'Current generated source differs: ' + name)
    directories = [path.parent for path in export.glob('SubProcesses/P*/check_sa.f')]
    require(len(directories) == 1, 'Exactly one closed subprocess required')
    directory = directories[0]
    require(digest(directory / 'check') == compiled['executable_sha256'] == binding['executable_sha256'],
            'Closed donor executable differs')
    inputs_path = ROOT / binding['numerical_inputs']['path']
    require(str(inputs_path.relative_to(ROOT)) in checkpoint['files'] and
            digest(inputs_path) == binding['numerical_inputs']['sha256'], 'Closed numerical input receipt omitted/changed')
    spec = read(inputs_path)
    require(spec['generation'] == generated and spec['compilation'] == compiled,
            'Historical closed build binding differs')
    card_sha = digest(export / 'Cards/param_card.dat')
    require(card_sha == spec['param_card_sha256'] == digest(retained / 'param_card.dat'), 'Physical card differs')
    for section in (spec['runtime'][name] for name in ('archives', 'libraries', 'metadata')):
        for name, checksum in section.items():
            require(digest(name) == checksum, 'Closed runtime/archive/metadata provider differs: ' + name)
    points = ROOT / value['points']['path']; metadata_path = ROOT / value['points']['metadata_path']
    require(digest(points) == value['points']['sha256']
            and digest(metadata_path) == value['points']['metadata_sha256'], 'Saved cut-qualified points differ')
    metadata = read(metadata_path); inputs = metadata['inputs']
    require(inputs['resolved_physical_card_sha256'] == card_sha and inputs['count'] == 100
            and inputs['incoming_masses'] + inputs['masses'] == compiled['masses'], 'Saved points/card/masses differ')
    vectors = p.point_array(points.read_text(), compiled['worker_render_parameters']['nexternal'])
    require(vectors == [row['momenta'] for row in metadata['points']], 'Point vectors differ from saved metadata')
    helpers['points'].helper().validate_points(metadata['points'], masses=inputs['masses'],
                                              incoming_masses=inputs['incoming_masses'])
    cuts = p.check_points(value, points, compiled['worker_render_parameters']['nexternal'])
    matrix = directory / 'loop_matrix.f'
    classifier.generated_entry_reset(matrix.read_bytes(), compiled['worker_render_parameters']['prefix'],
        compiled['worker_render_parameters']['nexternal'], (ROOT / classifier.TEMPLATE_KEY).read_bytes())
    return {'export': str(export), 'subprocess': str(directory.relative_to(export)), 'compiled': compiled,
            'physical_card_sha256': card_sha, 'runtime': {name: spec['runtime'][name] for name in ('archives', 'libraries', 'metadata')}, 'points': str(points),
            'points_sha256': digest(points), 'point_metadata_sha256': digest(metadata_path),
            'mass_inputs': inputs, 'cuts': cuts,
            'generation_revision': generated['source_identity']['revision'],
            'closed_case_receipt': {'path': str(checkpoint_path), 'sha256': digest(checkpoint_path)}}


def readonly():
    source = seal(); value = protocol(); p, helpers, classifier = components()
    require(not WORK.exists() and not WORK.is_symlink(), 'Read-only preflight needs a fresh work namespace')
    affinity = sorted(p.os.sched_getaffinity(0))
    require(CPU in affinity, 'Proposed nonexclusive CPU13 is outside allowed affinity')
    data = closed_donor(value, p, helpers, classifier); links = helpers['links']
    export = Path(data['export']); directory = export / data['subprocess']
    snapshot = links.tree_snapshot(export); audit = p.link_audit(links, snapshot)
    links._guard_writes(export, snapshot)
    inventories, leaves, observations, transitive = links._external_parts(audit)
    files, directories, nodes = links._allowed_graph(inventories + [snapshot], leaves, observations, transitive)
    links._admit_links(inventories + [snapshot], files, directories, nodes)
    layout = p.group_layout(directory, data['compiled']['worker_render_parameters']['prefix'])
    prerequisites = p.process_prerequisites(directory, data['compiled'])
    require(digest(export / 'Source/MODEL/lha_read.f') == p.LHA_READ_SHA
            and digest(export / 'Source/MODEL/rw_para.f') == p.RW_PARA_SHA,
            'Compiled physical-card search sources differ from the qualified resolver')
    compiler = data['compiled']['effective_compiler_flags']['scopes']['process']['recipes']['check_sa.f']['compiler']
    require(len(compiler) == 1 and digest(compiler[0]['path']) == compiler[0]['sha256'], 'Recorded GNU Fortran compiler differs')
    loader = p.elf_interpreter(directory / 'check')
    require(data['runtime']['libraries'].get(str(loader)) == digest(loader), 'Actual target interpreter is not bound')
    driver = p.render((SOURCE5 / 'stability_worker.f').read_text(), data['compiled']['worker_render_parameters'])
    links.verify_tree(export, snapshot); links.verify_external_graph(audit); production(p)
    return {'source_index_sha256': source, 'protocol_sha256': digest(HERE / 'protocol.json'),
            'donor': data, 'donor_snapshot': snapshot, 'audit': audit, 'group_layout': layout,
            'prerequisites': prerequisites, 'compiler': compiler[0], 'target_interpreter': str(loader),
            'rendered_driver_sha256': hashlib.sha256(driver.encode()).hexdigest(),
            'requests': requests(), 'commands_executed': 0, 'compiler_invocations': 0,
            'loader_invocations': 0, 'amplitude_evaluations': 0, 'point_generation_calls': 0,
            'allowed_affinity': affinity, 'selected_cpu': CPU, 'cpu_exclusive': False}


def prepare():
    admitted = readonly(); value = protocol(); p, helpers, classifier = components(); links = helpers['links']
    destination, unused, chain = links._resolve(WORK, missing=True)
    require(destination == WORK and not chain, 'Private work path traverses an alias')
    WORK.mkdir(parents=True); REPORT.mkdir()
    p.save(REPORT / 'prepare-started.json', {'source_index_sha256': admitted['source_index_sha256'], 'amplitude_evaluations': 0})
    p.save(REPORT / 'prepare-inputs.json', admitted)
    case = None; error = None; guard_error = None
    try:
        data = admitted['donor']; target = WORK / 'master'
        context = links.prepare_worker(Path(data['export']), target, admitted['donor_snapshot'], admitted['audit'])
        directory = target / data['subprocess']; layout = admitted['group_layout']
        for name in ('check_sa.f', 'check_sa.o', 'pilot.mk', 'stability-check', 'stability-layout.inc'):
            links.assert_private_path(target, directory / name)
        (directory / 'check_sa.f').write_text(p.render((SOURCE5 / 'stability_worker.f').read_text(), data['compiled']['worker_render_parameters']))
        (directory / 'pilot.mk').write_text(p.MAKEFILE)
        (directory / 'stability-layout.inc').write_text(layout['include'])
        shell = shutil.which('bash', path=p.environment(directory)['PATH'])
        require(shell is not None, 'Pinned environment Bash missing')
        with p.signal_requests(helpers['cleanup']) as signals:
            command = p.command(['make', '--no-print-directory', '-j1', '-f', 'pilot.mk', 'pilot-build',
                'FC=' + admitted['compiler']['path'], 'SHELL=' + str(Path(shell).resolve())], directory,
                REPORT / 'prepare', value['limits']['prepare_wall_seconds'], helpers['cleanup'], signals)
            changed = [directory / 'check_sa.f', directory / 'check_sa.o']
            additions = [str((directory / name).relative_to(target)) for name in ('pilot.mk', 'stability-check', 'stability-layout.inc')]
            private = p.private_changes(links, target, context, changed, additions)
            require(command['error'] is None, 'Private driver build failed; no fallback')
            executable = directory / 'stability-check'; loader = p.elf_interpreter(executable)
            loader_command = p.command([loader, '--list', executable], directory, REPORT / 'loader',
                value['limits']['loader_wall_seconds'], helpers['cleanup'], signals,
                target_environment=p.native_environment(read(MANIFEST)))
            require(loader_command['error'] is None, 'Target PT_INTERP query failed')
            # Keep the exact excluded host-loader path/hash semantics, while
            # binding its documentary role to this current closed receipt.
            p.HISTORICAL_LDD_HOST_LOADER = dict(p.HISTORICAL_LDD_HOST_LOADER,
                binding='current closed QCD n3 numerical input receipt ' + read(HERE / 'closed-input.json')['numerical_inputs']['sha256'])
            runtime = p.qualify_loader_closure(executable, Path(data['export']) / data['subprocess'] / 'check',
                data['runtime']['libraries'], (REPORT / 'loader/stdout.txt').read_text())
            p.private_changes(links, target, context, changed, additions)
        case = {'donor': data, 'donor_context': context, 'master_snapshot': private,
                'executable_sha256': digest(executable), 'driver_sha256': digest(directory / 'check_sa.f'),
                'command': command, 'group_layout': layout, 'runtime_closure': runtime,
                'runtime_command': loader_command, 'native_environment': p.native_environment(read(MANIFEST))}
        qualification = classifier.qualify_sources(manifest=MANIFEST.read_bytes(),
            template=(ROOT / classifier.TEMPLATE_KEY).read_bytes(),
            generation=(Path(data['export']) / 'generation.json').read_bytes(),
            compilation=(Path(data['export']) / 'compilation.json').read_bytes(),
            matrix=(directory / 'loop_matrix.f').read_bytes(), worker_template=(SOURCE5 / 'stability_worker.f').read_bytes(),
            driver=(directory / 'check_sa.f').read_bytes(), private_build=canonical(case),
            matrix_source_key=data['subprocess'] + '/loop_matrix.f', worker_parameters=data['compiled']['worker_render_parameters'])
        case['classifier_binding'] = qualification._asdict()
        links.verify_tree(Path(data['export']), admitted['donor_snapshot']); links.verify_external_graph(admitted['audit']); production(p)
    except BaseException as caught:
        error = {'class': type(caught).__name__, 'message': str(caught)}
    finally:
        try:
            links.verify_tree(Path(admitted['donor']['export']), admitted['donor_snapshot'])
            links.verify_external_graph(admitted['audit']); production(p)
            require(seal() == admitted['source_index_sha256'], 'Pilot sources/saved point inputs changed during preparation')
        except BaseException as caught:
            guard_error = {'class': type(caught).__name__, 'message': str(caught)}
        p.save(REPORT / 'prepare-terminal.json', {'source_index_sha256': admitted['source_index_sha256'],
            'protocol_sha256': digest(HERE / 'protocol.json'), 'status': 'passed' if error is None and guard_error is None else 'failed',
            'error': error, 'protected_guard_error': guard_error, 'case': case, 'requests': requests(), 'amplitude_evaluations': 0})
    require(error is None and guard_error is None, 'Preparation failed; preserved without retry')


def prepared():
    source = seal(); value = protocol(); p, helpers, classifier = components(); production(p)
    checkpoint = read(REPORT / 'prepare-terminal.json')
    require(checkpoint['status'] == 'passed' and checkpoint['source_index_sha256'] == source
            and checkpoint['protocol_sha256'] == digest(HERE / 'protocol.json'), 'Passed matching private preparation required')
    case = checkpoint['case']; master = Path(case['master_snapshot']['root']); links = helpers['links']
    links.verify_tree(master, case['master_snapshot']); links.verify_external_graph(case['donor_context']['audit'])
    data = case['donor']; directory = master / data['subprocess']
    require(digest(directory / 'stability-check') == case['executable_sha256'], 'Prepared executable differs')
    for path, checksum in case['runtime_closure']['files'].items():
        require(digest(path) == checksum, 'Prepared target runtime differs')
    closed_donor(value, p, helpers, classifier)
    qualification = classifier.qualify_sources(manifest=MANIFEST.read_bytes(), template=(ROOT / classifier.TEMPLATE_KEY).read_bytes(),
        generation=(Path(data['export']) / 'generation.json').read_bytes(),
        compilation=(Path(data['export']) / 'compilation.json').read_bytes(), matrix=(directory / 'loop_matrix.f').read_bytes(),
        worker_template=(SOURCE5 / 'stability_worker.f').read_bytes(), driver=(directory / 'check_sa.f').read_bytes(),
        private_build=canonical({name: item for name, item in case.items() if name != 'classifier_binding'}),
        matrix_source_key=data['subprocess'] + '/loop_matrix.f', worker_parameters=data['compiled']['worker_render_parameters'])
    require(qualification._asdict() == case['classifier_binding'], 'Classifier source/build binding differs')
    return source, value, p, helpers, classifier, case, qualification


def point_metrics(p, value, results, spread):
    """Keep the frozen numerical aggregation, with explicit current provenance."""
    namespace = dict(p.point_metrics.__globals__, HERE=HERE, BACKENDS=BACKENDS,
                     REVISION=read(MANIFEST)['candidate']['exporter_identity']['revision'],
                     MANIFEST_SHA=MANIFEST_SHA, digest=digest)
    current = types.FunctionType(p.point_metrics.__code__, namespace,
                                 p.point_metrics.__name__, p.point_metrics.__defaults__)
    qualified = [dict(row, readonly_guard_passed=(row.get('readonly_guard_passed') is True
                 and row.get('protected_guards_passed') is True)) for row in results]
    return current(value, qualified, spread)


def protected_inputs(p, helpers, case, source):
    """Require the same frozen source/providers/points before and after workers."""
    links = helpers['links']
    links.verify_tree(Path(case['master_snapshot']['root']), case['master_snapshot'])
    links.verify_tree(Path(case['donor_context']['master']), case['donor_context']['expected_master'])
    links.verify_external_graph(case['donor_context']['audit'])
    for path, checksum in case['runtime_closure']['files'].items():
        require(digest(path) == checksum, 'Prepared runtime provider changed')
    production(p)
    require(seal() == source, 'Pilot sources/saved point inputs changed')


def collect_diagnostics(record, parser, output, directory, evidence, layout, links):
    """Retain partial data, and distinguish protocol failure from native failure."""
    raw_files = {}
    for name in ('stability-dimensions.dat', 'stability-values.dat', 'stability-groups.dat'):
        path = directory / name
        links.assert_private_path(output, path)
        if path.is_file():
            raw_files[name] = {'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size}
    record['raw_diagnostics'] = raw_files
    def text(name):
        return (directory / name).read_text() if name in raw_files else ''
    if text('stability-dimensions.dat'):
        console = evidence / 'stdout.txt'
        record['diagnostics'] = parser(text('stability-dimensions.dat'), text('stability-values.dat'),
            console.read_text() if console.is_file() else '', text('stability-groups.dat'), layout)
        require(record['diagnostics']['structurally_complete'] and
                record['diagnostics']['loop_groups_structurally_complete'],
                'Raw diagnostic journals are incomplete or structurally invalid')
    else:
        record['diagnostic_unavailable'] = 'Dimensions not produced; all 100 intended points remain unavailable'
        record['state'] = 'failed'


def workers(stage):
    source, value, p, helpers, classifier, case, qualification = prepared(); links = helpers['links']
    require(not (REPORT / (stage + '-started.json')).exists(), 'A stage cannot be retried in place')
    if stage == 'run':
        dry = read(REPORT / 'dryrun-terminal.json')
        require(dry['status'] == 'passed' and dry['source_index_sha256'] == source
                and dry['preparation_sha256'] == digest(REPORT / 'prepare-terminal.json'), 'Passed full five-worker dryrun required')
    p.save(REPORT / (stage + '-started.json'), {'source_index_sha256': source,
        'preparation_sha256': digest(REPORT / 'prepare-terminal.json'), 'requests': requests()})
    parser = classifier.parsers((SOURCE5 / 'pilot.py').read_bytes(), qualification=qualification)['parse_diagnostics']
    results = []; error = None; metrics_error = None
    try:
        with p.signal_requests(helpers['cleanup']) as signals:
            for request in requests():
                output = WORK / stage / request['id']; record = {'request': request, 'state': 'unexecuted', 'native_started': False}
                context = None
                try:
                    try:
                        protected_inputs(p, helpers, case, source)
                    except BaseException as caught:
                        record['source_admission_error'] = {'class': type(caught).__name__, 'message': str(caught)}
                        raise
                    context = links.prepare_worker(Path(case['master_snapshot']['root']), output,
                        case['master_snapshot'], case['donor_context']['audit'])
                    directory = p.prelaunch_worker(value, request, case, output, helpers, record)
                    if stage == 'run':
                        executable = directory / 'stability-check'
                        with p.numerical_lock(value, request, signals) as lock:
                            command = p.command([executable, case['donor']['points'], str(request['backend'])], directory,
                                REPORT / stage / request['id'], value['limits']['request_wall_seconds'],
                                helpers['cleanup'], signals, target_environment=case['native_environment'])
                        record.update(command=command, lock=lock, native_started=command['process_started'],
                                      state='complete' if command['error'] is None else 'failed')
                        record['runtime_binding'] = p.runtime_binding(REPORT / stage / request['id'], executable,
                                                                    case['runtime_closure']['files'])
                        require(record['runtime_binding']['verified'], 'Actual worker runtime binding failed')
                    else:
                        record['state'] = 'complete'
                except BaseException as caught:
                    record.update(state='failed', error={'class': type(caught).__name__, 'message': str(caught)})
                finally:
                    if stage == 'run' and context is not None:
                        try:
                            directory = output / case['donor']['subprocess']
                            collect_diagnostics(record, parser, output, directory,
                                REPORT / stage / request['id'], case['group_layout'], links)
                        except BaseException as caught:
                            record.update(state='failed', diagnostic_error={'class': type(caught).__name__, 'message': str(caught)})
                    if context is not None:
                        try:
                            record['readonly_guard'] = p.worker_postguard(value, case, output, context, helpers, record)
                            record['readonly_guard_passed'] = True
                        except BaseException as caught:
                            record.update(state='failed', readonly_guard_error={'class': type(caught).__name__, 'message': str(caught)})
                    try:
                        protected_inputs(p, helpers, case, source)
                        record['protected_guards_passed'] = True
                    except BaseException as caught:
                        record.update(state='failed', readonly_guard_passed=False,
                                      protected_guard_error={'class': type(caught).__name__, 'message': str(caught)})
                    results.append(record)
                if record.get('source_admission_error') or record.get('protected_guard_error'):
                    break
    except BaseException as caught:
        error = {'class': type(caught).__name__, 'message': str(caught)}
    finally:
        for request in requests()[len(results):]:
            results.append({'request': request, 'state': 'unexecuted', 'native_started': False,
                            'reason': 'No retry/replacement; prior fatal stage or protected-input failure'})
        if stage == 'run':
            try:
                dataset = point_metrics(p, value, results, helpers['plot_metrics'].relative_spread)
                unavailable = dataset['provenance']['unavailable_reasons']
                dataset['provenance'] = {
                'campaign_manifest_sha256': MANIFEST_SHA,
                'source_revision': read(MANIFEST)['candidate']['exporter_identity']['revision'],
                'generation_receipt_revision': case['donor']['generation_revision'],
                'source_index_sha256': source, 'protocol_sha256': digest(HERE / 'protocol.json'),
                'classifier_source_index_sha256': CLASSIFIER_INDEX_SHA,
                'preparation_sha256': digest(REPORT / 'prepare-terminal.json'),
                'points_sha256': value['points']['sha256'], 'point_count_per_case': 100,
                'scope': 'host forced-DP/card-selected backend internal consistency; no host accuracy or true-error qualification',
                'observable': 'finite/Born; common alpha_s/(2*pi) factor cancels in relative spread',
                'backend_observation': 'effective singleton numerical card; unchecked110 does not identify reducer/internal precision',
                    'unavailable_reasons': unavailable}
                p.save(REPORT / 'point-metrics.json', dataset)
            except BaseException as caught:
                metrics_error = {'class': type(caught).__name__, 'message': str(caught)}
        p.save(REPORT / (stage + '-terminal.json'), {'source_index_sha256': source,
            'protocol_sha256': digest(HERE / 'protocol.json'), 'preparation_sha256': digest(REPORT / 'prepare-terminal.json'),
            'status': 'passed' if error is None and metrics_error is None and all(row['state'] == 'complete' for row in results) else 'failed',
            'error': error, 'metrics_error': metrics_error, 'results': results, 'intended_points_per_backend': 100,
            'maximum_primary_public_calls': 1500, 'maximum_bootstrap_public_calls': 80,
            'point_generation_calls': 0, 'numerical_acceptance_claimed': False})
    require(error is None and metrics_error is None and all(row['state'] == 'complete' for row in results), 'Stage retains failures/unexecuted requests')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', nargs='?', default='plan', choices=('plan', 'readonly', 'prepare', 'dryrun', 'run'))
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if args.stage == 'plan' or (args.stage in ('prepare', 'run') and not args.execute):
        print(json.dumps({'stage': args.stage, 'execute': False, 'case': CASE, 'requests': requests(),
            'commands_executed': 0, 'work_directory': str(WORK)}, indent=2))
    elif args.stage == 'readonly':
        print(json.dumps(readonly(), indent=2, sort_keys=True, allow_nan=False))
    elif args.stage == 'prepare':
        prepare()
    else:
        workers(args.stage)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
