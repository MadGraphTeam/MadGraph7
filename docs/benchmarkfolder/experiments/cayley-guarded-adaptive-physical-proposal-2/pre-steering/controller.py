"""Reuse the qualified checked executable; bind a real future DSO receipt."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import signal
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WORK = ROOT / '.tirust-build/cayley-guarded-adaptive-physical-proposal-2'
RUN = WORK / 'run1'
OLD = ROOT / 'docs/benchmarkfolder/experiments/adaptive-qp-checked-trap-proposal-1'
BUILD_PROPOSAL = ROOT / 'docs/benchmarkfolder/experiments/cayley-guarded-adaptive-build-proposal-3'
BUILD_RECEIPT = ROOT / '.tirust-build/cayley-guarded-adaptive-build-proposal-3/attempt1/execution.json'

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def modules():
    return (load('frozen_checked_controller', OLD / 'controller.py'),
            load('frozen_checked_journal', OLD / 'parser.py'),
            load('frozen_named_ieee_controls', OLD / 'control_log.py'),
            load('guarded_replay_assessment', HERE / 'assessment.py'))

def verify():
    assert Path.cwd().resolve() == ROOT
    index = json.loads((HERE / 'source-index.json').read_text())
    for name, digest in index['sources'].items():
        assert sha(HERE / name) == digest, name
    actual = {str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file()
              and p.name != 'source-index.json' and '__pycache__' not in p.parts}
    assert actual == set(index['sources'])
    for name, digest in index['dependencies'].items():
        assert sha(ROOT / name) == digest, name
    old, _, _, _ = modules()
    old.built()
    return json.loads((HERE / 'prepared.json').read_text())

def bind():
    """Read/hash/ELF-decode only. No runtime copying, loader or lock."""
    spec = verify()
    assert not (WORK / 'binding1.json').exists(), 'One immutable artifact binding only'
    receipt_bytes = BUILD_RECEIPT.read_bytes()
    receipt = json.loads(receipt_bytes)
    plan = json.loads((BUILD_PROPOSAL / 'plan.json').read_text())
    old, _, _, assessment = modules()
    library = assessment.accept_build_receipt(receipt,
        source_index_path=str(BUILD_PROPOSAL / 'source-index.json'),
        source_index_sha256=spec['expected_build_source_index_sha256'], build_command=plan['build_command'])
    assert receipt['environment'] == plan['environment']
    source_root = BUILD_RECEIPT.parents[1] / 'sources'
    for name, relative in [('iregi_rust', 'pole-development/crates/iregi-rust/Cargo.toml'),
                           ('madloop_tirust_adaptive_private', 'adaptive-qp-rescue-prototype3/Cargo.toml')]:
        records = [r for r in receipt['cargo_artifacts'] if r['target']['name'] == name]
        assert len(records) == 1
        artifact = records[0]
        assert Path(artifact['manifest_path']).resolve() == (source_root / relative).resolve()
        assert artifact['fresh'] is False and artifact['profile']['test'] is False and artifact['profile']['opt_level'] == '3'
    source = Path(library['path'])
    assert source.resolve() == (BUILD_RECEIPT.parent / 'runtime/libmadloop_tirust_adaptive_private.so').resolve()
    assert source.stat().st_size == library['bytes'] and sha(source) == library['sha256']
    elf = load('frozen_literal_elf_reader', old.ELF)
    metadata = elf.inspect(source)
    adaptive = json.loads((OLD / 'prepared.json').read_text())['bridge_metadata']['adaptive']
    for field in ('needed', 'version_requirements', 'exported_ml_tirust_symbols'):
        assert metadata['metadata'][field] == adaptive['metadata'][field], field
    assert sha(BUILD_RECEIPT) == hashlib.sha256(receipt_bytes).hexdigest()
    binding = {'schema': 'guarded-physical-artifact-binding-v1',
               'source_index_sha256': sha(HERE / 'source-index.json'),
               'build_receipt_path': str(BUILD_RECEIPT), 'build_receipt_sha256': sha(BUILD_RECEIPT),
               'library': library, 'literal_ELF_metadata': metadata,
               'read_hash_ELF_decode_only': True, 'loader_or_native_executed': False}
    old.save(WORK / 'binding1.json', binding)
    print(json.dumps({'bound_actual_library': library, 'loader_or_native_executed': False}, indent=2))

def bound():
    spec = verify()
    binding = json.loads((WORK / 'binding1.json').read_text())
    assert binding['source_index_sha256'] == sha(HERE / 'source-index.json')
    assert sha(BUILD_RECEIPT) == binding['build_receipt_sha256']
    path = Path(binding['library']['path'])
    assert sha(path) == binding['library']['sha256'] and path.stat().st_size == binding['library']['bytes']
    return spec, binding

def run():
    old, _, _, _ = modules()
    return old.timed_call(run_stage, 600)

def run_stage(start, deadline):
    spec, binding = bound()
    old, parser, controls, assessment = modules()
    frozen_spec, frozen_build = old.built()
    helper, adapted_hash = old.supervisor()  # CPU12 and SIGALRM deferral already reviewed.
    with (ROOT / '.tirust-build/numerical-tests.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(75)
        assert not RUN.exists(), 'One runtime attempt only; no retry'
        RUN.mkdir()
        old.limits()
        record = {'schema': 'guarded-checked-fixed-trap-replay-v1', 'execution_complete': False,
                  'source_index_sha256': sha(HERE / 'source-index.json'),
                  'artifact_binding_sha256': sha(WORK / 'binding1.json'),
                  'guarded_library_sha256': binding['library']['sha256'],
                  'unchanged_checked_executable_sha256': sha(old.EXE),
                  'unchanged_numerical_objects': frozen_spec['unchanged_numerical_objects'],
                  'cpu': 12, 'cpu_exclusive': False, 'address_space_bytes': 4294967296,
                  'automatic_retry': False, 'aggregate_seconds': 600, 'workers': {},
                  'adapted_supervisor_sha256': adapted_hash,
                  'checked_trap_gate_qualified': False, 'forced_qp_trap_gate_qualified': False,
                  'physical_truth_or_performance_qualified': False, 'per_call_precision_qualified': False}
        expected = {}
        captures = {}
        pending = {'checked': False, 'forced-qp': False}
        runtime = RUN / 'runtime-guarded'
        def checkpoint():
            old.save(RUN / 'execution.json', record)
        try:
            assert old.interpreter(old.EXE) == frozen_spec['loader']
            runtime.mkdir()
            for name, info in frozen_spec['runtime_inputs'].items():
                if name == 'ld-linux-x86-64.so.2':
                    expected[str(Path(info['path']).resolve())] = info['sha256']
                    continue
                donor = Path(binding['library']['path']) if name == 'libmadloop_tirust.so' else Path(info['path'])
                target = runtime / name
                shutil.copyfile(donor, target)
                expected[str(target.resolve())] = sha(target)
                assert sha(target) == (binding['library']['sha256'] if name == 'libmadloop_tirust.so' else info['sha256'])
            assert len(expected) == 15
            env = old.base_environment(frozen_spec)
            env['LD_LIBRARY_PATH'] = str(runtime)
            loader = {'argv': [frozen_spec['loader'], '--list', str(old.EXE)], 'progress': {}, 'environment': env}
            record['loader'] = loader
            checkpoint()
            with (RUN / 'loader.log').open('xb') as log:
                loader['exit_code'] = helper['run_plain_capped'](loader['argv'], log, cwd=ROOT, env=env,
                    progress=loader['progress'], seconds=min(10, old.remaining(deadline, 5)))
            assert loader['exit_code'] == 0
            loader['closure'] = old.listed((RUN / 'loader.log').read_text(), expected)
            jobs = [('fault-' + n, 'fault', n, None) for n in spec['fault_controls']]
            jobs += [(mode + '-' + policy, 'physical', policy, mode)
                     for mode in ('checked', 'forced-qp') for policy in ('masked', 'trapped')]
            for name, kind, policy, mode in jobs:
                bound()
                old.remaining(deadline, 5)
                directory = RUN / name
                directory.mkdir()
                if kind == 'physical':
                    for source in (HERE / 'fixtures' / mode).rglob('*'):
                        if source.is_file():
                            target = directory / source.relative_to(HERE / 'fixtures' / mode)
                            target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(source, target)
                before = {str(p.relative_to(directory)): sha(p) for p in directory.rglob('*') if p.is_file()}
                argv = [str(old.EXE), '--fault', policy] if kind == 'fault' else [str(old.EXE), str(directory / 'selected.dat'), policy]
                row = {'kind': kind, 'mode': mode, 'policy': policy, 'argv': argv, 'progress': {},
                       'expected_runtime_sha256': expected, 'inputs_before': before, 'environment': env}
                record['workers'][name] = row
                checkpoint()
                capture_maps = helper['capture_maps']
                def acknowledge(*args, **kwargs):
                    mapping = capture_maps(*args, **kwargs)
                    if mapping is not None and not (directory / 'mapping-ready').exists():
                        (directory / 'mapping-ready').write_text('Complete actual ELF mapping captured\n')
                    return mapping
                helper['capture_maps'] = acknowledge
                try:
                    with (directory / 'stdout.log').open('xb') as log:
                        row['exit_code'], row['actual_mapping'] = helper['run_captured'](argv, log,
                            cwd=directory, env=env, expected_runtime=expected, executable=str(old.EXE.resolve()),
                            selected_library=str((runtime / 'libmadloop_tirust.so').resolve()),
                            maps_path=directory / 'native-maps.txt', progress=row['progress'],
                            seconds=min(10 if kind == 'fault' else 120, old.remaining(deadline, 5)))
                except BaseException as error:
                    if isinstance(error, InterruptedError):
                        raise
                    row['process_failure'] = {'type': type(error).__name__, 'message': str(error)}
                finally:
                    helper['capture_maps'] = capture_maps
                    after = {str(p.relative_to(directory)): sha(p) for p in directory.rglob('*') if p.is_file()}
                    row['files_after'] = after
                    row['changed_inputs'] = {p: {'before': h, 'after': after.get(p)} for p, h in before.items() if after.get(p) != h}
                    checkpoint()
                assert not row['changed_inputs']
                code = row.get('exit_code', row['progress'].get('returncode'))
                row['native_output_closure'] = old.output_closure(before, after, kind, code)
                try:
                    named = controls.parse(directory / 'checked-control.log', allow_partial=code != 0)
                    row['control_assessment'] = controls.assess(named, exit_code=code)
                    old.save(directory / 'named-control.json', named)
                    if kind == 'physical':
                        journal = parser.parse(directory / 'mass-observer.hex', allow_partial=code != 0)
                        captures[name] = journal
                        old.save(directory / 'journal-summary.json', parser.summarize(journal))
                        omissions = sum('##W03' in line for line in (directory / 'stdout.log').read_text().splitlines())
                        row['omissions'] = omissions
                        row['numerical_assessment'] = (parser.qualify_checked(journal, named['observed'], omissions=omissions)
                            if mode == 'checked' else assessment.qualify_fixed(journal, named['observed'], parser, omissions=omissions))
                except Exception as error:
                    row['offline_assessment_failure'] = {'type': type(error).__name__, 'message': str(error)}
                checkpoint()
            comparisons = {}
            for mode in ('checked', 'forced-qp'):
                left, right = mode + '-masked', mode + '-trapped'
                if left in captures and right in captures:
                    inputs = parser.compare_inputs_and_born(captures[left], captures[right])
                    outputs = old.compare_policy_outputs(captures[left], captures[right])
                    comparisons[mode + '_policy_inputs'] = inputs
                    comparisons[mode + '_policy_outputs'] = outputs
                    gate = 'checked_outputs_qualified' if mode == 'checked' else 'fixed_qp_outputs_qualified'
                    pending[mode] = inputs['inputs_and_born_exact'] and inputs['all_input_records_comparable'] and outputs['outputs_exact'] and all(
                        record['workers'][key].get('numerical_assessment', {}).get(gate, False) and
                        record['workers'][key].get('control_assessment', {}).get('physical_worker_completed_with_requested_IEEE_policy', False)
                        for key in (left, right))
                    if mode == 'checked':
                        pending[mode] &= all(record['workers'][key]['numerical_assessment'].get('qp_coverage_observed', False) for key in (left, right))
                for reference in spec['comparators']:
                    if reference['mode'] != mode or left not in captures:
                        continue
                    previous = parser.parse(ROOT / reference['journal'])
                    key = reference['label'] + '__' + left
                    comparisons[key] = {'inputs_and_born': parser.compare_inputs_and_born(previous, captures[left]),
                                        'previously_served': assessment.previously_served_preserved(previous, captures[left], parser)}
                    pending[mode] &= comparisons[key]['previously_served']['preserved_exact'] and comparisons[key]['inputs_and_born']['inputs_and_born_exact'] and comparisons[key]['inputs_and_born']['all_input_records_comparable']
                    if reference['requires_all_outputs_exact']:
                        comparisons[key]['outputs'] = old.compare_policy_outputs(previous, captures[left])
                        pending[mode] &= comparisons[key]['outputs']['outputs_exact'] and comparisons[key]['inputs_and_born']['inputs_and_born_exact'] and comparisons[key]['inputs_and_born']['all_input_records_comparable']
            old.save(RUN / 'comparisons.json', comparisons)
            record['all_trap_controls_demonstrated'] = all(record['workers']['fault-' + n].get('control_assessment', {}).get('fault_control_demonstrated', False) for n in spec['fault_controls'])
        except BaseException as error:
            record['failure'] = {'type': type(error).__name__, 'message': str(error)}
        finally:
            def closure():
                bound()
                assert {p: sha(p) for p in expected} == expected
                record['protected_inputs_after'] = 'unchanged'
                record['runtime_sha256_after'] = expected
                for name, row in record['workers'].items():
                    row['complete_case_files_sha256'] = {str(p.relative_to(ROOT)): sha(p) for p in (RUN / name).rglob('*') if p.is_file()}
                record['execution_complete'] = old.transport_complete(record['workers'], top_failure='failure' in record)
                valid = record['execution_complete'] and record.get('all_trap_controls_demonstrated', False)
                record['checked_trap_gate_qualified'] = bool(valid and pending['checked'])
                record['forced_qp_trap_gate_qualified'] = bool(valid and pending['forced-qp'])
            old.publish_final(RUN / 'execution.json', record, deadline, start, 'execution_complete', closure)
            if not record['execution_complete']:
                record['checked_trap_gate_qualified'] = record['forced_qp_trap_gate_qualified'] = False
                old.save(RUN / 'execution.json', record)
        print(json.dumps({k: record[k] for k in ('execution_complete', 'checked_trap_gate_qualified', 'forced_qp_trap_gate_qualified')}, indent=2))
        if not record['execution_complete']:
            raise SystemExit(1)

if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action', choices=('verify', 'bind', 'run'))
    action = cli.parse_args().action
    if action == 'verify':
        print(json.dumps({'verified': bool(verify()), 'compiler_loader_or_native_executed': False}))
    elif action == 'bind':
        bind()
    else:
        run()
