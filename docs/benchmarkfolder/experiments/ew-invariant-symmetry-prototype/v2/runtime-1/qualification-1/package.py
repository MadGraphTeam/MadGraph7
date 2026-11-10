#!/usr/bin/env python3
"""Retain one completed physical replay; default verifies bytes, --reparse only text."""
import argparse
import codecs
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
import tempfile
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parent
ROOT = RUNTIME.parents[5]
WORK_REL = '.tirust-build/ew-invariant-symmetry-prototype-v2/runtime-1/attempt1'
WORK = ROOT / WORK_REL
CAUSE_REL = 'docs/benchmarkfolder/experiments/dp-rescue-cause-observer'
OLD_REL = '.tirust-build/dp-rescue-cause-observer/cause-2-qualified/controller-native/attempt1/ew_n3_epem_ttxa/original'
REFERENCE_REL = 'docs/benchmarkfolder/campaigns/split-dp-terminal-1/experiments/numerical/full100-warmup-ew_n3_epem_ttxa-1'
LANES = ('baseline-1', 'candidate-1', 'candidate-6', 'candidate-7', 'candidate-8', 'candidate-9')
RESULTS_SHA = '803e76b3d767e0a108a9056b1251edc6769b7abf3c27816cf05ce9c58cab6d55'
QUALIFICATION_SHA = '95ef387372667b0aaa18e9399abdf0ed17759d3a937cddc0beafbf422cc1fb7d'
PREPARED_SHA = 'd38bb8d1e3a0a53accb481a8bb46f95c2b9b042436af8f351c54c81e8aa35b82'
RUNNER_SHA = '1234c24917827e456fdc8c985c4e0e89a5bc31c7acfd05be5136e175e10cad09'
OLD_INDEX_SHA = '9f934ced6cfe06d2f01252b8a86d6c5f50cd7bd29890fd2bf78b9e5db40769e2'
RUNTIME_REL = str(RUNTIME.relative_to(ROOT))
PARSERS = (RUNTIME_REL + '/run.py', CAUSE_REL + '/analyze.py', CAUSE_REL + '/protocol.py',
           'tools/tirust_benchmark.py', 'tools/tirust_benchmark_warmup.py', 'tools/tirust_benchmark_worker.py')
MAX_RAW = 16 * 1024**2


def need(value, message):
    if not value:
        raise ValueError(message)


def pin(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def relative(base, name):
    need(isinstance(name, str), 'Nontext relative name')
    item = PurePosixPath(name)
    need(item.parts and not item.is_absolute() and '..' not in item.parts and item.as_posix() == name,
         'Unsafe relative path')
    path = base.joinpath(*item.parts)
    need(path.resolve() == path and not path.is_symlink(), 'Aliased retained path')
    return path


def write(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def normalized(data):
    return json.loads(json.dumps(data, allow_nan=False))


def protected(prepared):
    need(len(prepared['pins']) == 2307, 'Wrong preparation inventory')
    for name, digest in prepared['pins'].items():
        path = Path(name)
        need(path.is_absolute() and any(path.is_relative_to(base) and path.resolve().is_relative_to(base)
             for base in (ROOT, Path('/nix/store'))), 'Protected input outside roots')
        need(pin(path)['sha256'] == digest, 'Protected input changed: ' + name)


def validate(execution, qualification, results, hashes):
    need(qualification['complete'] and qualification['regression_passed']
         and qualification['protected_inputs_unchanged'] and not qualification['performance_eligible'],
         'Runtime qualification did not complete')
    need(qualification['prepared_sha256'] == PREPARED_SHA and qualification['worker'] == execution,
         'Runtime qualification/worker identity differs')
    need(execution['complete'] and execution['process_spawned'] and execution['process_reaped']
         and execution['returncode'] == 0, 'Worker not complete and reaped')
    actual = execution['actual_runtime']
    need(actual['captured_before_stop'] and actual['cpu'] == 12
         and actual['maps_sha256'] == hashes['process-maps.txt'], 'Actual runtime maps not retained')
    need(actual['sha256'] == {**execution['runtime'], execution['executable']['path']: execution['executable']['sha256']},
         'Actual numerical runtime differs from pinned executable/dependencies')
    bridges = [v for p, v in execution['runtime'].items() if PurePosixPath(p).name == 'libmadloop_tirust.so']
    need(bridges == ['eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9'], 'Wrong installed bridge')
    need(set(hashes) == set(execution['output_sha256']) | {'execution.json', 'qualification.json', 'results.json'},
         'Runtime output coverage differs')
    for name, digest in execution['output_sha256'].items():
        need(hashes[name] == digest, 'Retained worker output differs: ' + name)
    for name, digest in execution['copied_inputs'].items():
        marker = '/' + WORK_REL + '/'
        need(marker in name and hashes[name.split(marker, 1)[1]] == digest, 'Copied numerical input changed')
    need(results['schema'] == 'ew-symmetry-full100-results-v2' and results['regression_passed']
         and results['ordinary_target_passed'] and results['input_words_exact']
         and results['previously_accepted_DP_words_exact'] and results['halting_and_rounding_unchanged'],
         'Full100 checks did not pass')
    need(results['status_counts'] == {'219': 100} and results['status_transitions'] ==
         [{'new': 219, 'old': 219, 'points': 82}, {'new': 219, 'old': 323, 'points': 18}], 'DP/QP status outcome differs')
    need(not results['ieee_state_differences'] and not results['warning_W03_lines']
         and not results['performance_eligible'], 'Unexpected IEEE/warning/performance claim')
    need(set(results['comparisons']) == set(LANES) and all(r['passed'] and r['ordinary_target_passed']
         and r['tolerance'] == 3e-6 and r['ordinary_target_tolerance'] == 1e-8
         for r in results['comparisons'].values()), 'Reference comparison coverage differs')
    return {'schema': 'ew-v2-full100-retained-summary-v1', 'physical_points': 100, 'public_calls_including_bootstrap': 101,
            'old_accepted_DP_points': 82, 'old_accepted_QP_points': 18, 'new_accepted_DP_points': 100,
            'new_accepted_QP_points': 0, 'all_six_reference_comparisons_passed': True,
            'comparator_thresholds': [3e-6, 1e-8], 'input_words_exact': True,
            'previously_accepted_DP_answer_and_accuracy_words_exact': True,
            'ieee_state_differences': 0, 'W03_lines': 0, 'worker_exit': 0, 'worker_reaped': True,
            'protected_inputs_unchanged': True, 'actual_loaded_runtime_bound': True,
            'results_sha256': RESULTS_SHA, 'qualification_sha256': QUALIFICATION_SHA,
            'comparisons': {name: {k: v for k, v in row.items() if k not in ('residuals', 'failures')}
                            for name, row in results['comparisons'].items()},
            'performance_eligible': False, 'global_campaign_qualified': False, 'new_native_execution': False,
            'scope': 'This exact frozen full100 EW stream, installed runtime and candidate; no other campaign/version or timing claim'}


def semantic(index, read_bytes):
    """Materialize exact frozen sources and text in a temporary repository, then call analyze()."""
    names = ('protocol', 'tirust_benchmark', 'tirust_benchmark_worker', 'tirust_benchmark_warmup')
    saved_modules = {name: sys.modules.get(name) for name in names}
    saved_path = list(sys.path)
    def module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec)
        sys.modules[name] = value
        spec.loader.exec_module(value)
        return value
    try:
        with tempfile.TemporaryDirectory(prefix='ew-v2-retained-', dir='/tmp') as temp:
            mirror = Path(temp) / 'repository'
            materialized = {}
            for name, row in index['evidence'].items():
                destination = row['replay_path']
                if destination is None:
                    continue
                path = relative(mirror, destination)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(read_bytes(name))
                need(pin(path)['sha256'] == row['raw_sha256'], 'Materialized replay bytes differ')
                materialized[destination] = row['raw_sha256']
            for name in names:
                sys.modules.pop(name, None)
            module('protocol', mirror / CAUSE_REL / 'protocol.py')
            public = module('retained_ew_public', mirror / CAUSE_REL / 'analyze.py')
            runtime = module('retained_ew_runtime', mirror / RUNTIME_REL / 'run.py')
            # Only supply the existing parser factory used by frozen analyze(); no lifecycle methods exist here.
            replay = runtime.analyze(mirror / WORK_REL, SimpleNamespace(analyzer_module=lambda: public))
            need(normalized(replay) == json.loads(read_bytes('run/results.json')), 'Frozen runtime analyzer replay differs')
            for name, digest in materialized.items():
                need(pin(mirror / name)['sha256'] == digest, 'Replay changed its frozen source or input bytes')
    finally:
        sys.path[:] = saved_path
        for name in names:
            if saved_modules[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved_modules[name]


def verify(reparse=False):
    index = json.loads((HERE / 'index.json').read_text())
    need(index['schema'] == 'ew-v2-portable-runtime-v1', 'Wrong package schema')
    need({str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file()}
         == set(index['files']) | {'index.json'}, 'Unindexed or missing package file')
    for name, value in index['files'].items():
        need(pin(relative(HERE, name)) == value, 'Retained package file changed')
    for name, value in index['repository_bindings'].items():
        need(name.startswith(('docs/benchmarkfolder/', 'tools/')), 'Nonportable repository binding')
        need(pin(relative(ROOT, name)) == value, 'Frozen repository binding changed')
    def read_bytes(name):
        row = index['evidence'][name]
        need(row['raw_bytes'] <= MAX_RAW, 'Raw payload exceeds bound')
        with gzip.open(relative(HERE, row['payload']), 'rb') as stream:
            data = stream.read(MAX_RAW + 1)
        need(len(data) == row['raw_bytes'] and hashlib.sha256(data).hexdigest() == row['raw_sha256'],
             'Raw retained bytes differ: ' + name)
        return data
    for name in index['evidence']:
        read_bytes(name)
    for name, digest in (('run/results.json', RESULTS_SHA), ('run/qualification.json', QUALIFICATION_SHA),
                         ('source/prepared.json', PREPARED_SHA), ('source/' + RUNTIME_REL + '/run.py', RUNNER_SHA)):
        need(index['evidence'][name]['raw_sha256'] == digest, 'Runtime anchor differs')
    hashes = {name[4:]: row['raw_sha256'] for name, row in index['evidence'].items() if name.startswith('run/')}
    summary = validate(*(json.loads(read_bytes('run/' + name)) for name in ('execution.json', 'qualification.json', 'results.json')), hashes)
    need(json.loads((HERE / 'summary.json').read_text()) == summary, 'Summary differs from actual execution')
    if reparse:
        semantic(index, read_bytes)
    return {'verified': True, 'semantic_reparse_performed': reparse, 'new_native_execution': False,
            'index_sha256': pin(HERE / 'index.json')['sha256'], 'files': len(index['files']) + 1,
            'text_payloads': len(index['evidence']), 'full100_EW_passed': True, 'performance_eligible': False}


def package():
    need(Path.cwd().resolve() == ROOT and not (HERE / 'index.json').exists()
         and not (HERE / 'packaging-start.json').exists(), 'Package once from repository root')
    need(pin(RUNTIME / 'prepared.json')['sha256'] == PREPARED_SHA and pin(RUNTIME / 'run.py')['sha256'] == RUNNER_SHA,
         'Runtime preparation/source changed')
    prepared = json.loads((RUNTIME / 'prepared.json').read_text())
    protected(prepared)
    bindings, sources, replay_paths = {}, {}, {}
    for name, digest in prepared['pins'].items():
        path = Path(name)
        if path.is_relative_to(ROOT):
            rel = str(path.relative_to(ROOT))
            if rel.startswith(('docs/benchmarkfolder/', 'tools/')):
                value = pin(path)
                need(value['sha256'] == digest, 'Frozen source/input drift')
                bindings[rel] = value
    for name in PARSERS:
        sources['source/' + name] = relative(ROOT, name)
        replay_paths['source/' + name] = name
    sources['source/prepared.json'] = RUNTIME / 'prepared.json'
    replay_paths['source/prepared.json'] = None
    for path in sorted(WORK.rglob('*')):
        if path.is_file():
            name = str(path.relative_to(WORK))
            sources['run/' + name] = relative(WORK, name)
            replay_paths['run/' + name] = WORK_REL + '/' + name
    for lane in LANES:
        for name in ('warmup.dat', 'dimensions.dat', 'ready.json'):
            rel = REFERENCE_REL + '/' + lane + '/raw/' + name
            sources['reference/' + lane + '/' + name] = relative(ROOT, rel)
            replay_paths['reference/' + lane + '/' + name] = rel
    archive = ROOT / CAUSE_REL / 'numerical-qualification-1'
    need(pin(archive / 'index.json')['sha256'] == OLD_INDEX_SHA, 'Original stream archive index changed')
    bindings[str((archive / 'index.json').relative_to(ROOT))] = pin(archive / 'index.json')
    old = json.loads((archive / 'index.json').read_text())
    original_bytes = {}
    for name in ('cause-public.txt', 'warmup.dat', 'points.dat', 'points.json'):
        row = old['inventory']['ew_n3_epem_ttxa/original/' + name]
        path = relative(archive, row['payload'])
        need(pin(path) == old['files'][row['payload']], 'Original archived payload changed')
        bindings[str(path.relative_to(ROOT))] = pin(path)
        with gzip.open(path, 'rb') as stream:
            data = stream.read(MAX_RAW + 1)
        need(len(data) == row['raw_bytes'] and hashlib.sha256(data).hexdigest() == row['raw_sha256'], 'Original raw stream changed')
        original_bytes['original/' + name] = data
        replay_paths['original/' + name] = OLD_REL + '/' + name
    for name, digest in (('build-qualification-1/index.json', '9749d31821172abb1cb5d8da6c45568b7728af4afcef7b0a6833f3d3bc74bb4e'),
                         ('fortran-fixtures/qualification-1/index.json', '62368b50b659f388a031f35528bd0cda6ae54c82ed751dd39c49a3df3db97300')):
        path = RUNTIME.parent / name
        need(pin(path)['sha256'] == digest, 'Prior v2 qualification changed')
        bindings[str(path.relative_to(ROOT))] = pin(path)
    values = {name: path.read_bytes() for name, path in sources.items()}
    values.update(original_bytes)
    for name in ('points.dat', 'points.json'):
        need(values['run/' + name] == values['original/' + name], 'Original full100 point stream differs')
    hashes = {name[4:]: hashlib.sha256(data).hexdigest() for name, data in values.items() if name.startswith('run/')}
    need(hashes['results.json'] == RESULTS_SHA and hashes['qualification.json'] == QUALIFICATION_SHA, 'Wrong completed runtime')
    summary = validate(*(json.loads(values['run/' + name]) for name in ('execution.json', 'qualification.json', 'results.json')), hashes)
    write(HERE / 'packaging-start.json', {'results_sha256': RESULTS_SHA, 'qualification_sha256': QUALIFICATION_SHA,
          'packager_sha256': pin(Path(__file__))['sha256'], 'new_native_execution': False})
    evidence = {}
    try:
        for name, data in values.items():
            need(len(data) <= MAX_RAW and data[:4] not in (b'\x7fELF', b'!<ar'), 'Binary/oversized payload forbidden')
            data.decode('utf-8')
            path = relative(HERE, 'evidence/' + name + '.gz')
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as out, gzip.GzipFile(filename='', mode='wb', fileobj=out, mtime=0, compresslevel=6) as compressed:
                compressed.write(data)
            evidence[name] = {'payload': str(path.relative_to(HERE)), 'raw_sha256': hashlib.sha256(data).hexdigest(),
                              'raw_bytes': len(data), 'replay_path': replay_paths[name]}
        candidate_index = {'evidence': evidence}
        semantic(candidate_index, lambda name: gzip.decompress(relative(HERE, evidence[name]['payload']).read_bytes()))
        for name, path in sources.items():
            need(pin(path)['sha256'] == evidence[name]['raw_sha256'], 'Runtime source/output changed during retention')
        protected(prepared)
        for name, value in bindings.items():
            need(pin(relative(ROOT, name)) == value, 'Bound repository evidence changed during retention')
        write(HERE / 'summary.json', summary)
        files = {str(p.relative_to(HERE)): pin(p) for p in sorted(HERE.rglob('*')) if p.is_file()}
        write(HERE / 'index.json', {'schema': 'ew-v2-portable-runtime-v1', 'files': files, 'evidence': evidence,
              'repository_bindings': bindings, 'codec': 'gzip;mtime=0;filename-empty;level=6',
              'binary_payloads': 0, 'new_native_execution': False, 'performance_eligible': False})
    except BaseException as error:
        try:
            write(HERE / 'packaging-failure.json', {'error': repr(error), 'partial_outputs_preserved': True,
                  'automatic_retry': False, 'new_native_execution': False})
        except BaseException as other:
            error.add_note('Failure retention error: ' + repr(other))
        raise
    return verify()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--package', action='store_true')
    modes.add_argument('--reparse', action='store_true')
    args = parser.parse_args()
    print(json.dumps(package() if args.package else verify(args.reparse), indent=2, sort_keys=True))
