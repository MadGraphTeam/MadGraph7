#!/usr/bin/env python3
"""Retain a completed, passing fixture once; verify or reparse text without workloads."""
import argparse
import codecs
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import runpy
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
FIXTURE = HERE.parent
ROOT = FIXTURE.parents[5]
WORK = ROOT / '.tirust-build/ew-invariant-symmetry-fixtures-v2/attempt1'
EXECUTION_SHA = 'bc0049e3c2212f71d3372ff772ec1b47737ff73b9fc4c205530691ab8ba4dc91'
SOURCE_SHA = 'fe0f30e8503c959beeac34d314f5b8434f562746bc4d73d90f83d32d9b5cb44b'
ANALYZER_SHA = '602fec89e78ccd22105d21fb323e93608ab35ce0e86b01c751dfeb2933c329ed'
PLAN_SHA = '83401e62657d078b6fa116ae719b2e503346ccdb9e8b0561fd7df93bbf9bca37'
MAX_RAW = 64 * 1024**2


def need(value, message):
    if not value:
        raise ValueError(message)


def pin(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def relative(base, name):
    need(isinstance(name, str), 'Relative path must be text')
    item = PurePosixPath(name)
    need(item.parts and not item.is_absolute() and '..' not in item.parts
         and item.as_posix() == name, 'Unsafe relative path')
    path = base.joinpath(*item.parts)
    need(path.resolve() == path and not path.is_symlink(), 'Aliased retained path')
    return path


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def normalized(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def protected_current(record):
    for name, expected in record['protected_before'].items():
        path = ROOT / name
        resolved = path.resolve(strict=True)
        need(any(path.is_relative_to(base) and resolved.is_relative_to(base)
                 for base in (ROOT, Path('/nix/store'))), 'Protected path escaped roots')
        need(pin(path)['sha256'] == expected, 'Protected input changed: ' + name)


def source_bindings():
    index_path = FIXTURE / 'source-index.json'
    need(pin(index_path)['sha256'] == SOURCE_SHA, 'Frozen source index changed')
    source = json.loads(index_path.read_text())
    need(len(source['files']) == 15, 'Wrong frozen source coverage')
    result = {str(index_path.relative_to(ROOT)): pin(index_path)}
    for name, expected in source['files'].items():
        path = relative(FIXTURE, name)
        value = {'sha256': expected['sha256'], 'bytes': expected['bytes']}
        need(pin(path) == value, 'Frozen source payload changed')
        result[str(path.relative_to(ROOT))] = value
    plan = json.loads((FIXTURE / 'plan.json').read_text())
    need(pin(FIXTURE / 'plan.json')['sha256'] == PLAN_SHA, 'Wrong fixture plan')
    for row in plan['inputs']:
        if row['path'].startswith('docs/benchmarkfolder/'):
            path = relative(ROOT, row['path'])
            value = {'sha256': row['sha256'], 'bytes': row['size']}
            need(pin(path) == value, 'Frozen repository source changed')
            result[row['path']] = value
    top_path = FIXTURE.parent / 'index.json'
    need(pin(top_path)['sha256'] == 'd3c1e4212669d66846195bf65af0a05153ce406c87ac12cb30cc58a612b3bb93',
         'Frozen v2 adapter index changed')
    result[str(top_path.relative_to(ROOT))] = pin(top_path)
    prior = FIXTURE.parents[1] / 'fortran-fixtures/qualification-1/index.json'
    need(pin(prior)['sha256'] == source['prior_failed_index_sha256'], 'Prior failure archive binding differs')
    result[str(prior.relative_to(ROOT))] = pin(prior)
    return result


def validate(record):
    need(record['schema'] == 'ew-invariant-compiled-fixture-execution-v2'
         and record['status'] == 'complete' and record['passed'] is True,
         'Expected completed fixture with passing component parity')
    need(record['plan_sha256'] == PLAN_SHA, 'Execution used a different plan')
    need(record['protected_before'] == record['protected_after']
         and len(record['protected_before']) == 139, 'Protected source drift')
    need(record['matrix_and_ieee_parity_passed'] is True
         and record['supported_trap_parity_passed'] is True, 'Component parity did not pass')
    for key in ('reducers', 'physical_amplitude', 'performance_eligible'):
        need(record[key] is False, 'Historical scope changed')
    commands = record['commands']
    need(len(commands) == 124 and all(c['process_spawned'] and c['process_reaped'] for c in commands),
         'Incomplete owned command lifecycle')
    need(Counter(c['returncode'] for c in commands) == {0: 92, -8: 32}, 'Command exits differ')
    by_label = {c['label']: c for c in commands}
    need(len(by_label) == 124, 'Duplicate command label')
    for label, command in by_label.items():
        need(PurePosixPath(command['stdout']).name == label + '.stdout', 'Log identity differs')
    need(len(record['bulk']) == 4 and len(record['traps']) == 48
         and len(record['mutations']) == 2 and len(record['artifacts']) == 6
         and len(record['loader_lists']) == 6, 'Fixture coverage differs')
    return by_label


def semantic(record, read_text):
    """Replay only the frozen analyzer's integer/text functions, never the runner."""
    commands = validate(record)
    analyzer_path = FIXTURE / 'analyze.py'
    before = pin(analyzer_path)
    need(before['sha256'] == ANALYZER_SHA, 'Frozen analyzer changed')
    try:
        analyzer = runpy.run_path(str(analyzer_path), run_name='retained_fixture_analysis')
        bulk, traps, mutations = [], [], []
        for row in record['bulk']:
            label = row['config'] + '-' + row['precision'] + '-bulk'
            result = analyzer['bulk'](read_text(label + '.stdout'), 8 if row['precision'] == 'dp' else 16)
            actual = normalized({'config': row['config'], 'precision': row['precision'], **result})
            need(actual == row, 'Retained bulk reparse differs: ' + label)
            problems = Counter(p for failure in result['failures'] for p in failure['problems'])
            need(result['pairs'] == 1344 and result['passed'] is True
                 and not result['failures'] and not problems, 'Bulk component parity did not pass')
            bulk.append({'config': row['config'], 'precision': row['precision'],
                         'pairs': result['pairs'], 'frames': result['frames'],
                         'failed_pairs': len(result['failures']), 'problems': dict(problems),
                         'changed_matrix_pairs': result['changed_matrix_pairs'],
                         'matrix_policy_passed': True, 'caller_input_words_unchanged': True,
                         'halting_and_rounding_unchanged': True,
                         'ieee_state_parity_passed': True,
                         'rounding_support': result['rounding_support'], 'skipped_flags': result['skipped_flags']})
        for row in record['traps']:
            keys = ('config', 'precision', 'route', 'case', 'rounding', 'trap')
            prefix = '{}-{}-trap-{}-{}-{}-{}'.format(*(row[k] for k in keys))
            a, b = commands[prefix + '-0'], commands[prefix + '-1']
            result = analyzer['trap'](read_text(prefix + '-0.stdout'), read_text(prefix + '-1.stdout'),
                                      a['returncode'], b['returncode'])
            need(normalized({**{k: row[k] for k in keys}, **result}) == row,
                 'Retained trap reparse differs: ' + prefix)
            traps.append(row)
        for row in record['mutations']:
            result = analyzer['bulk'](read_text('historical-flags-' + row['mutation'] + '-bulk.stdout'), 8)
            discriminates = any('matrix policy differs' in f['problems'] for f in result['failures'])
            need(normalized({'mutation': row['mutation'], 'detected_matrix_failure': discriminates, **result}) == row,
                 'Retained mutation reparse differs')
            mutations.append({'mutation': row['mutation'], 'detected_matrix_failure': discriminates,
                              'pairs': result['pairs'], 'failed_pairs': len(result['failures']),
                              'problems': dict(Counter(p for f in result['failures'] for p in f['problems']))})
        need(all(r['supported'] and r['equivalent'] for r in traps), 'Trap parity did not pass')
        returned = [r for r in traps if r['returned']]
        terminated = [r for r in traps if not r['returned']]
        need(len(returned) == 32 and all(r['trap'] == 1 and r['comparison']['passed'] for r in returned),
             'Invalid-trap returned comparison differs')
        need(len(terminated) == 16 and all(r['original_exit'] == r['candidate_exit'] == -8
             and r['same_input_and_entry_state'] and r['trap'] == 5 for r in terminated), 'Inexact-trap outcome differs')
        need(all(r['detected_matrix_failure'] for r in mutations), 'Negative controls did not discriminate')
        return normalized({'schema': 'ew-invariant-fixture-retained-reparse-v2',
            'execution_sha256': EXECUTION_SHA, 'analyzer_sha256': ANALYZER_SHA,
            'semantic_reparse_matches_execution': True, 'fixture_status': 'complete', 'fixture_passed': True,
            'expected_controller_exit': 0, 'controller_exit_basis': 'Frozen run.py returns after completed passed=true',
            'owned_commands_reaped': 124, 'command_exits': {'0': 92, '-8': 32},
            'protected_inputs_unchanged': 139, 'bulk': bulk,
            'traps': {'pairs': 48, 'supported': 48, 'mismatched': 0,
                      'matching_returned_invalid_pairs': 32,
                      'matching_sigfpe_inexact_pairs': 16, 'post_call_evidence_for_trapped_calls': False},
            'mutations': mutations, 'scope': 'Extracted local components and synthetic inputs, not full adapter transport',
            'reducers': False, 'physical_amplitude': False, 'full_adapter_qualified': False,
            'performance_eligible': False, 'new_native_execution': False})
    finally:
        need(pin(analyzer_path) == before, 'Analyzer changed during read-only reparse')


def verify(reparse=False):
    index = json.loads((HERE / 'index.json').read_text())
    need(index['schema'] == 'ew-invariant-fixture-component-package-v2', 'Wrong package schema')
    need({str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file()}
         == set(index['files']) | {'index.json'}, 'Unindexed or missing retained file')
    for name, expected in index['files'].items():
        need(pin(relative(HERE, name)) == expected, 'Retained file changed')
    for name, expected in index['repository_bindings'].items():
        need(name.startswith('docs/benchmarkfolder/'), 'Nonportable repository binding')
        need(pin(relative(ROOT, name)) == expected, 'Frozen repository binding changed')
    for name, row in index['evidence'].items():
        size, digest = 0, hashlib.sha256()
        with gzip.open(relative(HERE, row['payload']), 'rb') as stream:
            for chunk in iter(lambda: stream.read(1024**2), b''):
                size += len(chunk)
                need(size <= row['raw_bytes'] <= MAX_RAW, 'Raw text exceeds bound')
                digest.update(chunk)
        need(size == row['raw_bytes'] and digest.hexdigest() == row['raw_sha256'], 'Raw text differs: ' + name)
    def read_text(name):
        with gzip.open(relative(HERE, index['evidence'][name]['payload']), 'rt', encoding='utf-8') as stream:
            return stream.read()
    need(index['evidence']['execution.json']['raw_sha256'] == EXECUTION_SHA, 'Wrong execution manifest')
    record = json.loads(read_text('execution.json'))
    commands = validate(record)
    need(set(index['evidence']) == {'execution.json', 'progress.json'}
         | {label + '.stdout' for label in commands}, 'Missing command log coverage')
    for label, command in commands.items():
        row = index['evidence'][label + '.stdout']
        need((row['raw_sha256'], row['raw_bytes']) == (command['stdout_sha256'], command['stdout_size']),
             'Command log metadata differs')
    artifacts = json.loads((HERE / 'artifact-hashes.json').read_text())
    need(len(artifacts['files']) == 16 and artifacts['binary_payloads'] == 0, 'Artifact metadata coverage differs')
    for path, digest in record['artifacts'].items():
        name = '/'.join(PurePosixPath(path).parts[-2:])
        need(artifacts['files'][name]['sha256'] == digest, 'Executable metadata differs')
    summary = json.loads((HERE / 'summary.json').read_text())
    need(summary['fixture_passed'] is True and summary['execution_sha256'] == EXECUTION_SHA,
         'Summary changed historical outcome')
    if reparse:
        need(semantic(record, read_text) == summary, 'Summary differs from frozen analyzer replay')
    return {'verification_passed': True, 'semantic_reparse_performed': reparse,
            'fixture_passed': True, 'new_native_execution': False,
            'index_sha256': pin(HERE / 'index.json')['sha256'], 'files': len(index['files']) + 1,
            'text_payloads': len(index['evidence'])}


def package():
    need(Path.cwd().resolve() == ROOT and not (HERE / 'index.json').exists()
         and not (HERE / 'packaging-start.json').exists(), 'Package once from repository root')
    manifest = WORK / 'execution.json'
    need(pin(manifest)['sha256'] == EXECUTION_SHA, 'Completed fixture record changed')
    record = json.loads(manifest.read_text())
    commands = validate(record)
    bindings = source_bindings()
    protected_current(record)
    artifact_paths = sorted(WORK.glob('*/*.o')) + sorted(Path(p) for p in record['artifacts'])
    need(len(artifact_paths) == len(set(artifact_paths)) == 16, 'Expected ten objects and six executables')
    artifacts = {}
    for path in artifact_paths:
        name = str(path.relative_to(WORK))
        need(path.resolve() == path, 'Artifact path aliases')
        value = pin(path)
        need(str(path) not in record['artifacts'] or value['sha256'] == record['artifacts'][str(path)], 'Executable changed')
        artifacts[name] = value
    sources = {'execution.json': manifest, 'progress.json': WORK / 'progress.json'}
    for label, command in commands.items():
        path = relative(WORK, label + '.stdout')
        need(str(path) == command['stdout'], 'Command log outside attempt')
        need(pin(path) == {'sha256': command['stdout_sha256'], 'bytes': command['stdout_size']}, 'Command output changed')
        sources[label + '.stdout'] = path
    originals = {name: pin(path) for name, path in sources.items()}
    write(HERE / 'packaging-start.json', {'execution_sha256': EXECUTION_SHA,
          'packager_sha256': pin(Path(__file__))['sha256'], 'new_native_execution': False})
    evidence = {}
    try:
        for name, path in sources.items():
            expected = originals[name]
            need(expected['bytes'] <= MAX_RAW, 'Text evidence exceeds bound')
            target = relative(HERE, 'evidence/' + name + '.gz')
            target.parent.mkdir(parents=True, exist_ok=True)
            digest, size = hashlib.sha256(), 0
            decoder = codecs.getincrementaldecoder('utf-8')('strict')
            with path.open('rb') as src, target.open('xb') as out:
                need(src.read(4) not in (b'\x7fELF', b'!<ar'), 'Binary/core payload forbidden')
                src.seek(0)
                with gzip.GzipFile(filename='', mode='wb', fileobj=out, mtime=0, compresslevel=6) as compressed:
                    for chunk in iter(lambda: src.read(1024**2), b''):
                        size += len(chunk)
                        need(size <= expected['bytes'], 'Text changed while copying')
                        decoder.decode(chunk); digest.update(chunk); compressed.write(chunk)
                    decoder.decode(b'', final=True)
            need(size == expected['bytes'] and digest.hexdigest() == expected['sha256'], 'Raw text changed')
            evidence[name] = {'payload': str(target.relative_to(HERE)), 'raw_bytes': size, 'raw_sha256': digest.hexdigest()}
        def read_text(name):
            with gzip.open(relative(HERE, evidence[name]['payload']), 'rt', encoding='utf-8') as stream:
                return stream.read()
        summary = semantic(record, read_text)
        for name, path in sources.items():
            need(pin(path) == originals[name], 'Original text changed during retention')
        for name, value in artifacts.items():
            need(pin(relative(WORK, name)) == value, 'Artifact changed during retention')
        protected_current(record)
        for name, value in bindings.items():
            need(pin(relative(ROOT, name)) == value, 'Frozen source changed during retention')
        write(HERE / 'artifact-hashes.json', {'files': artifacts, 'binary_payloads': 0, 'core_payloads': 0,
              'scope': 'Six executable hashes match execution; ten object hashes observed during retention'})
        write(HERE / 'summary.json', summary)
        files = {str(p.relative_to(HERE)): pin(p) for p in sorted(HERE.rglob('*')) if p.is_file()}
        write(HERE / 'index.json', {'schema': 'ew-invariant-fixture-component-package-v2', 'files': files,
              'evidence': evidence, 'repository_bindings': bindings, 'execution_sha256': EXECUTION_SHA,
              'fixture_passed': True, 'codec': 'gzip;mtime=0;filename-empty;level=6', 'binary_payloads': 0,
              'core_payloads': 0, 'new_native_execution': False})
    except BaseException as error:
        try:
            write(HERE / 'packaging-failure.json', {'error': repr(error), 'partial_evidence_preserved': True,
                  'automatic_retry': False, 'new_native_execution': False})
        except BaseException as retention_error:
            error.add_note('Packaging metadata failed: ' + repr(retention_error))
        raise
    return verify()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--package', action='store_true', help='Retain this completed attempt once')
    modes.add_argument('--reparse', action='store_true', help='Verify and replay frozen text analyzer only')
    args = parser.parse_args()
    print(json.dumps(package() if args.package else verify(args.reparse), indent=2, sort_keys=True))
