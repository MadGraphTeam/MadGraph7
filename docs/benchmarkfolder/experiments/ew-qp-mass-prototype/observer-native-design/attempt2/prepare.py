#!/usr/bin/env python3
"""Create append-only attempt 2 snapshots; never compile or execute a binary.

Default invocation reports the plan. --prepare copies already-built attempt 1
artifacts and creates explicit driver-only rebuild and card-smoke recipes.
"""
import argparse
import json
from pathlib import Path
import shutil
import struct
import sys

sys.dont_write_bytecode = True
import control as c

HERE, ROOT, WORK, PRIOR = c.HERE, c.ROOT, c.WORK, c.PRIOR
OLDWORK = WORK.parent
PROCESS = Path('SubProcesses/P0_epem_ttxa')


def read_checked(path, expected=None):
    if expected is not None:
        c.require(c.digest(path) == expected, 'Prior evidence changed: ' + str(path))
    return json.loads(path.read_text())


def promoted_words(value):
    word = struct.unpack('<Q', struct.pack('<d', value))[0]
    exponent, fraction = (word >> 52) & 2047, word & ((1 << 52)-1)
    if exponent == 0:
        c.require(fraction == 0, 'Smoke reference unexpectedly subnormal')
        quad = (word >> 63) << 127
    else:
        c.require(exponent != 2047, 'Smoke reference nonfinite')
        quad = ((word >> 63) << 127) | ((exponent + 16383-1023) << 112) | (fraction << 60)
    return [f'{word:016X}', f'{quad & ((1 << 64)-1):016X}', f'{quad >> 64:016X}']


def prepare():
    c.require(Path.cwd().resolve() == ROOT, 'Run from MadGraph repository root')
    c.require(not WORK.exists() and not (HERE/'prepared.json').exists(), 'Attempt 2 already staged; no overwrite')
    index = read_checked(PRIOR/'evidence-index.json')
    c.require(len(index['files']) == 34, 'Wrong frozen source package size')
    for name, sha in index['files'].items():
        c.require(c.digest(PRIOR/name) == sha, 'Frozen source package changed: '+name)
    previous = read_checked(PRIOR/'prepared.json', index['files']['prepared.json'])
    built = read_checked(PRIOR/'build.json')
    failed = read_checked(PRIOR/'run.json')
    c.require(built['state'] == 'complete' and failed['state'] != 'complete', 'Expected successful build and failed first run')
    c.require(failed['build_json_sha256'] == c.digest(PRIOR/'build.json'), 'Prior run/build mismatch')
    c.require(c.digest(c.COMPILER) == c.COMPILER_SHA256, 'Pinned compiler changed')
    old_source = (PRIOR/'check_observed.f').read_text()
    before, after = "CALL SETPARA('param_card.dat')", "CALL SETPARA('MadLoop5_resources/param_card.dat')"
    c.require(old_source.count(before) == 1, 'Ambiguous original driver call')
    c.require((HERE/'check_observed.f').read_text() == old_source.replace(before, after), 'Driver change is not exactly the path literal')
    prior_evidence = {c.relative(PRIOR/name): c.digest(PRIOR/name)
                      for name in (*index['files'], 'evidence-index.json', 'build.json', 'build.log', 'run.json')}
    parent_index_path = PRIOR.parent/'evidence-index.json'
    parent_index = read_checked(parent_index_path, index['parent_index_sha256'])
    prior_evidence[c.relative(parent_index_path)] = c.digest(parent_index_path)
    for name, sha in parent_index['files'].items():
        path = PRIOR.parent/name
        c.require(c.digest(path) == sha, 'Frozen parent source changed: '+name)
        prior_evidence[c.relative(path)] = sha
    model_sources = ROOT/'.tirust-build/ew-qp-mass-prototype/original/Source/MODEL'
    reader_reference = PRIOR.parent/'sources/reference-03-lha_read.f'
    c.require(c.digest(model_sources/'lha_read.f') == c.digest(reader_reference), 'Generated LHA reader differs from frozen source')
    for name in ('lha_read.f', 'rw_para.f', 'param_read.inc', 'couplings.f'):
        path = model_sources/name
        prior_evidence[c.relative(path)] = c.digest(path)
    for case in failed['cases']:
        directory = c.root_path(case['directory'])
        c.require(directory.is_relative_to(OLDWORK/'runs'), 'Unexpected prior run directory')
        for name, sha in case['files_after'].items():
            path = directory/name
            c.require(c.digest(path) == sha, 'Prior failed-run artifact changed')
            prior_evidence[c.relative(path)] = sha
    expected = dict(previous['artifacts'])
    expected.update(built['outputs'])
    for name, sha in expected.items():
        c.require(c.digest(c.root_path(name)) == sha, 'Prior built input changed: '+name)
    WORK.mkdir()
    origins = {}
    def copy(source, destination, sha=None):
        source = Path(source);destination = c.writable(destination)
        c.require(not destination.exists(), 'Snapshot target already exists')
        if sha is not None:c.require(c.digest(source) == sha, 'Copy source changed')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        c.require(c.digest(source) == c.digest(destination), 'Snapshot copy mismatch')
        origins[c.relative(destination)] = {'source': str(source), 'sha256': c.digest(destination)}
    preserved = {'check_observed.f': 'check_observed.attempt1.f',
                 'check_sa.o': 'check_sa.attempt1.o', 'check_observed': 'check_observed.attempt1'}
    for name, sha in sorted(expected.items()):
        source = c.root_path(name)
        c.require(source.is_relative_to(OLDWORK) and not source.is_relative_to(WORK), 'Unexpected prior artifact')
        destination = WORK/source.relative_to(OLDWORK)
        if source.name in preserved and source.parent.name == 'P0_epem_ttxa':
            destination = destination.with_name(preserved[source.name])
        copy(source, destination, sha)
    def relocated(value):
        return value.replace(str(OLDWORK)+'/', str(WORK)+'/')
    jobs, outputs, arms = [], [], {}
    for arm in ('original', 'candidate'):
        directory = WORK/arm/PROCESS
        copy(HERE/'check_observed.f', directory/'check_observed.f')
        old = previous['arms'][arm]
        commands = [x for x in old['compile_commands'] if Path(x[x.index('-c')+1]).name == 'check_observed.f']
        c.require(len(commands) == 1, 'Ambiguous historical driver compile command')
        compile_command = [relocated(x) for x in commands[0]]
        link_command = [relocated(x) for x in old['link_command']]
        for kind, argv in [('driver-compile', compile_command), ('driver-link', link_command)]:
            jobs.append({'name': arm+'-'+kind, 'cwd': c.relative(directory), 'argv': argv})
            outputs.append(c.relative(Path(argv[argv.index('-o')+1])))
        arms[arm] = {'process_dir': c.relative(directory), 'executable': c.relative(directory/'check_observed')}
    # Same model archive and compiler environment, with no reducer/amplitude input.
    smoke_dir = WORK/'smoke-build';smoke_dir.mkdir()
    copy(HERE/'check_setpara.f', smoke_dir/'check_setpara.f')
    includes = [x for x in jobs[0]['argv'] if x.startswith('-I')]
    compile_smoke = [str(c.COMPILER), '-w', '-fPIC', '-ffixed-line-length-132',
                     '-J'+str(smoke_dir), *includes, '-c', str(smoke_dir/'check_setpara.f'),
                     '-o', str(smoke_dir/'check_setpara.o')]
    link_smoke = [str(c.COMPILER), '-w', '-fPIC', '-ffixed-line-length-132',
                  '-o', str(smoke_dir/'check_setpara'), str(smoke_dir/'check_setpara.o'), str(WORK/'libraries/libmodel.a')]
    for name, argv in [('smoke-compile', compile_smoke), ('smoke-link', link_smoke)]:
        jobs.append({'name': name, 'cwd': c.relative(smoke_dir), 'argv': argv})
        outputs.append(c.relative(Path(argv[argv.index('-o')+1])))
    cards = {}
    for arm in arms:
        directory = WORK/arm/PROCESS
        cards[arm] = {name: c.digest(directory/name) for name in
                      ('param_card.dat', 'MadLoop5_resources/param_card.dat', 'MadLoop5_resources/ident_card.dat')}
        c.require(cards[arm]['param_card.dat'] == cards[arm]['MadLoop5_resources/param_card.dat'], 'Root/resource cards differ')
    c.require(cards['original'] == cards['candidate'], 'Model input pair differs across arms')
    source_files = {c.relative(path): c.digest(path) for path in sorted(HERE.iterdir())
                    if path.is_file() and path.name not in ('prepared.json', 'evidence-index.json')}
    prior_runtime = failed['cases'][0]['loaded_libraries']
    c.require(all(case['loaded_libraries'] == prior_runtime for case in failed['cases']), 'Prior runtime varied')
    record = {'schema': 'ew-qp-mass-observer-attempt2-prepare-v1', 'state': 'source_only_uncompiled',
              'work': c.relative(WORK), 'arms': arms, 'build_jobs': jobs, 'build_outputs': outputs,
              'artifacts': {c.relative(p): c.digest(p) for p in sorted(WORK.rglob('*')) if p.is_file()},
              'snapshot_origins': origins, 'prior_evidence': prior_evidence, 'source_files': source_files,
              'source_package_index_sha256': c.digest(PRIOR/'evidence-index.json'),
              'prior_prepared_sha256': c.digest(PRIOR/'prepared.json'), 'prior_build_sha256': c.digest(PRIOR/'build.json'),
              'prior_run_sha256': c.digest(PRIOR/'run.json'),
              'prior_build_environment': built['environment'],
              'prior_system_runtime': {name: info for name, info in prior_runtime.items() if name not in previous['expected_runtime']},
              'expected_runtime': {name: {'path': c.relative(WORK/c.root_path(info['path']).relative_to(OLDWORK)), 'sha256': info['sha256']}
                                   for name, info in previous['expected_runtime'].items()},
              'parameter_path_proof': {'path': 'MadLoop5_resources/param_card.dat', 'cards': cards,
                  'reader': 'LHA_open_file extracts filename(1:len_trim-15); LHA_blockread appends /ident_card.dat',
                  'reader_reference': c.relative(reader_reference),
                  'model_reader_sources': {name: c.digest(model_sources/name) for name in ('lha_read.f', 'rw_para.f', 'param_read.inc', 'couplings.f')},
                  'driver_change': 'One SETPARA path literal, identical in both arms; numerical objects unchanged'},
              'smoke': {'executable': c.relative(smoke_dir/'check_setpara'),
                  'model_archive': c.relative(WORK/'libraries/libmodel.a'),
                  'model_archive_sha256': c.digest(WORK/'libraries/libmodel.a'),
                  'output_contract': {'header': 'EW_CARD_SMOKE_V1',
                    'ieee_flags_order': ['OVERFLOW', 'DIVIDE_BY_ZERO', 'INVALID', 'UNDERFLOW', 'INEXACT'],
                    'mp_scale_scope': 'Real component storage words plus numeric-zero imaginary check in the smoke source',
                    'ieee_flags_scope': 'Cumulative immediately after SETPARA; no subtraction or clearing',
                    'model_words':
                    {name: promoted_words(value) for name, value in zip(
                        ('MT', 'MZ', 'MW', 'MH', 'WT', 'WZ', 'WW', 'WH', 'MU'),
                        (173.3, 91.188, 80.419, 125.0, 0.0, 0.0, 0.0, 0.0, 1000.0))}}},
              'compiler_invocations': 0, 'native_reducer_calls': 0, 'runtime_mutable_inputs': [],
              'scope': 'Append-only path correction and actual model-card smoke; no numerical algorithm change or acceptance claim'}
    c.reserve(HERE/'prepared.json', record)
    print(json.dumps({'state': record['state'], 'build_jobs': len(jobs), 'artifacts': len(record['artifacts'])}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if not args.prepare:
        print(json.dumps({'state': 'source_only_no_effects', 'work': str(WORK),
                          'planned_compiles': 3, 'planned_links': 3, 'native_reducer_calls': 0}, indent=2))
        return
    prepare()


if __name__ == '__main__':
    main()
