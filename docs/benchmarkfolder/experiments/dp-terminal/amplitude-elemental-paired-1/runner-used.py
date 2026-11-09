#!/usr/bin/env python3
"""Compare isolated DP terminal wavefunction variants using frozen full-amplitude workers."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
FORTRAN = HERE.parent/'n1-profile/fortran'
WORK = ROOT/'.tirust-build/dp-terminal/whole-amplitude'
sys.path[:0] = [str(FORTRAN/'protocol'), str(FORTRAN), str(ROOT)]
sys.path.insert(0, str(HERE/'consumer-audit'))
from audit_consumers import audit as audit_consumers
import tirust_benchmark as campaign
campaign.ROOT = ROOT
campaign.REPORT = ROOT/'docs/benchmarkfolder'
campaign.WORK = ROOT/'.benchmark-work/tirust'
campaign.BASELINE = campaign.WORK/'baseline'
from tirust_benchmark_worker import Worker, parse_outputs, paired_statistics
from prepare import clone
from run import parse_profile
from madgraph.various.banner import MadLoopParam


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def diagnostic_statistics(reference, candidate):
    if len(reference) < 2:
        return {'pairs': len(reference), 'performance_eligible': False,
                'minimum_pair_count_satisfied': False,
                'note': 'Numerical validation only; no confidence interval or performance inference.'}
    result = paired_statistics(reference, candidate, threshold=1.05, minimum_pairs=7)
    result['minimum_pair_count_satisfied'] = result.pop('acceptance_eligible')
    result['performance_eligible'] = False
    result['note'] = 'Diagnostic comparison, not an acceptance campaign.'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--build-metadata', type=Path, required=True)
    parser.add_argument('--fortran-build-metadata', type=Path,
                        help='Use a separately qualified DP terminal-only build as the split variant')
    parser.add_argument('--original-use-candidate-library', action='store_true',
                        help='Use the same DSO for original/split to isolate a Fortran-only difference')
    parser.add_argument('--cpu', type=int, default=9)
    parser.add_argument('--loops', type=int, default=200)
    parser.add_argument('--pairs', type=int, default=7)
    parser.add_argument('--profile-repetitions', type=int, default=0)
    parser.add_argument('--timeout', type=float, default=180.,
                        help='Bound each worker protocol operation')
    args = parser.parse_args()
    if not args.name.replace('-','').replace('_','').isalnum():
        raise ValueError('Invalid output name')
    if args.cpu not in os.sched_getaffinity(0) or min(args.loops,args.pairs)<1 or args.profile_repetitions<0:
        raise ValueError('Invalid CPU or run dimensions')
    library = args.library.resolve(strict=True)
    metadata = args.build_metadata.resolve(strict=True)
    if json.loads(metadata.read_text())['library_sha256'] != digest(library):
        raise RuntimeError('Build metadata does not identify the supplied library')
    fortran_build = None
    consumer_proof = None
    if not args.fortran_build_metadata or args.profile_repetitions:
        raise ValueError('This isolated trial requires a qualified terminal build and no unrelated profile variant')
    if args.fortran_build_metadata:
        if not args.original_use_candidate_library:
            raise ValueError('DP terminal-only trial requires --original-use-candidate-library')
        if args.profile_repetitions:
            raise ValueError('DP terminal-only build comparison requires --profile-repetitions 0')
        fortran_build = json.loads(args.fortran_build_metadata.read_text())
        if not fortran_build.get('built') or fortran_build['changed_objects'] != ['coef_construction_1.o', 'polynomial.o']:
            raise ValueError('Expected a completed polynomial.o + coef_construction_1.o-only build')
        if fortran_build.get('trace_admission') and (args.loops != 1 or args.pairs != 1):
            raise ValueError('Admission tracing is only allowed in one-pass diagnostics')
        helpers = {'elemental': HERE/'terminal.f90', 'packed': HERE/'packed-scan/terminal.f90'}
        helper = helpers[fortran_build['variant']]
        consumer_proof = audit_consumers(
            ROOT/'.benchmark-work/tirust/outputs/candidate/qcd_n1_uux_z',
            ROOT/'docs/benchmarkfolder/runs/qcd_n1_uux_z/tirust-first-attempt-3/candidate-inputs.json',
            transformed_root=Path(fortran_build['output']), helper_path=helper,
            helper_sha256=digest(helper),
            trace_admission=fortran_build.get('trace_admission', False))
    report = HERE/args.name
    report.mkdir(parents=True, exist_ok=False)
    work = WORK/args.name
    work.mkdir(parents=True, exist_ok=False)
    frozen = ROOT/'docs/benchmarkfolder/runs/qcd_n1_uux_z/tirust-first-attempt-3'
    points = ROOT/'docs/benchmarkfolder/points/qcd_n1_uux_z.dat'
    result = {'performance_eligible': False, 'purpose': __doc__, 'cpu': args.cpu,
              'machine': campaign.machine_identity(args.cpu),
              'loops': args.loops, 'pairs': args.pairs, 'profile_repetitions': args.profile_repetitions,
              'runner_sha256': digest(__file__), 'points_sha256': digest(points),
              'worker_sha256': digest(FORTRAN/'protocol/tirust_benchmark_worker.py'),
              'candidate_library': {'path': str(library), 'sha256': digest(library)},
              'build_metadata': {'path': str(metadata), 'sha256': digest(metadata)},
              'original_uses_candidate_library': args.original_use_candidate_library,
              'inputs': {}, 'runs': []}
    if fortran_build:
        result['fortran_build_metadata'] = {'path':str(args.fortran_build_metadata.resolve()),
                                            'sha256':digest(args.fortran_build_metadata)}
        shutil.copy2(args.fortran_build_metadata,report/'candidate-fortran-build-metadata.json')
        dump(report/'consumer-audit.json', consumer_proof)
        result['consumer_audit_sha256'] = digest(report/'consumer-audit.json')
        result['consumer_auditor_sha256'] = digest(HERE/'consumer-audit/audit_consumers.py')
    shutil.copy2(metadata, report/'candidate-build-metadata.json')
    shutil.copy2(__file__, report/'runner-used.py')
    variant_paths = {}
    chains = {}
    frozen_libraries = {}
    with (ROOT/'.tirust-build/numerical-tests.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for variant in ('baseline', 'original', 'terminal', 'profile-split'):
            if variant == 'profile-split' and not args.profile_repetitions:
                continue
            source = 'baseline' if variant == 'baseline' else 'candidate'
            inputs = json.loads((frozen/(source+'-inputs.json')).read_text())
            original = ROOT/'.benchmark-work/tirust/outputs'/source/'qcd_n1_uux_z'
            original_process = original/'SubProcesses/P0_uux_z'
            if digest(original_process/'check') != inputs['compilation']['executable_sha256']:
                raise RuntimeError('Original executable changed since qualified gate')
            if digest(original_process/'check_sa.f') != inputs['compilation']['driver_sha256']:
                raise RuntimeError('Original worker source changed since qualified gate')
            if digest(points) != inputs['points_sha256']:
                raise RuntimeError('Frozen points changed')
            if digest(original/'Cards/param_card.dat') != inputs['param_card_sha256']:
                raise RuntimeError('Frozen parameter card changed')
            if variant == 'profile-split':
                original = ROOT/'.tirust-build/n1-profile/fortran/instrumented'
                manifest = json.loads((FORTRAN/'built-artifacts.json').read_text())
                artifact = next(x for x in manifest['artifacts'] if '/instrumented/' in x['path'])
                if digest(original/'SubProcesses/P0_uux_z/check') != artifact['sha256']:
                    raise RuntimeError('Instrumented executable changed')
            elif variant == 'terminal' and fortran_build:
                if fortran_build['original_executable_sha256'] != inputs['compilation']['executable_sha256']:
                    raise RuntimeError('DP terminal-only build derives from a different original executable')
                original = Path(fortran_build['output'])
                local = original/'SubProcesses/P0_uux_z'
                for source_name, expected_sha in fortran_build['source_sha256'].items():
                    if digest(local/source_name) != expected_sha:
                        raise RuntimeError('DP terminal source changed: '+source_name)
                if digest(local/'check') != fortran_build['executable_sha256']:
                    raise RuntimeError('DP terminal-only executable changed')
                if digest(local/'check_sa.f') != fortran_build['driver_sha256']:
                    raise RuntimeError('DP terminal-only worker changed')
                for name, expected in fortran_build['objects_after'].items():
                    if digest(local/name) != expected:
                        raise RuntimeError('DP terminal-only object changed: '+name)
            output = work/variant
            clone(original, output)
            directory = output/'SubProcesses/P0_uux_z'
            variant_paths[variant] = directory
            card_path = output/'Cards/MadLoopParams.dat'
            card = MadLoopParam(str(card_path))
            for key,value in inputs['settings'].items():
                card[key] = value
            card.write(str(card_path), template=str(card_path))
            for local in (output/'SubProcesses').rglob('MadLoopParams.dat'):
                if local.is_symlink():
                    local.unlink()
                shutil.copy2(card_path,local)
            for pattern in ('*HelFilter.dat','*LoopFilter.dat'):
                for local in (output/'SubProcesses').rglob(pattern):
                    local.unlink()
            shutil.copy2(card_path, report/(variant+'-MadLoopParams.dat'))
            chains[variant] = campaign.parse_chain(inputs['settings']['mlreductionlib'])
            frozen_libraries[variant] = inputs['runtime']['libraries']
            result['inputs'][variant] = {
                'frozen_input_sha256': digest(frozen/(source+'-inputs.json')),
                'settings': inputs['settings'], 'executable_sha256': digest(directory/'check'),
                'driver_sha256': digest(directory/'check_sa.f'),
                'param_card_sha256': digest(output/'Cards/param_card.dat')}
        reference = None
        times = {variant: [] for variant in ('baseline','original','terminal')}
        rows_by_pair = {}
        variants = ['baseline','original','terminal']
        sequence = []
        for repetition in range(args.pairs):
            # Balanced rotations and reversal distribute early/late positions.
            order = variants[repetition%3:]+variants[:repetition%3]
            if repetition%2:
                order.reverse()
            sequence.extend((repetition, variant) for variant in order)
        sequence.extend((i, 'profile-split') for i in range(args.profile_repetitions))
        for repetition, variant in sequence:
            if digest(library) != result['candidate_library']['sha256']:
                raise RuntimeError('Candidate library changed during the experiment')
            directory = variant_paths[variant]
            env = campaign.environment()
            env.pop('ML_TIRUST_FORTRAN_PROFILE', None)
            env.pop('ML_TIRUST_BRIDGE_PROFILE', None)
            expected_library = ROOT/'HEPTools/tirust/lib/libmadloop_tirust.so'
            uses_candidate_library = variant in ('terminal','profile-split') or (
                variant == 'original' and args.original_use_candidate_library)
            if uses_candidate_library:
                expected_library = library
                env['LD_LIBRARY_PATH'] = str(library.parent)+os.pathsep+env.get('LD_LIBRARY_PATH','')
            if variant == 'profile-split':
                env['ML_TIRUST_FORTRAN_PROFILE'] = '1'
                env['ML_TIRUST_BRIDGE_PROFILE'] = '2'
            target = report/str(repetition)/variant
            target.mkdir(parents=True)
            linkage = subprocess.check_output(['ldd',str(directory/'check')], env=env, text=True)
            (target/'ldd.txt').write_text(linkage)
            runtime = {name:digest(name) for name in re.findall(r'(?:=>\s*)?(/[^\s]+)',linkage)
                       if Path(name).is_file()}
            expected_runtime = dict(frozen_libraries[variant])
            if uses_candidate_library:
                previous = [p for p in expected_runtime if Path(p).name == 'libmadloop_tirust.so']
                if len(previous) != 1:
                    raise RuntimeError('Expected exactly one frozen TIRust DSO')
                del expected_runtime[previous[0]]
                expected_runtime[str(library)] = result['candidate_library']['sha256']
            dump(target/'runtime-identity.json', {'actual':runtime,'expected':expected_runtime})
            if runtime != expected_runtime:
                raise RuntimeError('Runtime dependencies changed outside the selected TIRust DSO')
            resolved = [line.split('=>',1)[1].strip().split()[0] for line in linkage.splitlines()
                        if line.strip().startswith('libmadloop_tirust.so =>')]
            if variant != 'baseline' and (len(resolved)!=1 or Path(resolved[0]).resolve()!=expected_library.resolve()):
                raise RuntimeError('Unexpected runtime library')
            before = campaign.host_load_snapshot()
            # A single live native worker avoids overlapping Symbolica sessions.
            with Worker(['taskset','-c',str(args.cpu),'./check',str(points)],
                        directory,target/'worker.log',env=env,timeout=args.timeout) as worker:
                worker.run(args.loops)
                calibration = target/'calibration'
                calibration.mkdir()
                for filename in ('timing.dat','values.dat','timed_status.dat','dimensions.dat'):
                    shutil.copy2(directory/filename, calibration/filename)
                warm_observation = parse_outputs(calibration,chains[variant],100,args.loops)
                worker.run(args.loops)
                for filename in ('timing.dat','values.dat','timed_status.dat','dimensions.dat'):
                    shutil.copy2(directory/filename,target/filename)
                observation = parse_outputs(target,chains[variant],100,args.loops)
                if reference is None:
                    reference = observation['rows']
                parity = campaign.compare_rows(reference,observation['rows'],1e-8)
                warm_parity = campaign.compare_rows(reference,warm_observation['rows'],1e-8)
                expected_backend = '8:DP' if variant == 'baseline' else '9:DP'
                expected_usage = {expected_backend:args.loops*100}
                record = {'repetition': repetition, 'variant': variant, 'observation': observation,
                          'calibration': warm_observation,
                          'maximum_relative_error': parity['maximum_relative_error'], 'parity_passed': parity['passed'],
                          'calibration_parity_passed': warm_parity['passed'],
                          'expected_backend_usage_passed': all(o['backend_precision_counts'] == expected_usage
                              for o in (observation,warm_observation)),
                          'host_load_before': before, 'host_load_after': campaign.host_load_snapshot(),
                          'resolved_library': resolved,
                          'resolved_library_sha256': [digest(p) for p in resolved]}
                if variant == 'profile-split':
                    for filename in ('fortran-clock.txt','fortran-profile.tsv','bridge-profile.json'):
                        shutil.copy2(directory/filename,target/filename)
                    record['profile'] = parse_profile(target,args.loops*100)
                    actual = {(r['stage'],r['ctmode'],r['propagators'],r['rank'],r['precision']):r['calls']
                              for r in record['profile']['stages']}
                    expected = {(stage,mode,3,2,1):args.loops*100 for stage in range(1,8) for mode in (1,2)}
                    record['expected_stage_counts_passed'] = actual == expected
                    record['bridge_failed_calls'] = sum(r['failed_calls'] for r in record['profile']['bridge']['rows'])
                else:
                    times[variant].append(observation['seconds_per_point'])
                    rows_by_pair.setdefault(repetition,{})[variant] = observation['rows']
                result['runs'].append(record)
                dump(target/'observation.json', record)
                dump(report/'results.json', result)
                if not parity['passed'] or not warm_parity['passed'] or not record['expected_backend_usage_passed']:
                    raise RuntimeError('Unexpected numerical output or precision; evidence retained')
                if variant == 'profile-split' and (not record['expected_stage_counts_passed'] or record['bridge_failed_calls']):
                    raise RuntimeError('Invalid stage counts or failed bridge calls')
            # Read only after process shutdown and stdout-drainer completion.
            admission = re.findall(r'DP_TERMINAL_ADMISSION\s+([TF])\s+([TF])',
                                   (target/'worker.log').read_text())
            if admission:
                record['admission_trace'] = {
                    'scope': 'Includes cold initialization, warmup and both one-pass blocks; no timing claim',
                    'calls': len(admission),
                    'preexisting_inexact_calls': sum(entry == 'T' for entry, fast in admission),
                    'fast_calls': sum(fast == 'T' for entry, fast in admission),
                    'fallback_calls': sum(fast == 'F' for entry, fast in admission)}
                dump(target/'observation.json', record)
                dump(report/'results.json', result)
        exact_parity = [campaign.compare_rows(p['original'],p['terminal'],1e-8) for p in rows_by_pair.values()]
        def packed_coefficients(rows):
            return b''.join(struct.pack('!d',value) for row in rows for value in row['coefficients'])
        result['all_original_terminal_bit_identical'] = all(
            packed_coefficients(p['original']) == packed_coefficients(p['terminal']) for p in rows_by_pair.values())
        result['maximum_original_terminal_relative_error'] = max(p['maximum_relative_error'] for p in exact_parity)
        result['timings'] = times
        result['median_microseconds_per_amplitude'] = {k:statistics.median(v)*1e6 for k,v in times.items()}
        result['paired_statistics'] = {label: diagnostic_statistics(times[left],times[right])
            for label,left,right in [('original_over_baseline','baseline','original'),
                                     ('terminal_over_baseline','baseline','terminal'),
                                     ('terminal_over_original','original','terminal')]}
        dump(report/'results.json',result)
    print(json.dumps({k:result[k] for k in ('median_microseconds_per_amplitude','paired_statistics',
                                          'all_original_terminal_bit_identical')},indent=2))


if __name__ == '__main__':
    main()
