#!/usr/bin/env python3
"""Prepare two ordinary generic loop-induced exports; never evaluate an amplitude."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'madgraph/__init__.py').is_file())
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import ninjars_benchmark as bench
import ninjars_benchmark_pair_v3 as pair
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_workers as workers
from madgraph.various.banner import MadLoopParam

HERE = Path(__file__).resolve().parent
REPORT = HERE / 'run-v1'
WORK = ROOT / '.benchmark-work/policy-v3/final-pins-20261021-v1/higher-higgs-generation/run-v1'
CASES = [
    {'id': 'profile-gg-hhh', 'model': 'loop_sm', 'process': 'g g > h h h [sqrvirt=QCD]', 'nfinal': 3, 'seed': 940086},
    {'id': 'profile-gg-hhhh', 'model': 'loop_sm', 'process': 'g g > h h h h [sqrvirt=QCD]', 'nfinal': 4, 'seed': 940087},
]


def bound(path):
    content = path.read_bytes()
    return {'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}


def source_inventory():
    paths = [*bench.GENERATOR_PATHS, 'vendor']
    names = subprocess.check_output(['git', 'ls-files', '-z', '--', *paths], cwd=ROOT).decode().split('\0')
    tracked = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() if (ROOT / name).is_file() else None
               for name in names if name}
    return {'generator_tree': bench.generator_tree_state(), 'tracked_generator_vendor_files': tracked}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-application', type=Path, required=True)
    parser.add_argument('--smoke-proof', type=Path, required=True)
    parser.add_argument('--execute-generation', action='store_true')
    args = parser.parse_args()
    if not args.execute_generation or os.sched_getaffinity(0) != {8} or any(os.environ.get(n) != '1' for n in pair.THREADS):
        parser.error('Root-authorized generation requires CPU8 alone and all four thread limits1')
    application = json.loads(args.source_application.read_text())
    for name, image in application['after'].items():
        if bound(ROOT / name)['sha256'] != image['sha256']:
            raise ValueError('Live source differs from audited application: ' + name)
    json.loads(args.smoke_proof.read_text())
    if REPORT.exists() or WORK.exists():
        raise ValueError('Preserve every earlier generation attempt; no append/restart')
    REPORT.mkdir(); WORK.mkdir(parents=True)
    shutil.copy2(args.source_application, REPORT / 'source-application.json')
    shutil.copy2(args.smoke_proof, REPORT / 'accepted-default-smoke.json')
    before = source_inventory()
    bench.dump(REPORT / 'source-inventory-before.json', before)
    common_card = HERE.parent / 'controls-generation/inputs/candidate/control-gg-hh/param_card.dat'
    common_bytes = common_card.read_bytes()
    status = {'schema': 'higher-higgs-actual-cli-preparation-v1', 'started_utc': now(), 'status': 'incomplete',
              'cases': {}, 'source_application': bound(REPORT / 'source-application.json'),
              'accepted_default_smoke': bound(REPORT / 'accepted-default-smoke.json'), 'cpu': 8, 'make_jobs': 1,
              'thread_environment': {name: os.environ[name] for name in pair.THREADS},
              'amplitude_evaluations': 0, 'reduction_calls': 0, 'timing_measurements': 0,
              'scope': 'Actual bin/madgraph standalone exports and clean Fortran compile/link plus external-masses query only.',
              'source_inventory_before': bound(REPORT / 'source-inventory-before.json'), 'physical_card_anchor': bound(common_card),
              'script': bound(Path(__file__).resolve())}
    bench.dump(REPORT / 'status.json', status)
    old = bench.WORK, bench.REPORT
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    def interrupted(signum, frame):
        raise InterruptedError('Generation interrupted by signal ' + str(signum))
    for sig in previous:
        signal.signal(sig, interrupted)
    bench.WORK, bench.REPORT = WORK, REPORT
    try:
        for case in CASES:
            if source_inventory() != before:
                raise ValueError('Actual generator/vendor bytes changed before export')
            status['cases'][case['id']] = {'status': 'started', 'case': case}
            bench.dump(REPORT / 'status.json', status)
            with pair.owned_preparation(REPORT, label=case['id']):
                owned_run = bench.run
                def serial_run(command, cwd, log, **kwargs):
                    if command[:3] == [sys.executable, str(ROOT / 'bin/madgraph'), '--plain'] and len(command) == 4:
                        card = Path(command[3])
                        text = card.read_text()
                        if text.count('set nb_core 4\n') != 1:
                            raise ValueError('Unexpected generic generation command card')
                        card.write_text(text.replace('set nb_core 4\n', 'set nb_core 1\n'))
                    elif command == ['make', '-j4', 'check']:
                        command = ['make', '-j1', 'check']
                    elif command != ['./check', '--masses']:
                        raise ValueError('Only generation, serial compile, and masses query are allowed')
                    return owned_run(command, cwd, log, **kwargs)
                bench.run = serial_run
                try:
                    bench.generate('candidate', case)
                    output = bench.output_dir('candidate', case)
                    if (output / 'Cards/param_card.dat').read_bytes() != common_bytes:
                        raise ValueError('Fresh loop_sm physical card differs from common zero-width anchor')
                    card = output / 'Cards/MadLoopParams.dat'
                    if str(MadLoopParam(str(card))['MLReductionLib']).replace(' ', '') != '7|6|1':
                        raise ValueError('Loop-induced original default must remain7|6|1')
                    destination = REPORT / 'inputs/candidate' / case['id'] / 'MadLoopParams-origin.dat'
                    shutil.copy2(card, destination)
                    masses = bench.build('candidate', case)
                finally:
                    bench.run = owned_run
            manifest = json.loads((output / 'compiled-sources.json').read_text())
            directory = bench.subprocess_dir('candidate', case)
            libraries = policy.library_hashes(output)
            metadata = {}
            for name, path in policy.runtime_metadata_files(output).items():
                target = REPORT / 'runtime-assets' / hashlib.sha256(path.read_bytes()).hexdigest()
                target.parent.mkdir(exist_ok=True)
                if not target.exists():
                    shutil.copy2(path, target)
                metadata[name] = bound(target)
            runtime = policy.reducer_signature({'backend': 8, 'libraries': libraries, 'runtime_metadata': metadata,
                                                'compiled_manifest': manifest})
            bench.dump(REPORT / 'runtime' / (case['id'] + '.json'), runtime)
            for stage in ('generate', 'compile', 'masses'):
                source = WORK / 'logs' / f'{stage}-candidate-{case["id"]}.log'
                target = REPORT / 'logs' / source.name
                target.parent.mkdir(exist_ok=True); shutil.copy2(source, target)
            proof = {'case': case, 'output': str(output.relative_to(ROOT)), 'subprocess': directory.name,
                     'generation': bound(REPORT / 'inputs/candidate' / case['id'] / 'generation.json'),
                     'compilation': bound(REPORT / 'inputs/candidate' / case['id'] / 'compilation.json'),
                     'compiled_manifest': bound(REPORT / 'inputs/candidate' / case['id'] / 'compiled-sources.json'),
                     'physical_card': bound(REPORT / 'inputs/candidate' / case['id'] / 'param_card.dat'),
                     'madloop_origin_card': bound(destination), 'origin_reduction_order': '7|6|1',
                     'runtime_signature': bound(REPORT / 'runtime' / (case['id'] + '.json')),
                     'runtime_metadata': metadata, 'libraries': libraries, 'external_masses': masses,
                     'generated_source_identity': json.loads((output / 'generation.json').read_text())['source_identity'],
                     'compiled_only': True, 'amplitude_evaluations': 0, 'reduction_calls': 0}
            bench.dump(REPORT / 'source-proofs' / (case['id'] + '.json'), proof)
            status['cases'][case['id']] = {'status': 'complete', 'case': case,
                                         'proof': bound(REPORT / 'source-proofs' / (case['id'] + '.json'))}
            bench.dump(REPORT / 'status.json', status)
            print(case['id'], 'complete', flush=True)
        after = source_inventory()
        bench.dump(REPORT / 'source-inventory-after.json', after)
        if after != before or bound(Path(__file__).resolve()) != status['script']:
            raise ValueError('Actual source bytes changed during generation/build')
        status.update(status='complete', source_bytes_unchanged=True,
                      source_inventory_after=bound(REPORT / 'source-inventory-after.json'), finished_utc=now())
    except BaseException as error:
        status.update(status='failed', failure=repr(error), finished_utc=now())
        raise
    finally:
        bench.dump(REPORT / 'status.json', status)
        bench.WORK, bench.REPORT = old
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    main()
