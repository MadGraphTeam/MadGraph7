#!/usr/bin/env python3
"""Actual CLI exports and exact legacy-Ninja checks of the generated port."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'madgraph/__init__.py').is_file())
HERE = Path(__file__).resolve().parent
WORK = ROOT/'.benchmark-work/experiments/terminal-production'
REPORT = HERE/'production-port'
sys.path[:0] = [str(ROOT), str(ROOT/'tools'), str(HERE)]
import ninjars_benchmark as bench
from madgraph.various.banner import MadLoopParam
from terminal import current_classifier

SOURCES = [
    'madgraph/various/q_polynomial.py', 'madgraph/iolibs/helas_call_writers.py',
    'madgraph/loop/loop_exporters.py',
    'madgraph/iolibs/template_files/loop_optimized/mp_helas_calls_split.inc',
    'madgraph/iolibs/template_files/loop_optimized/polynomial.inc',
    'madgraph/iolibs/export_v4.py', 'madgraph/iolibs/ufo_expression_parsers.py']
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run(command, cwd, log, env):
    start = time.monotonic()
    with log.open('w') as stream:
        subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    return time.monotonic()-start

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cases', nargs='+')
    parser.add_argument('--cpu', type=int, default=2)
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    config = json.loads((ROOT/'docs/benchmarkfolder/cases.json').read_text())
    controls = json.loads((ROOT/'docs/benchmarkfolder/controls.json').read_text())
    cases = {c['id']: c for c in config['cases']+controls['cases']}
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        MAKEFLAGS='SHELL='+shutil.which('bash'),
        LD_LIBRARY_PATH=os.pathsep.join(str(bench.BASELINE/'HEPTools'/p)
                                       for p in ('ninja/lib', 'collier')))
    for name in args.cases:
        case = cases[name]
        report, work = REPORT/name, WORK/name
        report.mkdir(parents=True, exist_ok=False)
        work.mkdir(parents=True, exist_ok=False)
        output = work/'production'
        original = ROOT/'.benchmark-work/policy-v2/generation/outputs/candidate'/name
        hashes = {p: sha(ROOT/p) for p in SOURCES}
        for relative in SOURCES:
            target = report/'generator'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/relative, target)
        result = {'case': name, 'complete': False, 'acceptance_eligible': False,
            'scope': 'Production-port correctness only; original Ninja6, no timing acceptance.',
            'generator_sources': hashes, 'git_revision': subprocess.check_output(
                ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(), 'variants': {},
            'script_sha256': sha(__file__)}
        shutil.copy2(__file__,report/'validate_port.py')
        try:
            model = case['model'] if case['model']=='loop_sm' else str(bench.BASELINE/'models'/case['model'])
            command = report/'generate.mg5'
            command.write_text('\n'.join([
                'set automatic_html_opening False', 'set auto_update 0', 'set nb_core 2',
                'set loop_optimized_output True', 'set output_dependencies external',
                'set complex_mass_scheme False',
                'set ninja '+str(bench.BASELINE/'HEPTools/ninja/lib'),
                'set collier '+str(bench.BASELINE/'HEPTools/collier'),
                'set ninjars None', 'import model '+model, 'generate '+case['process'],
                'output standalone '+str(output), 'quit'])+'\n')
            result['generation_seconds'] = run(
                [sys.executable,str(ROOT/'bin/madgraph'),'--plain',str(command)], ROOT,
                report/'generation.log',env)
            if (not (output/'Cards/param_card.dat').is_file() or
                    'Error detected in sub-command' in (report/'generation.log').read_text()):
                raise RuntimeError('CLI export failed')
            if hashes != {p: sha(ROOT/p) for p in SOURCES}:
                raise RuntimeError('Relevant generator sources changed during CLI export')
            shutil.copy2(original/'Cards/param_card.dat', output/'Cards/param_card.dat')
            directory = next(output.glob('SubProcesses/P*_*'))
            calls = sum(len(re.findall(r'CALL MP_\w+TERMINAL_WL_', p.read_text()))
                        for p in directory.glob('mp_coef_construction_*.f'))
            result['terminal_calls'] = calls
            if not calls or (name in ('qcd-3','control-mixed') and
                             calls != {'qcd-3':224,'control-mixed':151}[name]):
                raise RuntimeError('Rendered terminal proof disabled or changed expected calls')
            result['generated_sources'] = bench.generated_source_manifest(output)
            if args.evaluate:
                expected = None
                for variant in ('original','production'):
                    target = output if variant=='production' else work/'original'
                    if variant=='original':
                        subprocess.run(['cp','-a','--reflink=auto',str(original),str(target)],check=True)
                        for link in target.rglob('*'):
                            if link.is_symlink() and link.resolve().is_relative_to(original):
                                dest = target/link.resolve().relative_to(original)
                                link.unlink()
                                link.symlink_to(os.path.relpath(dest,link.parent))
                    proc = next(target.glob('SubProcesses/P*_*'))
                    poly = proc/'polynomial.f'
                    poly.write_text(current_classifier(poly.read_text()))
                    prefix = re.search(r'CALL\s+(\w*)GET_ANSWER_DIMENSION',
                                       (proc/'check_sa.f').read_text(),re.I)[1]
                    born = proc/'nsqso_born.inc'
                    nb = re.search(r'NSQSO_BORN\s*=\s*(\d+)',born.read_text(),re.I)[1] if born.exists() else '0'
                    driver = (ROOT/'tools/ninjars_benchmark_driver.f').read_text().replace(
                        '@PREFIX@',prefix).replace('@NEXTERNAL@',str(case['nfinal']+2)).replace('@NBORN@',nb)
                    forced = name.startswith('control-')
                    if forced:
                        driver = driver.replace('FORCE_STABILITY_CHECK(.TRUE.)','FORCE_STABILITY_CHECK(.FALSE.)')
                        driver = driver.replace('DO K=1,1000+NPOINTS','DO K=1,MAX(20,2*NPOINTS)')
                    (proc/'check_sa.f').write_text(driver)
                    card = MadLoopParam(str(target/'Cards/MadLoopParams.dat'))
                    for key,value in {'MLReductionLib':'6','CTModeRun':4 if forced else -1,
                            'MLStabThres':1e-9,'COLLIERRequiredAccuracy':1e-10}.items():
                        card[key]=value
                    for filename in [target/'Cards/MadLoopParams.dat',*(target/'SubProcesses').rglob('MadLoopParams.dat')]:
                        card.write(str(filename),template=str(ROOT/'Template/loop_material/StandAlone/Cards/MadLoopParams.dat'))
                    elapsed = run(['make','-j2','check'],proc,report/(variant+'-build.log'),env)
                    for pattern in ('*HelFilter.dat','*LoopFilter.dat'):
                        for filename in (target/'SubProcesses').rglob(pattern):
                            filename.unlink()
                    points = ROOT/'docs/benchmarkfolder/points'/(name+'.dat')
                    run(['./check',str(points),'1'],proc,report/(variant+'-run.log'),env)
                    for filename in ('values.dat','timed_status.dat','dimensions.dat'):
                        shutil.copy2(proc/filename,report/(variant+'-'+filename))
                    rows = (proc/'values.dat').read_bytes(),(proc/'timed_status.dat').read_bytes()
                    records = [line.split() for line in rows[0].decode().splitlines()]
                    dimensions = list(map(int,(proc/'dimensions.dat').read_text().split()))
                    npoints = int(points.read_text().split()[0])
                    if [(int(row[0]),int(row[1])) for row in records] != [
                            (point,order) for point in range(1,npoints+1)
                            for order in range(dimensions[1]+1)]:
                        raise RuntimeError('Incomplete physical output rows')
                    for row in records:
                        if not all(math.isfinite(float(value)) for value in row[3:]):
                            raise RuntimeError('Nonfinite output')
                        if forced:
                            if int(row[2])//100 != 1 or int(row[2])%100//10 != 2:
                                raise RuntimeError('Forced QP must remain explicitly unchecked')
                        else:
                            bench.validate_status(int(row[2]),6)
                    if expected is None:
                        expected = rows
                    if rows != expected:
                        raise RuntimeError('Production and original coefficient/accuracy/status rows differ')
                    result['variants'][variant] = {'compiled_sources':bench.generated_source_manifest(target),
                        'executable_sha256':sha(proc/'check'),'compilation_seconds':elapsed,
                        'points_sha256':sha(points),'param_card_sha256':sha(target/'Cards/param_card.dat'),
                        'madloop_card_sha256':sha(target/'Cards/MadLoopParams.dat'),
                        'forced_unchecked_qp':forced}
                result['all_rows_and_statuses_exact'] = True
            result['complete'] = True
        except Exception as error:
            result['error'] = repr(error)
            raise
        finally:
            bench.dump(report/'result.json',result)
        print(name,'PASS',calls,flush=True)

if __name__=='__main__':
    main()
