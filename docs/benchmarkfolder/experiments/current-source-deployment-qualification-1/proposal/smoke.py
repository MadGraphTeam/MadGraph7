"""One-use fixed-driver deployment smoke, admitted separately from build."""
from decimal import Decimal, InvalidOperation
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare_source import BIND, ROOT, WORK, protected, require, digest

MODES = {'standalone': [(1, 1), (8, 1), (9, 1), (9, 2), (9, 4), (9, 5)],
         'nlo': [(1, 1), (9, 1), (9, 4)]}


def number(value):
    try:
        result = Decimal(value.replace('D', 'E').replace('d', 'e'))
    except InvalidOperation as exc:
        raise ValueError('Malformed numeric field') from exc
    require(result.is_finite(), 'Nonfinite result')
    return result


def parse(text, mode):
    scalar = {}
    points = []
    split = {}
    for line in text.splitlines():
        row = line.split()
        if not row:
            continue
        if row[0] == 'PS':
            require(len(row) == 5, 'Point dimensions')
            points.append([number(x) for x in row[1:]])
        elif row[0] in {'BORN', 'FIN', '1EPS', '2EPS', 'ASO2PI', 'ACC', 'RETCODE', 'EXP'}:
            require(len(row) == 2 and row[0] not in scalar, 'Duplicate/malformed scalar')
            scalar[row[0]] = number(row[1])
        elif row[0] in {'SO_Born', 'SO_Loop'}:
            key = ':'.join(row[:2])
            require(len(row) == 3 and key not in split, 'Duplicate/malformed split order')
            split[key] = number(row[2])
    require(len(points) == 3 and len(scalar) == 8, 'Incomplete result')
    require(set(split) == {'SO_Born:BORN', 'SO_Loop:ACC', 'SO_Loop:FIN',
                           'SO_Loop:1EPS', 'SO_Loop:2EPS'}, 'Incomplete split orders')
    require(scalar['EXP'] == 2 and scalar['BORN'] > 0 and scalar['ASO2PI'] > 0, 'Invalid normalization')
    require(scalar['RETCODE'] == (120 if mode >= 4 else 110), 'Unexpected fixed-mode status')
    require(scalar['ACC'] == -1 and split['SO_Loop:ACC'] == -1, 'Unexpected fixed-mode accuracy')
    require(points[0][0] > 0 and points[0][0] == points[1][0], 'Initial energies')
    require(points[0][1:3] == points[1][1:3] == [0, 0], 'Transverse initial momentum')
    require(points[0][3] == -points[1][3] == points[0][0], 'Initial mass shell')
    require(points[2] == [points[0][0] + points[1][0], Decimal(0), Decimal(0), Decimal(0)],
            'Final mass shell/conservation')
    return {'point': [[str(x) for x in row] for row in points],
            'scalar': {k: str(v) for k, v in scalar.items()},
            'split': {k: str(v) for k, v in split.items()}}


def compare(reference, current):
    require(reference['point'] == current['point'], 'Different physical input')
    require(reference['scalar']['ASO2PI'] == current['scalar']['ASO2PI'], 'Different coupling normalization')
    residuals = {}
    for section, names in [('scalar', ['BORN', 'FIN', '1EPS', '2EPS']),
                           ('split', ['SO_Born:BORN', 'SO_Loop:FIN', 'SO_Loop:1EPS', 'SO_Loop:2EPS'])]:
        finite = max(abs(number(reference[section][names[1]])), abs(number(current[section][names[1]])))
        for key in names:
            a, b = number(reference[section][key]), number(current[section][key])
            ratio = abs(a-b) / abs(a+b) if a+b else (Decimal(0) if a == b else None)
            pole = key.endswith('EPS')
            passes = {}
            for tolerance in ['0.000003', '0.00000001']:
                tol = Decimal(tolerance)
                passes[tolerance] = ((ratio is not None and ratio < tol) or
                    (pole and finite > 0 and max(abs(a), abs(b)) / finite < tol))
            residuals[section+':'+key] = {'symmetric': str(ratio) if ratio is not None else None,
                                        'regression_passed': passes['0.000003'],
                                        'ordinary_target_passed': passes['0.00000001']}
    return residuals


def child_setup():
    os.sched_setaffinity(0, {20})
    resource.setrlimit(resource.RLIMIT_AS, (16*1024**3, 16*1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def verify_build_inputs(receipt):
    require(receipt.get('schema') == 'tirust-final-deployment-build-v1', 'Unrecognized build receipt')
    require(receipt.get('revision') == '0fcd2cd873d61d134837e1d9e87afb13a47824fd', 'Wrong build source')
    require(receipt.get('exporter_inventory') == BIND['candidate']['exporter_identity']['files_sha256'],
            'Wrong exporter closure')
    require(receipt.get('passed') is True and receipt.get('native_evaluations') == 0,
            'Build not closed before native smoke')
    require(isinstance(receipt.get('files'), dict) and receipt['files'], 'Missing build input inventory')
    require(isinstance(receipt.get('runtime_providers'),dict) and receipt['runtime_providers'],
            'Missing pre-runtime provider inventory')
    require(receipt.get('scope_inventory_complete') is True,
            'Build sources, model, archives, modules, recipes and cards were not fully inventoried')
    for name, checksum in receipt['files'].items():
        path=Path(name)
        require(path.is_file() and digest(path)==checksum, 'Compiled source/input drift: '+name)
    for name,checksum in receipt['runtime_providers'].items():
        require(Path(name).is_file() and digest(Path(name))==checksum, 'Runtime provider drift: '+name)
    for label, suffix in [('standalone','SubProcesses/P0_uux_z'),
                          ('nlo','SubProcesses/P0_uux_z/V0_uux_z')]:
        export=WORK/'relocated'/label
        for name in [str(export/suffix/'check'), str(export/suffix/'check_sa.f'),
                     str(export/'Cards/param_card.dat'), str(export/'lib/libmadloop_tirust.so')]:
            require(name in receipt['files'], 'Missing critical executable/input binding: '+name)
        active_card=export/('Cards/MadLoopParams.dat' if label=='standalone' else 'SubProcesses/MadLoopParams.dat')
        require(str(active_card) in receipt['files'], 'Missing active runtime card binding')


def loader_evidence(label, pid, expected_dso, receipt):
    require(expected_dso.is_file() and not expected_dso.is_symlink() and
            expected_dso.resolve()==expected_dso.absolute(), 'Relocated DSO must be an internal regular file')
    files=sorted((WORK/'logs').glob('loader-'+label+'.*'))
    require(len(files)==1 and files[0].name=='loader-'+label+'.'+str(pid), 'Missing or extra loader log')
    providers={}
    for line in files[0].read_text().splitlines():
        if 'calling init:' in line:
            name=line.split('calling init:',1)[1].strip()
            require(name.startswith('/'), 'Unresolved loader initialization path')
            path=Path(name).resolve()
            require(path.is_file(), 'Missing runtime provider')
            providers[str(path)]=digest(path)
    require(str(expected_dso.resolve()) in providers, 'Relocated TIRust DSO was not observed')
    require(providers[str(expected_dso.resolve())] ==
            BIND['candidate']['native']['files']['lib/libmadloop_tirust.so'], 'Wrong relocated native DSO')
    require(all(name==str(expected_dso.resolve()) or name.startswith('/nix/store/')
                for name in providers), 'Unexpected non-system dynamic provider')
    require(all(receipt['runtime_providers'].get(name)==checksum for name,checksum in providers.items()),
            'Observed runtime provider was not prebound')
    return {'log':str(files[0].relative_to(WORK)), 'log_sha256':digest(files[0]),
            'initialized_providers':providers,
            'scope':'dynamic loader initialization paths; static objects bound by build receipt'}


def run_worker(executable, directory, label, cap, deadline, records):
    cap = min(cap, deadline-time.monotonic()-15)
    require(cap > 0, 'Runtime deadline exhausted')
    env = os.environ.copy()
    for name in ['LD_LIBRARY_PATH', 'LD_PRELOAD', 'LIBRARY_PATH', 'MADGRAPH_BASE', 'PYTHONPATH']:
        env.pop(name, None)
    env.update(LD_DEBUG='libs,files', LD_DEBUG_OUTPUT=str(WORK/'logs'/('loader-'+label)),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', RAYON_NUM_THREADS='1')
    log = WORK/'logs'/(label+'.log')
    started = time.monotonic()
    row={'label':label,'command':[str(executable)],'cwd':str(directory),'cap_seconds':cap,
         'pid':None,'returncode':None,'timeout':False,'cleanup_complete':False,
         'stdout':str(log.relative_to(WORK))}
    records.append(row)
    try:
        with log.open('xb') as stream:
            child = subprocess.Popen([str(executable)], cwd=directory, env=env, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=child_setup)
            row['pid']=child.pid
            try:
                row['returncode']=child.wait(timeout=cap)
            finally:
                def alive():
                    try:
                        os.killpg(child.pid, 0)
                        return True
                    except ProcessLookupError:
                        return False
                if alive():
                    os.killpg(child.pid, signal.SIGTERM)
                    end=time.monotonic()+5
                    while alive() and time.monotonic()<end:
                        child.poll()
                        time.sleep(0.05)
                    if alive():
                        os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
                row['cleanup_complete']=not alive()
                row['returncode']=child.returncode
                require(row['cleanup_complete'], 'Unreaped native process group')
    except BaseException as error:
        row['failure']=repr(error)
        row['timeout']=isinstance(error,subprocess.TimeoutExpired)
        raise
    finally:
        row['seconds_diagnostic_only']=time.monotonic()-started
        if log.is_file(): row['stdout_sha256']=digest(log)
        output=directory/'result.dat'
        raw=output.read_bytes() if output.is_file() else b''
        target=WORK/'results'/(label+'-result.dat')
        with target.open('xb') as stream:stream.write(raw)
        row['raw_result']=str(target.relative_to(WORK))
        row['raw_result_sha256']=digest(target)
        row['loader_logs']={str(path.relative_to(WORK)):digest(path)
            for path in sorted((WORK/'logs').glob('loader-'+label+'.*')) if path.is_file()}
        with (WORK/'results'/(label+'-execution.json')).open('x') as stream:
            json.dump(row,stream,indent=2,sort_keys=True);stream.write('\n')
    return row


def main():
    require(Path.cwd().resolve() == ROOT, 'Run from the required repository root')
    require((WORK/'build-admitted.json').is_file(), 'Missing separately reviewed build receipt')
    require(not (WORK/'runtime-start.json').exists(), 'Runtime already attempted')
    require(not (WORK/'source-prefixes').exists(), 'Original isolated native prefixes still available')
    protected()
    receipt=json.loads((WORK/'build-admitted.json').read_text())
    verify_build_inputs(receipt)
    sys.path.insert(0, str(WORK/'mg'))
    from madgraph.various.banner import MadLoopParam
    deadline = time.monotonic()+900
    records = []
    lock = (ROOT/'.tirust-build/numerical-tests.lock').open('a+')
    lock_deadline = min(time.monotonic()+120,deadline-15)
    acquired=False
    (WORK/'runtime-start.json').write_text(json.dumps({'pid':os.getpid(),'lock':str(lock.name),
        'scope':'fixed-mode deployment; unchecked statuses; no timing acceptance'})+'\n')
    try:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
                acquired=True
                break
            except BlockingIOError:
                require(time.monotonic() < lock_deadline, 'Numerical lock deadline')
                time.sleep(0.25)
        for label, modes in MODES.items():
            export = WORK/'relocated'/label
            directory = export/('SubProcesses/P0_uux_z' if label=='standalone' else 'SubProcesses/P0_uux_z/V0_uux_z')
            card = export/('Cards/MadLoopParams.dat' if label=='standalone' else 'SubProcesses/MadLoopParams.dat')
            original = card.read_bytes()
            reference = None
            try:
                for backend, mode in modes:
                    params=MadLoopParam(str(card))
                    params['MLReductionLib']=str(backend)
                    params['CTModeRun']=mode; params['CTModeInit']=mode; params['CheckCycle']=1
                    params['TIRustComputeFinite']=True; params['LoopPoleMode']=0; params['TIRustGlobalCache']=-1
                    params.write(str(card))
                    name=f'{label}-{backend}-{mode}'
                    (WORK/'results'/(name+'-card.dat')).write_bytes(card.read_bytes())
                    output=directory/'result.dat'
                    output.unlink(missing_ok=True)
                    row=run_worker(directory/'check',directory,name,120,deadline,records)
                    row.update(export=label,backend=backend,mode=mode)
                    row['loader']=loader_evidence(name,row['pid'],export/'lib/libmadloop_tirust.so',receipt)
                    raw=(WORK/row['raw_result']).read_bytes()
                    require(row['returncode']==0 and raw, 'Worker failed: '+name)
                    row['result']=parse(raw.decode(),mode)
                    if reference is None: reference=row['result']
                    row['comparison']=compare(reference,row['result'])
                    require(all(x['regression_passed'] for x in row['comparison'].values()),'Numerical mismatch: '+name)
                if label=='nlo':
                    params=MadLoopParam(str(card)); params['MLReductionLib']='9'
                    params['CTModeRun']=1; params['CTModeInit']=1; params['LoopPoleMode']=0
                    params['TIRustComputeFinite']=False; params.write(str(card))
                    (WORK/'results/nlo-partial-card.dat').write_bytes(card.read_bytes())
                    (directory/'result.dat').unlink(missing_ok=True)
                    row=run_worker(directory/'check',directory,'nlo-partial',60,deadline,records)
                    row['loader']=loader_evidence('nlo-partial',row['pid'],export/'lib/libmadloop_tirust.so',receipt)
                    text=(WORK/row['stdout']).read_text()
                    require(row['returncode']==1 and 'Poles-only tensors are not complete amplitudes' in text,
                            'Missing ordinary-amplitude partial-result guard')
                    row['partial_guard_passed']=True
            finally:
                card.write_bytes(original)
                require(card.read_bytes()==original, 'Card restoration failed')
        protected()
        verify_build_inputs(receipt)
        result={'complete':True,'scope':'unchecked fixed-mode deployment smoke','records':records}
    except BaseException as error:
        result={'complete':False,'failure':repr(error),'records':records}
        raise
    finally:
        result['lock_acquired']=acquired
        try:
            protected()
            result['protected_files_unchanged']=True
        except BaseException as error:
            result['complete']=False
            result['protected_files_unchanged']=False
            result['protected_error']=repr(error)
        with (WORK/'runtime-result.json').open('x') as stream:
            json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
        if acquired:fcntl.flock(lock,fcntl.LOCK_UN)
        lock.close()
    require(result['complete'], 'Incomplete deployment smoke')


if __name__=='__main__':
    main()
