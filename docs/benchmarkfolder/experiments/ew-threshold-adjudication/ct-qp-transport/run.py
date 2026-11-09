#!/usr/bin/env python3
"""Private EW CutTools QP-denominator transport diagnostic, unchanged DP init."""
import argparse
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[4]
OLD = ROOT/'docs/benchmarkfolder/experiments/physical-stress'
DIAGNOSTIC = OLD/'diagnostics/ew_n3_epem_ttxa/candidate-1-qp-init-dp'
ORIGIN = ROOT/'.tirust-build/physical-stress/unchecked/ew_n3_epem_ttxa/candidate-1-qp-init-dp'
WORK = ROOT/'.tirust-build/ew-threshold-ct-qp-transport'
PROCESS = Path('SubProcesses/P0_epem_ttxa')
PRIVATE_DSO = ROOT/'.tirust-build/ew-threshold-geometry/private-libraries/libmadloop_tirust.so'
EXPECTED_DSO = '49af12f0b88252a2d06420a2ea8768529eb5ab51de34e3db2257ee61cd71ca12'
VENDOR = ROOT/'vendor/CutTools/src/cts/cts_cutroutines.f90'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True)+'\n')


def inventory(path):
    return {str(p.relative_to(path)): digest(p)
            for p in sorted(list(path.rglob('*.o'))+list(path.rglob('*.a')))}


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Unexpected source anchor: '+old)
    return text.replace(old, new)


def patch_interface(text):
    original_dp, tail = text.split('      SUBROUTINE ML5_0_INITCT()\n', 1)
    qp = original_dp
    for before, after in (
        ('SUBROUTINE ML5_0_CTLOOP(', 'SUBROUTINE ML5_0_CTLOOP_QPINPUT('),
        ('      REAL*8 PL(0:3,NLOOPLINE)', '      REAL*16 PL(0:3,NLOOPLINE)'),
        ('      REAL*8 PCT(0:3,0:NLOOPLINE-1),ABSPCT(0:3)',
         '      REAL*16 PCT(0:3,0:NLOOPLINE-1),ABSPCT(0:3)'),
        ('      REAL*8 REF_P', '      REAL*16 REF_P'),
        ('      CALL CTSXCUT(CTMODE', '      CALL CTSXCUT_QPINPUT(CTMODE'),
        ('C     INITIALIZE CUTTOOLS IF NEEDED',
         "      IF(CTMODE.LT.4.OR.CTMODE.GT.6)\n"
         "     $ STOP 'QP denominator transport requires mode 4, 5 or 6'\n"
         'C     INITIALIZE CUTTOOLS IF NEEDED'),
    ):
        qp = once(qp, before, after)
    call = ('          CALL ML5_0_CTLOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1\n'
            '     $     ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))')
    replacement = ('          IF (DOING_QP) THEN\n'
        '            CALL ML5_0_CTLOOP_QPINPUT(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1\n'
        '     $       ,SQUAREDSOINDEX,LOOPNUM),S(SQUAREDSOINDEX,LOOPNUM))\n'
        '          ELSE\n'+call+'\n          ENDIF')
    if tail.count(call) != 7:
        raise ValueError('Unexpected EW CutTools call count')
    tail = tail.replace(call, replacement)
    result = original_dp+qp+'      SUBROUTINE ML5_0_INITCT()\n'+tail
    assert result.startswith(original_dp)
    return result


def patch_cuttools(text):
    # Same reviewed private-symbol delta as the CMS transport experiment.
    text = once(text, '  subroutine ctsxcut(imode,', '  subroutine ctsxcut_qpinput(imode,')
    text = once(text, '  end subroutine ctsxcut\n', '  end subroutine ctsxcut_qpinput\n')
    text = once(text,
        "  include 'cts_dpr.h'\n   , intent(in), dimension(0:3,0:(number_propagators-1)) :: p",
        "  include 'cts_mpr.h'\n   , intent(in), dimension(0:3,0:(number_propagators-1)) :: p")
    old = '      do k= 0,3; mp_dn(j)%p(k)= dn(j)%p(k); enddo'
    if text.count(old) != 2:
        raise ValueError('Unexpected direct/reverse promotion sites')
    text = text.replace(old, '      do k= 0,3; mp_dn(j)%p(k)= p(k,j); enddo', 1)
    text = text.replace(old, '      do k= 0,3; mp_dn(j)%p(k)= p(k,number_propagators-1-j); enddo', 1)
    return text


def pair(stem, original, patched):
    (REPORT/'sources'/(stem+'.original')).write_text(original)
    (REPORT/'sources'/stem).write_text(patched)
    (REPORT/'sources'/(stem+'.patch')).write_text(''.join(difflib.unified_diff(
        original.splitlines(True), patched.splitlines(True), fromfile=stem+'.original', tofile=stem)))


def external_archives(expected):
    observed = {p: digest(Path(p)) for p in expected}
    if observed != expected:
        raise ValueError('External archive differs from historical compilation')
    return observed


def prepare():
    if WORK.exists() or (REPORT/'prepared.json').exists():
        raise ValueError('Refusing to overwrite prepared evidence')
    historical = json.loads((DIAGNOSTIC/'diagnostic-compilation.json').read_text())
    for category in ('sources', 'numerical_objects_after'):
        for p, expected in historical[category].items():
            if digest(ORIGIN/p) != expected:
                raise ValueError('Historical input differs: '+p)
    if digest(PRIVATE_DSO) != EXPECTED_DSO:
        raise ValueError('Historical private DSO mismatch')
    archives = external_archives(historical['link_inputs_after']['archives'])
    before = inventory(ORIGIN)
    shutil.copytree(ORIGIN, WORK, symlinks=True,
                    ignore=shutil.ignore_patterns('worker-raw', 'worker-protocol.fifo'))
    proc = WORK/PROCESS
    (REPORT/'sources').mkdir()
    runtime = WORK/'private-libraries'
    runtime.mkdir()
    shutil.copy2(PRIVATE_DSO, runtime/PRIVATE_DSO.name)
    for name in ('param_card.dat', 'MadLoopParams.dat'):
        destinations = [proc/name]
        if name == 'MadLoopParams.dat':
            # A cwd reduction card changes discovery of all MadLoop resources.
            # Preserve the historical resource-directory lookup as one unit.
            (proc/name).unlink(missing_ok=True)
            destinations = [proc/'MadLoop5_resources'/name]
        for dest in destinations:
            if not dest.parent.resolve().is_relative_to(WORK):
                raise ValueError('Private card path resolves outside clone')
            dest.unlink(missing_ok=True)
            shutil.copy2(DIAGNOSTIC/name, dest)
        shutil.copy2(DIAGNOSTIC/name, REPORT/name)
    points = OLD/'points/ew_n3_epem_ttxa/selected.dat'
    shutil.copy2(points, REPORT/'selected.dat')
    shutil.copy2(points, proc/'selected.dat')
    original = (proc/'CT_interface.f').read_text()
    modified = patch_interface(original)
    (proc/'CT_interface.f').unlink()
    (proc/'CT_interface.f').write_text(modified)
    pair('CT_interface.f', original, modified)
    original = VENDOR.read_text()
    modified = patch_cuttools(original)
    private = proc/'qpinput'
    private.mkdir()
    for p in VENDOR.parent.glob('*.h'):
        shutil.copy2(p, private/p.name)
    (private/'cts_cutroutines_qpinput.f90').write_text(modified)
    pair('cts_cutroutines_qpinput.f90', original, modified)
    overlay = ('include makefile\n'
        'LINKLIBS := cts_cutroutines_qpinput.o $(subst -lmadloop_tirust,'+
        str(runtime/PRIVATE_DSO.name)+',$(LINKLIBS))\n'
        'ct-transport-build:\n'
        '\t$(FC) $(FFLAGS) -c CT_interface.f -o CT_interface.o $(LOOP_INCLUDE)\n'
        '\tcd qpinput && $(FC) -w -fPIC -I'+str(ROOT/'vendor/CutTools/includects')+
        ' -c cts_cutroutines_qpinput.f90 -o ../cts_cutroutines_qpinput.o\n'
        '\t$(FC) $(FFLAGS) -o check check_sa.o $(PROCESS) $(LINKLIBS)\n')
    (proc/'ct_transport.mk').write_text(overlay)
    (REPORT/'ct_transport.mk').write_text(overlay)
    controller = OLD/'protocol-sources/tirust_benchmark_worker.py'
    shutil.copy2(controller, REPORT/'sources'/controller.name)
    dump(REPORT/'prepared.json', {
        'origin': str(ORIGIN.relative_to(ROOT)), 'work': str(WORK.relative_to(ROOT)),
        'historical_compilation_sha256': digest(DIAGNOSTIC/'diagnostic-compilation.json'),
        'original_sources': historical['sources'], 'objects_archives_before': before,
        'external_archives_before': archives, 'run_script_sha256': digest(Path(__file__)),
        'patched_interface_sha256': digest(proc/'CT_interface.f'),
        'private_source_sha256': digest(private/'cts_cutroutines_qpinput.f90'),
        'private_headers': {p.name: digest(p) for p in sorted(private.glob('*.h'))},
        'vendor_source_sha256': digest(VENDOR), 'private_dso_sha256': digest(runtime/PRIVATE_DSO.name),
        'driver_sha256': digest(proc/'check_sa.f'), 'point_sha256': digest(points),
        'card_sha256': {p: digest(REPORT/p) for p in ('param_card.dat','MadLoopParams.dat')},
        'controller_sha256': digest(REPORT/'sources'/controller.name),
        'overlay_sha256': digest(REPORT/'ct_transport.mk'),
        'policy': {'backend': 1, 'init_mode': 1, 'run_mode': 4, 'scalar_cutoff': 1e-6,
                   'stability_checks': False, 'QP_numerator': True},
        'scope': 'Private denominator-transport diagnostic; original DP initialization and scalar cutoff.',
        'performance_eligible': False})


def verify(record):
    assert inventory(ORIGIN) == record['objects_archives_before']
    for p, expected in record['original_sources'].items():
        assert digest(ORIGIN/p) == expected, p
    proc = WORK/PROCESS
    assert digest(proc/'CT_interface.f') == record['patched_interface_sha256']
    assert digest(proc/'qpinput/cts_cutroutines_qpinput.f90') == record['private_source_sha256']
    assert digest(proc/'ct_transport.mk') == record['overlay_sha256']
    assert digest(REPORT/'ct_transport.mk') == record['overlay_sha256']
    assert {p.name: digest(p) for p in sorted((proc/'qpinput').glob('*.h'))} == record['private_headers']
    assert digest(Path(__file__)) == record['run_script_sha256']
    assert digest(VENDOR) == record['vendor_source_sha256']
    assert digest(proc/'check_sa.f') == record['driver_sha256']
    assert digest(proc/'selected.dat') == record['point_sha256']
    assert digest(WORK/'private-libraries'/PRIVATE_DSO.name) == record['private_dso_sha256']
    for name, expected in record['card_sha256'].items():
        if name == 'MadLoopParams.dat':
            assert not (proc/name).exists()
            assert digest(proc/'MadLoop5_resources'/name) == expected
        else:
            assert digest(proc/name) == expected
    return external_archives(record['external_archives_before'])


def build():
    record = json.loads((REPORT/'prepared.json').read_text())
    verify(record)
    assert inventory(WORK) == record['objects_archives_before']
    proc = WORK/PROCESS
    command = ['make','-f','ct_transport.mk','SHELL='+shutil.which('bash'),'-j1','ct-transport-build']
    with (REPORT/'build.log').open('x') as log:
        result = subprocess.run(command, cwd=proc, env=dict(os.environ,PWD=str(proc)),
                                stdout=log, stderr=subprocess.STDOUT, timeout=180)
    dump(REPORT/'build-status.json', {'command':command,'returncode':result.returncode})
    result.check_returncode()
    after = inventory(WORK)
    changed = sorted(p for p,v in record['objects_archives_before'].items() if after.get(p)!=v)
    added = sorted(after.keys()-record['objects_archives_before'].keys())
    assert changed == [str(PROCESS/'CT_interface.o')], changed
    assert added == [str(PROCESS/'cts_cutroutines_qpinput.o')], added
    archives = verify(record)
    dump(REPORT/'compilation.json', {'objects_archives_after':after,
        'changed_objects':changed,'added_objects':added,'external_archives_after':archives,
        'executable_sha256':digest(proc/'check'),'build_log_sha256':digest(REPORT/'build.log'),
        'original_inputs_unchanged':True})


def run(cpu):
    record = json.loads((REPORT/'prepared.json').read_text())
    compiled = json.loads((REPORT/'compilation.json').read_text())
    verify(record)
    proc = WORK/PROCESS
    assert inventory(WORK) == compiled['objects_archives_after']
    assert digest(proc/'check') == compiled['executable_sha256']
    controller = REPORT/'sources/tirust_benchmark_worker.py'
    assert digest(controller) == record['controller_sha256']
    spec = importlib.util.spec_from_file_location('ct_ew_frozen_worker', controller)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env = dict(os.environ,LD_LIBRARY_PATH=str(WORK/'private-libraries')+':'+os.environ.get('LD_LIBRARY_PATH',''))
    resolved = subprocess.check_output(['ldd','./check'],cwd=proc,env=env,text=True)
    (REPORT/'ldd.txt').write_text(resolved)
    assert str(WORK/'private-libraries'/PRIVATE_DSO.name) in resolved
    outcome = {'cpu':cpu,'performance_eligible':False,'stability_certified':False,
        'policy':record['policy'],'private_dso_sha256':record['private_dso_sha256'],
        'ldd_sha256':digest(REPORT/'ldd.txt'),'uses_symbolica_api':False}
    worker = None
    try:
        # CutTools-only: explicitly authorized independent CPU; no Symbolica API.
        with module.Worker(['taskset','-c',str(cpu),'./check',str(proc/'selected.dat')],
                proc,REPORT/'output.log',env=env,timeout=180.) as worker:
            worker.run(1)
            shutil.copytree(worker.last_raw_directory,REPORT/'raw')
        outcome['complete'] = True
    except BaseException as exc:
        outcome.update(complete=False,error={'type':type(exc).__name__,'message':str(exc)})
        raise
    finally:
        for name in ('values.dat','dimensions.dat','timed_status.dat'):
            if (proc/name).exists():
                shutil.copy2(proc/name,REPORT/name)
        outcome['external_archives_after'] = verify(record)
        outcome['original_inputs_unchanged'] = True
        assert inventory(WORK) == compiled['objects_archives_after']
        outcome['output_sha256'] = digest(REPORT/'output.log')
        if (REPORT/'values.dat').exists():
            outcome['values_sha256'] = digest(REPORT/'values.dat')
        dump(REPORT/'execution.json',outcome)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','build','run'))
    parser.add_argument('--cpu',type=int,default=3)
    args = parser.parse_args()
    {'prepare':prepare,'build':build,'run':lambda:run(args.cpu)}[args.action]()
