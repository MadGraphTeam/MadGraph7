#!/usr/bin/env python3
"""Fresh dirty-source NLO export and exclusive legacy Ninja controls on CPU2."""
import sys
sys.dont_write_bytecode = True

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'madgraph/__init__.py').is_file())
REPORT = Path(__file__).resolve().parent
WORK = ROOT / '.benchmark-work/experiments/symmetric-factorization-production/dev-nlo-v3'
OUTPUT = WORK / 'nlo'
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import ninjars_benchmark as bench
from madgraph.various.banner import MadLoopParam


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name, value):
    (REPORT / name).write_text(json.dumps(value, indent=2) + '\n')


def environment():
    result = os.environ.copy()
    result.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        XDG_CONFIG_HOME=str(WORK / 'config'),
        LD_LIBRARY_PATH=str(bench.BASELINE / 'HEPTools/ninja/lib'),
        MAKEFLAGS='SHELL=' + shutil.which('bash'),
        SYMBOLICA_LOCK_DIR=str(ROOT / '.ninjars-build/symbolica-locks'))
    return result


def run(command, log):
    start = time.monotonic()
    with (REPORT / log).open('w') as stream:
        child = subprocess.run(command, cwd=ROOT, env=environment(),
            stdout=stream, stderr=subprocess.STDOUT)
    record = dict(command=command, returncode=child.returncode,
                  seconds=time.monotonic()-start, log_sha256=sha(REPORT / log))
    if child.returncode:
        raise RuntimeError(str(record))
    return record


def logical(path):
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line[0] in 'cC*!':
            continue
        body = re.sub(r'\s+', '', line[6:]).upper()
        if line[5] not in ' 0':
            rows[-1] += body
        else:
            rows.append(body)
    return rows


def inspect_calls():
    records = {}
    pattern = re.compile(r'CALL(MP_(?:UPDATE|TERMINAL)_WL_[2-4]_1_FACTOR_FFV1L[12]_[12])\(')
    for path in sorted((OUTPUT / 'SubProcesses').rglob('mp_coef_construction_*.f')):
        rows = logical(path)
        calls = [(index, pattern.match(row)) for index, row in enumerate(rows)
                 if pattern.match(row)]
        if not calls:
            continue
        assert 'USE,INTRINSIC::IEEE_EXCEPTIONS' in rows
        assert 'LOGICALWL_ENTRY_INEXACT' in rows
        polynomial = path.parent / 'polynomial.f'
        routines = '\n'.join(logical(polynomial))
        for index, match in calls:
            assert index > 0 and rows[index-1] == 'CALLIEEE_GET_FLAG(IEEE_INEXACT,WL_ENTRY_INEXACT)'
            assert 'SUBROUTINE' + match[1] + '(' in routines
            assert rows[index].endswith(',WL_ENTRY_INEXACT)')
        relative = str(path.relative_to(OUTPUT))
        records[relative] = dict(selected_calls=len(calls),
            routines=sorted({match[1] for _, match in calls}),
            empty_mp_prefix=True, fresh_flag_before_every_call=True,
            ieee_import_and_declaration=True, sha256=sha(path),
            polynomial_sha256=sha(polynomial))
        retained = REPORT / 'generated-factor-sources' / path.parent.name
        retained.mkdir(parents=True, exist_ok=True)
        for source in (path, polynomial):
            shutil.copy2(source, retained / source.name)
    assert records, 'No empty-prefix factored NLO call was generated'
    return records


def configure(backend=6, mode=-1):
    card = MadLoopParam(str(OUTPUT / 'Cards/MadLoopParams.dat'))
    for key, value in dict(MLReductionLib=str(backend), CTModeRun=mode,
                          MLStabThres=1e-9, COLLIERRequiredAccuracy=1e-10).items():
        card[key] = value
    for path in [OUTPUT / 'Cards/MadLoopParams.dat',
                 *(OUTPUT / 'SubProcesses').rglob('MadLoopParams.dat')]:
        card.write(str(path), template=str(
            ROOT / 'Template/loop_material/StandAlone/Cards/MadLoopParams.dat'))
    run_card = OUTPUT / 'Cards/run_card.dat'
    text, count = re.subn(r'(?m)^\s*\S+\s*=\s*pdlabel\b',
        ' nn23lo1 = pdlabel', run_card.read_text(), count=1)
    assert count == 1
    run_card.write_text(text)
    for name in ('param_card.dat', 'MadLoopParams.dat', 'run_card.dat'):
        shutil.copy2(OUTPUT / 'Cards' / name, REPORT / name)


def nlo_source_manifest(stage):
    """FO inventory: retain known template placeholders, reject other missing inputs."""
    suffixes = {'.f', '.f90', '.for', '.inc', '.c', '.cc', '.cpp', '.cxx', '.h', '.hpp'}
    files, missing = {}, {}
    for subtree in ('Source', 'SubProcesses'):
        for path in sorted((OUTPUT / subtree).rglob('*')):
            name = str(path.relative_to(OUTPUT))
            if path.is_symlink() and not path.exists():
                target = str(path.readlink())
                if (path.name in ('dire_fortran.cc', 'mint.inc', 'OLE_order.olc')
                        and path.parent.parent == OUTPUT / 'SubProcesses'
                        and target == '../' + path.name):
                    if path.name == 'dire_fortran.cc':
                        reason = 'Unconditional exporter link to absent optional Dire shower plugin; unused by FO targets.'
                    elif path.name == 'mint.inc':
                        reason = 'Unconditional obsolete include link; mint_module.f90 replaces it; no active include.'
                    else:
                        reason = 'Optional external OLP contract; this export uses built-in MadLoop.'
                elif (stage == 'before-build' and path.name == 'run_card.inc'
                      and path.resolve() == OUTPUT / 'Source/run_card.inc'):
                    reason = 'Pending generated run-card include; Source/makefile and treatcards create it before compilation.'
                else:
                    raise RuntimeError('Unexpected missing NLO input: ' + name)
                missing[name] = dict(target=target, resolved=str(path.resolve()), reason=reason)
                continue
            if not (path.suffix.lower() in suffixes or
                    path.name.lower().startswith('makefile') or
                    path.name in ('make_opts', 'MadLoop_makefile_definitions')):
                continue
            if path.is_file():
                files[name] = dict(sha256=sha(path), bytes=path.stat().st_size)
                if path.is_symlink():
                    files[name]['link_target'] = str(path.readlink())
    # No FO source references either absent optional source/include. Comments are excluded.
    for name in files:
        path = OUTPUT / name
        if path.name.lower().startswith('makefile'):
            assert 'dire_fortran' not in path.read_text() and 'mint.inc' not in path.read_text(), name
        elif path.suffix.lower() in ('.f', '.for'):
            assert not any(re.match(r"INCLUDE['\"]MINT\.INC['\"]", row) for row in (re.sub(r"\s+", "", line[6:]).upper() for line in path.read_text().splitlines() if len(line) >= 7 and line[0] not in "cC*!#")), name
        elif path.suffix.lower() == '.f90':
            assert not re.search(r"(?im)^\s*include\s*['\"]mint\.inc['\"]", path.read_text()), name
    assert missing and all(x['reason'] for x in missing.values())
    return dict(schema='nlo-fixed-order-source-manifest-v1', stage=stage, files=files,
                files_sha256=bench.content_digest(files), missing_template_links=missing,
                scope='Same generated source/build inventory as the frozen standalone harness, '
                      'with explicitly retained absent NLO template placeholders; all unknown '
                      'missing inputs rejected. run_card.inc must exist after compilation.')


def main():
    os.sched_setaffinity(0, {2})
    assert not OUTPUT.exists(), 'Refusing to overwrite an earlier NLO export'
    WORK.mkdir(parents=True, exist_ok=True)
    bench.REPORT = REPORT
    identity = bench.capture_generation_identity('candidate')
    source_state = bench.generator_tree_state()
    ninja = bench.BASELINE / 'HEPTools/ninja/lib'
    ninjars = ROOT / 'HEPTools/ninjars/lib'
    libraries = {}
    for prefix in (ninja, ninjars):
        for path in sorted(prefix.iterdir()):
            if path.is_file():
                libraries[str(path.relative_to(ROOT))] = dict(sha256=sha(path),
                    resolved=str(path.resolve()), bytes=path.stat().st_size)
    result = dict(complete=False, acceptance_eligible=False, cpu=2,
        process='u u~ > z g [QCD]', model='loop_sm', backend=6,
        source_identity=identity, dependencies=libraries,
        scope='Fresh actual NLO CLI export from the explicitly retained dirty '
              'generator patch. The extra gluon relative to the previous uuxZ '
              'installer smoke is required to exercise rank2+ factor helpers. '
              'External dependencies retain Ninja6, which internal exports omit. '
              'No NinjaRS evaluation or performance acceptance claim.')
    dump('result.json', result)
    try:
        commands = ['set automatic_html_opening False', 'set auto_update 0',
            'set nb_core 1', 'set ninja ' + str(ninja), 'set collier None',
            'set ninjars ' + str(ninjars), 'set output_dependencies external',
            'import model loop_sm', 'generate u u~ > z g [QCD]',
            'output ' + str(OUTPUT), 'quit']
        card = REPORT / 'nlo.mg5'
        card.write_text('\n'.join(commands) + '\n')
        result['generation'] = run(
            [sys.executable, str(ROOT / 'bin/madgraph'), '--plain', str(card)],
            'generation.log')
        assert bench.generator_tree_state() == source_state
        result['factored_batches'] = inspect_calls()
        result['selected_calls'] = sum(x['selected_calls'] for x in result['factored_batches'].values())
        dump('generated-sources.json', nlo_source_manifest('before-build'))
        configure()
        compile_card = REPORT / 'compile.mg5'
        compile_card.write_text('compile FO -f\nquit\n')
        dump('result.json', result)
        print('Fresh export selected', result['selected_calls'], 'empty-prefix factor calls', flush=True)
        result['compile_and_controls'] = run(
            [sys.executable, str(OUTPUT / 'bin/aMCatNLO'), str(compile_card)],
            'compile-and-controls.log')
        assert bench.generator_tree_state() == source_state
        text = (REPORT / 'compile-and-controls.log').read_text()
        assert 'Poles successfully cancel for 20 points over 20' in text
        assert 'Result for test_ME:' in text and 'Passed.' in text
        controls = {}
        for directory in sorted((OUTPUT / 'SubProcesses').glob('P*_*')):
            pole = directory / 'check_poles.log'
            soft = directory / 'test_ME.log'
            if not pole.is_file():
                continue
            assert re.search(r'>\s*MLReductionLib\s*=\s*6\b', pole.read_text())
            assert 'Fraction of failures: 0.00' in soft.read_text()
            assert 'FAILED' not in soft.read_text()
            for name in ('check_poles.log', 'test_ME.log'):
                shutil.copy2(directory / name, REPORT / (directory.name + '-' + name))
            executables = {name: sha(directory / name)
                for name in ('check_poles', 'madevent_mintFO')}
            controls[directory.name] = dict(pole_log_sha256=sha(pole),
                soft_collinear_log_sha256=sha(soft), executables=executables)
        assert controls
        result['controls'] = controls
        result['pole_test_tolerance'] = 1e-5
        result['pole_controls'] = '20/20 accepted by built-in analytic pole check'
        result['soft_collinear_controls'] = 'Passed with zero failures'
        dump('compiled-sources.json', nlo_source_manifest('after-build'))
        for path, record in libraries.items():
            assert sha(ROOT / path) == record['sha256']
        result['complete'] = True
        dump('result.json', result)
        print('NLO compile and Ninja6 pole/soft-collinear controls passed', flush=True)
    except Exception as error:
        result['error'] = repr(error)
        dump('result.json', result)
        raise


if __name__ == '__main__':
    main()
