#!/usr/bin/env python3
"""Recheck retained production-port rows and the current rendered-source proof."""
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'madgraph/__init__.py').is_file())
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from madgraph.loop.loop_exporters import LoopProcessOptimizedExporterFortranSA as Exporter
from madgraph.various import q_polynomial


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def logical(path):
    statements = []
    for raw in path.read_text().splitlines():
        if not raw.strip() or raw[0] in 'cC*!':
            continue
        if raw[5] not in ' 0':
            statements[-1] += raw[6:].strip().upper()
        else:
            statements.append(raw[6:].strip().upper())
    return '\n'.join(statements)


def main():
    retained = json.loads((HERE/'verification.json').read_text())
    result = {'acceptance_eligible': False, 'cases': {}}
    pattern = re.compile(r'CALL MP_(\w*)TERMINAL_WL_(\d+)_(\d+)'
        r'\(WL\(1,0,1,(-?\d+)\),(\d+),COEFS,(\d+),(\d+),'
        r'WL\(1,0,1,(\d+)\),WL_ENTRY_INEXACT\)')
    for name, count in [('qcd-3',224), ('ew-2',76), ('control-mixed',151), ('control-gg-h',4)]:
        report = HERE/name
        recorded = json.loads((report/'result.json').read_text())
        assert recorded['complete'] and recorded['all_rows_and_statuses_exact']
        points = ROOT/'docs/benchmarkfolder/points'/(name+'.dat')
        npoints = int(points.read_text().split()[0])
        dims = list(map(int, (report/'production-dimensions.dat').read_text().split()))
        rows = [line.split() for line in (report/'production-values.dat').read_text().splitlines()]
        assert [(int(row[0]),int(row[1])) for row in rows] == [
            (point,order) for point in range(1,npoints+1) for order in range(dims[1]+1)]
        assert all(math.isfinite(float(value)) for row in rows for value in row[3:])
        for filename in ('values.dat', 'timed_status.dat'):
            assert (report/('original-'+filename)).read_bytes() == (report/('production-'+filename)).read_bytes()
        statuses = collections.Counter(int(row[2]) for row in rows if int(row[1]) == 0)
        assert set(statuses) <= ({120} if name.startswith('control-') else {216,328})
        directory = next((ROOT/'.benchmark-work/experiments/terminal-production'/name/'production/SubProcesses').glob('P*_*'))
        expected = {}
        prefixes = set()
        for path in directory.glob('mp_coef_construction_*.f'):
            for match in pattern.finditer(logical(path)):
                prefix, left, right, origin, cut, ins, outs, target = match.groups()
                prefixes.add(prefix)
                left,right,origin,cut,ins,outs,target = map(int,(left,right,origin,cut,ins,outs,target))
                assert target not in expected
                expected[target] = dict(loop_mother_rank=left,vertex_rank=right,
                    loop_mother_number=origin,lcut_size=cut,in_size=ins,out_size=outs,out=target)
        assert len(expected) == count and len(prefixes) == 1
        assert Exporter.validate_terminal_wavefunctions(str(directory),prefixes.pop(),expected)
        result['cases'][name] = dict(physical_points=npoints, dimensions=dims,
            rows=len(rows), statuses=dict(statuses), private_calls=count,
            rows_and_status_exact=True, current_rendered_proof=True,
            original_values_sha256=sha(report/'original-values.dat'),
            production_values_sha256=sha(report/'production-values.dat'),
            points_sha256=sha(points), result_sha256=sha(report/'result.json'))
    revision = retained['public_routine_source_parity']['reference_revision']
    namespace = {'__name__': 'reference_q_polynomial'}
    exec(subprocess.check_output(['git','show',revision+':madgraph/various/q_polynomial.py'],
                                cwd=ROOT,text=True), namespace)
    comparisons = 0
    for kind in ('complex*16','complex*32'):
        current = q_polynomial.FortranPolynomialRoutines(6,coef_format=kind)
        reference = namespace['FortranPolynomialRoutines'](6,coef_format=kind)
        for left in range(4):
            for right in range(4):
                assert current.write_expanded_wl_updater(left,right) == reference.write_expanded_wl_updater(left,right)
                comparisons += 1
                for order in (False,True):
                    assert current.write_compact_wl_updater(left,right,order) == reference.write_compact_wl_updater(left,right,order)
                    comparisons += 1
    result['public_routine_source_parity'] = dict(comparisons=comparisons,reference_revision=revision)
    result['current_source_sha256'] = {name:sha(ROOT/name) for name in [
        'madgraph/various/q_polynomial.py','madgraph/iolibs/helas_call_writers.py',
        'madgraph/loop/loop_exporters.py','madgraph/iolibs/template_files/loop_optimized/mp_helas_calls_split.inc']}
    result['verifier_sha256'] = sha(__file__)
    # Normalize integer dictionary keys as they appear in the retained JSON.
    result = json.loads(json.dumps(result))
    assert result == retained, 'Current validation differs from the retained evidence'
    print('PASS: complete numerical rows, rendered consumers and pinned public-source parity')


if __name__ == '__main__':
    main()
