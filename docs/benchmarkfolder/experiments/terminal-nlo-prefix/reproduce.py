#!/usr/bin/env python3
"""Reproduce empty-prefix NLO declaration loss without modifying production."""
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'madgraph/__init__.py').is_file())
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from madgraph.loop import loop_exporters


def main():
    exporter_type = loop_exporters.LoopProcessOptimizedExporterFortranSA
    snapshot = HERE/'original-method.py'
    source = (snapshot.read_text() if snapshot.exists() else
              textwrap.dedent(inspect.getsource(exporter_type.split_HELASCALLS)))
    revision = 'c5496ca9b148b16a931214e1d541ead100ed1aad'
    archive = subprocess.check_output(['git','show',
        revision+':madgraph/loop/loop_exporters.py'],cwd=ROOT,text=True)
    nodes = [node for node in ast.walk(ast.parse(archive))
             if isinstance(node,ast.FunctionDef) and node.name=='split_HELASCALLS']
    assert len(nodes)==1
    node=nodes[0]
    assert textwrap.dedent('\n'.join(archive.splitlines()[node.lineno-1:node.end_lineno])+'\n')==source
    old = r"\bCALL MP_\w+TERMINAL_WL_\d+_\d+\("
    new = r"\bCALL MP_\w*TERMINAL_WL_\d+_\d+\("
    assert source.count(old)==1
    (HERE/'original-method.py').write_text(source)
    (HERE/'proposed-method.py').write_text(source.replace(old,new))
    rows = []
    for variant in ('original','proposed'):
        namespace = dict(vars(loop_exporters))
        exec(source if variant=='original' else source.replace(old,new),namespace)
        method = namespace['split_HELASCALLS']
        for prefix in ('TEST_',''):
            directory = HERE/(variant+'-'+('prefixed' if prefix else 'nlo-empty'))
            directory.mkdir(exist_ok=True)
            (directory/'batch.inc').write_text(
                '      SUBROUTINE %(bunch_name)s_%(bunch_number)d()\n'
                '%(terminal_ieee_use)s      IMPLICIT NONE\n'
                '%(terminal_ieee_declarations)s%(helas_calls)s\n      END\n')
            exporter = object.__new__(exporter_type)
            exporter.template_dir = str(directory)
            replacement = dict(proc_prefix=prefix,
                terminal_ieee_use='      USE, INTRINSIC :: IEEE_EXCEPTIONS\n',
                terminal_ieee_declarations='      LOGICAL WL_ENTRY_INEXACT\n')
            before = Path.cwd()
            try:
                os.chdir(directory)
                method(exporter,True,replacement,'batch.inc','',[
                    'CALL FULL()',
                    'CALL IEEE_GET_FLAG(IEEE_INEXACT,WL_ENTRY_INEXACT)\n'
                    f'CALL MP_{prefix}TERMINAL_WL_1_1()',
                    'CALL FULL()'],'entry','batch',n_helas=1)
            finally:
                os.chdir(before)
            command=['gfortran','-fno-automatic','-ffixed-line-length-132',
                     '-c','batch_1.f','batch_2.f','batch_3.f']
            ran=subprocess.run(command,cwd=directory,capture_output=True,text=True)
            (directory/'compile.log').write_text(ran.stdout+ran.stderr)
            for index in (1,3):
                assert 'IEEE_' not in (directory/f'batch_{index}.f').read_text()
            expected = variant=='proposed' or bool(prefix)
            assert (ran.returncode==0)==expected
            rows.append(dict(variant=variant,prefix=prefix,compiled=ran.returncode==0,
                             expected=expected,command=command))
    (HERE/'result.json').write_text(json.dumps(dict(
        rows=rows,all_expectations_passed=True,
        runtime_module_sha256=hashlib.sha256((ROOT/'madgraph/loop/loop_exporters.py').read_bytes()).hexdigest(),
        reference_method_sha256=hashlib.sha256(source.encode()).hexdigest(),
        reference_revision=revision,
        scope='Isolated exact exporter method and proposed one-character regex correction. '
              'No production mutation; actual empty-prefix FKS compile remains a separate check.'
    ),indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    main()
