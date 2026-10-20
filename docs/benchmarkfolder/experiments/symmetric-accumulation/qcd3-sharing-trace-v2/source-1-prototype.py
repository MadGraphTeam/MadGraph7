#!/usr/bin/env python3
"""Apply guarded product sharing only to isolated generated QP call sites."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import tempfile

import gather
import shared
from madgraph.iolibs.file_writers import FortranWriter
from madgraph.various.q_polynomial import FortranPolynomialRoutines


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def statements(text):
    result=[]
    for index,line in enumerate(text.splitlines(True)):
        if not line.strip() or line[0] in 'cC*!':
            continue
        assert len(line)>=6
        body=re.sub(r'\s+','',line[6:]).upper()
        if line[5] not in ' 0':
            assert result
            result[-1][2]+=body
        else:
            result.append([index,line[:6],body])
    return result


def render(source):
    with tempfile.TemporaryDirectory(prefix='sharing-render-',
            dir=gather.ROOT/'.benchmark-work') as temporary:
        path=Path(temporary)/'polynomial.f'
        with FortranWriter(str(path)) as writer:
            writer.writelines(source)
        return path.read_text()


def transform(directory,trace=False):
    directory=Path(directory)
    source=(directory/'polynomial.f').read_text()
    module=re.search(r'(?m)^      MODULE (\w+)POLYNOMIAL_CONSTANTS$',source)
    assert module
    prefix=module[1]
    gen=FortranPolynomialRoutines(4,updater_max_rank=1,proc_prefix=prefix,
        sub_prefix='MP_'+prefix,coef_format='complex*32')
    pattern=re.compile(r'CALL(MP_\w*(UPDATE|TERMINAL)_WL_(\d+)_1)'
        r'\(WL\(1,0,1,(-?\d+)\),(\d+),COEFS,(\d+),(\d+),'
        r'WL\(1,0,1,(-?\d+)\)(,WL_ENTRY_INEXACT)?\)$')
    known=json.loads((shared.HERE/'inventory.json').read_text())['qcd-3']['routines']
    files={}
    entries=[]
    helper_keys={}
    vertices={}
    for path in sorted(directory.glob('mp_coef_construction_*.f')):
        text=path.read_text()
        lines=text.splitlines(True)
        vertex=None
        for index,label,row in statements(text):
            match=re.fullmatch(r'CALL(MP_\w+)\(.*?,COEFS\)',row)
            if match:
                vertex=match[1]
                continue
            if row.startswith('CALLIEEE_GET_FLAG('):
                continue
            match=pattern.fullmatch(row)
            if match and int(match[3])>=2 and match[6]==match[7]=='4':
                assert vertex
                terminal=match[2]=='TERMINAL'
                assert bool(match[9])==terminal
                assert match[4]!=match[8], 'Aliasing input/output wavefunctions'
                if vertex in known and shared.groups(vertex,terminal):
                    key=(int(match[3]),terminal,vertex)
                    suffix='_SHARED_'+vertex.removeprefix('MP_')
                    newname=match[1]+suffix
                    assert len(newname)<=63
                    assert lines[index].count(match[1])==1
                    lines[index]=lines[index].replace(match[1],newname,1)
                    assert len(lines[index].rstrip())<=132
                    helper_keys[key]=(match[1],newname,suffix)
                    entries.append(dict(file=path.name,line=index+1,label=label[:5],
                        vertex=vertex,rank=key[0],terminal=terminal,
                        input_wavefunction=int(match[4]),output_wavefunction=int(match[8]),
                        original=match[1],replacement=newname))
                    vertex_path=directory.parents[1]/'Source/DHELAS'/(vertex+'.f')
                    vertex_text=vertex_path.read_text()
                    assignments={}
                    for _,_,statement in statements(vertex_text):
                        assignment=re.fullmatch(r'COEFF\((\d+),(\d+),(\d+)\)=(.*)',statement)
                        if assignment:
                            index_key=tuple(map(int,assignment.groups()[:3]))
                            assert index_key not in assignments
                            assignments[index_key]=assignment[4]
                    groups=shared.groups(vertex,terminal)
                    for group in groups:
                        expressions=[assignments[tuple(member)] for member in group]
                        assert len(set(expressions))==1
                        assert expressions[0] not in ('0','0D0','0Q0')
                    vertices[vertex]=dict(source_sha256=sha(vertex_text),
                        relative_source=str(vertex_path.relative_to(directory.parents[1])))
            if row.startswith('CALL') or 'COEFS' in row:
                vertex=None
        after=''.join(lines)
        if after!=text:
            files[path.name]=after
    assert entries
    helpers=[]
    count=len(helper_keys)
    for helper_index,(key,(old,new,suffix)) in enumerate(sorted(helper_keys.items()),1):
        rank,terminal,vertex=key
        # Bind the generated oracle to the actual process's routine before
        # deriving a private entry. No public source is replaced.
        original=gen.write_compact_wl_updater(rank,1,terminal=terminal)
        if not terminal:
            original=original.split('\n\nSUBROUTINE '+old+'_DENSE(',1)[0]
        assert render(original).strip() in source
        after=shared.generate(gen,rank,vertex,terminal,suffix)
        if trace:
            after=after.replace('  INTEGER I,J,K,L,M\n',
                f'  INTEGER I,J,K,L,M\n  INTEGER*8 SHARE_COUNTS(2,{count})\n'
                '  COMMON/SHARING_COUNTS/SHARE_COUNTS\n',1)
            after=after.replace('PRODUCT_READY=.FALSE.',
                f'SHARE_COUNTS(1,{helper_index})=SHARE_COUNTS(1,{helper_index})+1\n'
                'PRODUCT_READY=.FALSE.',1)
            after=re.sub(r'(?m)^( +)(CALL MP_\w*(?:UPDATE|TERMINAL)_WL_\d+_1(?:_DENSE)?\()',
                rf'\1SHARE_COUNTS(2,{helper_index})=SHARE_COUNTS(2,{helper_index})+1\n\1\2',after)
        source+='\n'+render(after)
        helpers.append(dict(index=helper_index,rank=rank,terminal=terminal,
            vertex=vertex,original=old,replacement=new,
            groups=shared.groups(vertex,terminal)))
    original_source=(directory/'polynomial.f').read_text()
    assert source.startswith(original_source), 'Original public/DP source modified'
    files['polynomial.f']=source
    if trace:
        path=directory/'check_sa.f'
        driver=path.read_text()
        assert driver.count('      IMPLICIT NONE')==1
        driver=driver.replace('      IMPLICIT NONE',
            f'      IMPLICIT NONE\n      INTEGER*8 SHARE_COUNTS(2,{count})\n'
            '      COMMON/SHARING_COUNTS/SHARE_COUNTS\n'
            f'      DATA SHARE_COUNTS/{2*count}*0/',1)
        marker="      OPEN(6,FILE='/dev/null',STATUS='OLD')"
        assert driver.count(marker)==1
        driver=driver.replace(marker,marker+'\n      SHARE_COUNTS=0',1)
        marker='      ELAPSED=DBLE(T1-T0)/DBLE(RATE)'
        assert driver.count(marker)==1
        driver=driver.replace(marker,marker+
            "\n      OPEN(74,FILE='sharing_trace.dat',STATUS='REPLACE')\n"
            f'      DO I=1,{count}\n'
            "         WRITE(74,'(3I20)') I,SHARE_COUNTS(:,I)\n"
            '      ENDDO\n      CLOSE(74)',1)
        files['check_sa.f']=driver
    proof=dict(trace=trace,helpers=helpers,calls=entries,vertices=vertices,
        call_counts=dict(Counter(('terminal' if e['terminal'] else 'full')+
                                '-rank'+str(e['rank']) for e in entries)),
        original_polynomial_sha256=sha(original_source),
        original_polynomial_preserved_as_prefix=True,
        generator_sha256=sha((gather.ROOT/'madgraph/various/q_polynomial.py').read_text()),
        scope='Private isolated QP helpers; same-K or terminal same-(J,K) source '
              'expressions plus runtime complex-bit equality. Original public, '
              'terminal and all DP routines are unchanged.')
    return files,proof
