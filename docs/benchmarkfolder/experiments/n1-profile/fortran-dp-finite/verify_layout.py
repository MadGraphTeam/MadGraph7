#!/usr/bin/env python3
"""Supplement the frozen layout controller with loaded-library and case checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--phase',choices=['before','after'],required=True)
    args=parser.parse_args()
    meta=json.loads((args.directory/'manifest.json').read_text())
    library=Path(meta['library'])
    assert digest(library)==meta['library_sha256']
    env=dict(os.environ)
    env['LD_LIBRARY_PATH']=str(library.parent)+os.pathsep+env.get('LD_LIBRARY_PATH','')
    result={'phase':args.phase,'controller_manifest_sha256':digest(args.directory/'manifest.json'),
            'verification_script_sha256':digest(__file__),'LD_LIBRARY_PATH':env['LD_LIBRARY_PATH'],
            'library_sha256':digest(library),'variants':{}}
    for variant,row in meta['variants'].items():
        process=Path(row['directory'])
        assert digest(process/'check')==row['executable_sha256']
        assert digest(process/'TIRUST_interface.f')==row['source_sha256']
        loaded=subprocess.check_output(['ldd',str(process/'check')],env=env,text=True,stderr=subprocess.STDOUT)
        (args.directory/(variant+'-'+args.phase+'-ldd.log')).write_text(loaded)
        paths=[Path(m.group(1)) for line in loaded.splitlines()
               if (m:=re.search(r'=>\s+(/\S+)\s+\(',line))]
        tirust=[p for p in paths if p.name=='libmadloop_tirust.so']
        assert len(tirust)==1 and tirust[0].resolve()==library.resolve()
        item={'executable_sha256':digest(process/'check'),
              'loaded_libraries':{str(p.resolve()):digest(p) for p in paths}}
        if args.phase=='after':
            text=(args.directory/(variant+'-values.dat')).read_text()
            lines=text.splitlines(); cursor=0; cases=[]
            for n in range(1,7):
                for rank in range(3):
                    for mode in (1,2):
                        for precision in ('DP','QP'):
                            match=re.fullmatch(r'\s*(DP|QP)\s+(\d+)\s+(\d+)\s+(\d+)\s+([TF])\s*',lines[cursor])
                            assert match and match.groups()[:4]==(precision,str(n),str(rank),str(mode))
                            stable=match.group(5)=='T'
                            cases.append({'precision':precision,'n':n,'rank':rank,'mode':mode,'stable':stable})
                            cursor+=1
                            assert re.fullmatch('[TF]{5}',lines[cursor])
                            cases[-1]['ieee_flags_invalid_divzero_overflow_underflow_inexact']=lines[cursor]
                            cursor+=1
                            lengths=[32 if precision=='DP' else 64]*45
                            lengths+=([16]*4+[32] if precision=='DP' else [32]*5)*n
                            for length in lengths:
                                assert re.fullmatch('[0-9A-F]{'+str(length)+'}',lines[cursor])
                                cursor+=1
            assert cursor==len(lines)
            stable_count=sum(c['stable'] for c in cases)
            assert stable_count == 72, 'Every expected adapter fixture must be served'
            log=(args.directory/(variant+'-run.log')).read_text()
            totals=re.findall(r'LAYOUT_CASES\s+(\d+)\s+STABLE_CASES\s+(\d+)',log)
            assert totals==[('72',str(stable_count))]
            item.update(case_count=len(cases),stable_count=stable_count,cases=cases,
                        all_cases_served=stable_count==72,values_sha256=digest(args.directory/(variant+'-values.dat')))
        result['variants'][variant]=item
    if args.phase=='after':
        result['exact_output_and_input_mutation_parity']=(args.directory/'original-values.dat').read_bytes()==(args.directory/'scratch-values.dat').read_bytes()
        assert result['exact_output_and_input_mutation_parity']
        before=json.loads((args.directory/'before-verification.json').read_text())
        result['runtime_identities_unchanged']=all(
            result['variants'][v]['loaded_libraries']==before['variants'][v]['loaded_libraries'] and
            result['variants'][v]['executable_sha256']==before['variants'][v]['executable_sha256']
            for v in meta['variants'])
        assert result['runtime_identities_unchanged']
    (args.directory/(args.phase+'-verification.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'phase':args.phase,'verified':True},indent=2))


if __name__=='__main__':
    main()
