#!/usr/bin/env python3
"""Static, read-only inspection of the closed build2; never runs an ELF target."""
from pathlib import Path
import gzip
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
SOURCE=HERE.parent/'source-proposal2'
BUILD=ROOT/'.tirust-build/qcd-n3-genuine-qp-diagnostic/build2'
PRIVATE=ROOT/'.tirust-build/qcd-n3-genuine-qp-diagnostic/static-audit2'

def sha(data): return hashlib.sha256(data).hexdigest()
def dump(name,value): (HERE/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def member(archive,wanted):
    raw=archive.read_bytes();assert raw[:8]==b'!<arch>\n'
    offset=8;strings=b''
    while offset<len(raw):
        header=raw[offset:offset+60];assert header[58:]==b'`\n'
        size=int(header[48:58]);data=raw[offset+60:offset+60+size]
        name=header[:16].decode().strip()
        if name=='//': strings=data
        elif name.startswith('/') and name[1:].isdigit():
            name=strings[int(name[1:]):].split(b'/\n',1)[0].decode()
        else: name=name.rstrip('/')
        if name==wanted: return data
        offset+=60+size+(size%2)
    raise ValueError('Missing archive member '+wanted)

def main():
    sys.path.insert(0,str(SOURCE));run=module('audited_run',SOURCE/'run.py')
    binding=run.source_check();build=json.loads((BUILD/'build.json').read_text())
    assert build['passed'] is True and build['elapsed_seconds']<=120
    assert build['source_index_sha256']==sha((SOURCE/'source-index.json').read_bytes())
    assert build['executable_sha256']=='ed6a51fff0312b06a45cd97a409aafa7d2447eef5ae7800579b44c0c904b8474'
    assert build['link_map_sha256']=='28baebd869a17b26e08a35f3f331679703c0f80ce0f244f0af46adc2b93401f2'
    for p,h in {build['executable']:build['executable_sha256'],**build['objects'],**build['original_objects']}.items():
        assert sha(Path(p).read_bytes())==h,p
    assert build['original_objects']=={p:binding['dependencies'][p] for p in binding['objects']}
    raw_map=(BUILD/'link.map').read_bytes();assert sha(raw_map)==build['link_map_sha256']
    assert run.exec_target.interpreter(Path(build['executable']).read_bytes())==binding['target_interpreter']
    plan=json.loads((BUILD/'plan.json').read_text());assert plan==run.commands(binding,BUILD)
    previous=None;receipts=[]
    for i,argv in enumerate(plan,1):
        p=BUILD/f'command-{i}.json';r=json.loads(p.read_text())
        assert r['argv']==run.prefix(binding)+argv and r['cwd']==str(BUILD)
        assert r['returncode']==0 and r['process_reaped'] is True and r['native'] is False
        assert not r.get('failure') and not r.get('timed_out') and r['target_loader_environment_at_execve']=={}
        assert r['seconds_cap']<=120
        if previous is not None: assert r['started_unix']>=previous
        previous=r['finished_unix'];receipts.append(r)
        shutil.copyfile(p,HERE/p.name);shutil.copyfile(BUILD/f'command-{i}.log',HERE/f'command-{i}.log')
    for name in ('build.json','plan.json'): shutil.copyfile(BUILD/name,HERE/name)
    shutil.copyfile(SOURCE/'source-index.json',HERE/'source-index.json')
    compressed=gzip.compress(raw_map,compresslevel=9,mtime=0)
    (HERE/'link.map.gz').write_bytes(compressed)
    dump('map-compression.json',{'raw_sha256':sha(raw_map),'raw_bytes':len(raw_map),'gzip_sha256':sha(compressed),'gzip_bytes':len(compressed)})
    known={str(Path(p).resolve()):h for p,h in binding['dependencies'].items()}
    loads=[]
    for line in raw_map.decode().splitlines():
        if not line.startswith('LOAD '): continue
        p=Path(line[5:]);actual=sha(p.read_bytes());canonical=str(p.resolve())
        if canonical in known: assert known[canonical]==actual
        loads.append({'path':str(p),'resolved_path':canonical,'sha256':actual,
                      'binding':'source dependency' if canonical in known else ('new diagnostic object' if str(BUILD) in canonical else 'post-link compiler support input')})
    assert set(binding['objects'])<={x['path'] for x in loads}
    dump('link-loads.json',loads)
    PRIVATE.mkdir(exist_ok=False)
    extracted=[]
    for archive,name in ((ROOT/'HEPTools/ninjars/lib/libninjars.a','mninjars.o'),
                         (ROOT/'.tirust-build/qp-transport-install/mg/HEPTools/tirust/lib/libtirust.a','mtirust.o')):
        data=member(archive,name);target=PRIVATE/name;target.write_bytes(data)
        extracted.append({'archive':str(archive),'archive_sha256':sha(archive.read_bytes()),'member':name,'member_sha256':sha(data),'path':str(target)})
    dump('archive-members.json',extracted)
    tools={n:str(Path(shutil.which(n)).resolve()) for n in ('nm','objdump','readelf')}
    env=run.env(binding);inspections=[];deadline=time.monotonic()+120
    def inspect(tool,args):
        argv=[tools[tool],*map(str,args)];start=time.monotonic()
        result=subprocess.run(argv,env=env,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                              timeout=min(30,max(0.001,deadline-start)),check=True)
        inspections.append({'argv':argv,'returncode':result.returncode,'elapsed_seconds':time.monotonic()-start,
                            'stdout_sha256':sha(result.stdout),'stdout_bytes':len(result.stdout),
                            'stderr':result.stderr.decode()})
        return result.stdout.decode()
    expected=set(run.WRAPS);objects=[Path(p) for p in binding['objects']]+[Path(x['path']) for x in extracted]
    edges=[];relocation_lines=[]
    for obj in objects:
        undefined={line.split()[-1] for line in inspect('nm',['-u',obj]).splitlines() if line.split()}
        text=inspect('objdump',['-dr',obj]);caller=None
        for line in text.splitlines():
            match=re.match(r'^[0-9a-f]+ <([^>]+)>:$',line)
            if match: caller=match[1]
            if 'R_X86_64_' not in line: continue
            fields=line.split();symbol=re.split(r'[+-]0x',fields[-1])[0]
            if symbol in expected:
                edges.append({'object':str(obj),'object_sha256':sha(obj.read_bytes()),'caller':caller,'target':symbol,'relocation':line.strip(),
                              'target_undefined_in_object':symbol in undefined})
                relocation_lines.append(str(obj)+' :: '+str(caller)+' :: '+line.strip())
    (HERE/'external-relocations.txt').write_text('\n'.join(relocation_lines)+'\n')
    assert {x['target'] for x in edges if x['target_undefined_in_object']}==expected
    symbols=inspect('nm',['-g',build['executable']]);selected=[]
    for line in symbols.splitlines():
        symbol=line.split()[-1] if line.split() else ''
        if symbol.removeprefix('__wrap_') in expected or symbol.startswith('__real_'): selected.append(line)
    (HERE/'executable-symbols.txt').write_text('\n'.join(selected)+'\n')
    for target in expected:
        matches=[line for line in selected if line.split()[-1]=='__wrap_'+target]
        assert len(matches)==1 and matches[0].split()[-2]=='T',target
    assert not any('__real_' in line for line in selected)
    final_edges=[];disassembly_lines=[]
    callers=sorted({x['caller'] for x in edges}|{'__wrap_'+s for s in expected})
    for caller in callers:
        text=inspect('objdump',['-d','--disassemble='+caller,build['executable']])
        for line in text.splitlines():
            if not re.search(r'\b(?:call|jmp|lea|mov)\b',line): continue
            targets=re.findall(r'<([^>]+)>',line)
            for target in targets:
                base=target.split('@')[0]
                if base.removeprefix('__wrap_') not in expected: continue
                final_edges.append({'caller':caller,'target':base,'instruction':line.strip()})
                disassembly_lines.append(caller+' :: '+line.strip())
    (HERE/'linked-edges.txt').write_text('\n'.join(disassembly_lines)+'\n')
    assert {'__wrap_'+s for s in expected}<={x['target'] for x in final_edges if not x['caller'].startswith('__wrap_')}
    for target in expected:
        assert any(x['caller']=='__wrap_'+target and x['target']==target for x in final_edges),target
    elf=inspect('readelf',['-W','-l','-d',build['executable']]);(HERE/'executable-elf.txt').write_text(elf)
    dump('wrapper-audit.json',{'expected_wrappers':list(run.WRAPS),'object_relocations':edges,'linked_edges':final_edges,
         'all_wrappers_have_external_relocations_and_linked_incoming_edges':True,'all_wrappers_forward_to_real_symbol':True,
         'static_edges_do_not_prove_runtime_traversal':True})
    dump('inspection-commands.json',{'tools':{n:{'path':p,'sha256':sha(Path(p).read_bytes())} for n,p in tools.items()},'commands':inspections})
    run.source_check()
    for x in loads: assert sha(Path(x['path']).read_bytes())==x['sha256']
    assert sha(Path(build['executable']).read_bytes())==build['executable_sha256']
    dump('summary.json',{'scope':'closed build2 static audit only; no target execution or loader invocation',
      'passed':True,'source_index_sha256':build['source_index_sha256'],'executable_sha256':build['executable_sha256'],
      'link_map_sha256':build['link_map_sha256'],'build_elapsed_seconds':build['elapsed_seconds'],
      'original_objects':len(binding['objects']),'wrapper_count':len(expected),'matching_object_relocations':len(edges),
      'external_relocations':sum(x['target_undefined_in_object'] for x in edges),
      'same_object_relocations':sum(not x['target_undefined_in_object'] for x in edges),
      'linked_edges':len(final_edges),'load_records':len(loads),'unique_resolved_loads':len({x['resolved_path'] for x in loads}),
      'post_link_support_inputs':len({x['resolved_path'] for x in loads if x['binding']=='post-link compiler support input'}),
      'root_reported_terminal':{'session':63632,'exit_code':0,'attribution':'reported by root; actual four retained command receipts independently checked'},
      'limits':['Runtime closure and plain/observed parity remain unexecuted.',
                'Compiler startup objects/scripts and selected support libraries are post-link snapshots, not previously frozen inputs.',
                'Linked libquadmath path differs from intended runtime libquadmath: runtime loader/maps gates must resolve the exact bound path.',
                'Historical donor object identity is not claimed; current frozen twenty-object identities are preserved.',
                'Internal DP-repair to QP-repair call in improve_ps.o is same-object and unwrapped; public QP-repair from loop_matrix.o is wrapped.',
                'Linker GNU-stack warning is retained; no scientific conclusion or runtime success is inferred.']})
    print('Static audit PASS:',len(expected),'wrappers;',len(edges),'external relocations;',len(final_edges),'linked edges')

if __name__=='__main__': main()
