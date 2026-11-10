#!/usr/bin/env python3
"""Verify retained static localization using package-local text only."""
from pathlib import Path
import gzip
import hashlib
import json
import re
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
DSO='69722452c234f494f0aa436fa24bbabfeae59100c03b05a1f0b76336cc17886b'
def raw(name):
 p=HERE/name;b=p.read_bytes();return gzip.decompress(b) if p.suffix=='.gz' else b
def sha(b):return hashlib.sha256(b).hexdigest()
def doc(name):return json.loads(raw(name))
def verify():
 index=doc('evidence-index.json');paths=list(HERE.rglob('*'))
 assert not any(p.is_symlink() for p in paths)
 assert {str(p.relative_to(HERE)) for p in paths if p.is_file()}==set(index['files'])|{'evidence-index.json'}
 for n,e in index['files'].items():
  assert not Path(n).is_absolute() and '..' not in Path(n).parts
  b=(HERE/n).read_bytes();assert len(b)==e['bytes']<=1048576 and sha(b)==e['sha256']
  assert not b.startswith((b'\x7fELF',b'!<arch>')) and Path(n).suffix not in ('.so','.a','.o','.rlib','.mod','.pyc')
  raw(n).decode('utf8')
 a=doc('analysis.json');r=doc('inputs/runtime-execution.json.gz');build=doc('inputs/build-execution.json.gz')
 assert a['library_sha256']==a['native_sha256_after_static_tools']==r['guarded_library_sha256']==build['private_library_not_loaded']['sha256']==DSO
 assert a['library_bytes']==build['private_library_not_loaded']['bytes']==69162336
 assert not a['native_ELF_payload_retained'] and not a['native_artifact_executed_or_loaded']
 assert r['execution_complete'] and build['complete']
 assert r['source_index_sha256']==sha(raw('inputs/physical-source-index.json'))=='7d9a40657b9df702fe1bdd1b3f302363813b1ba4a9514a2bcc88467f371dcd1c'
 assert r['artifact_binding_sha256']==sha(raw('inputs/binding1.json'))=='ee2933ed02f2f2c6ba1b004c0feb951553275c2967f829b5bb0efb894de7c61a'
 assert sha(raw('inputs/build-execution.json.gz'))=='59167eec84a767f0712b6905285a99564458ff0667427de87ce1921534b46632'
 for n,e in a['copies'].items():assert len(raw(n))==e['bytes'] and sha(raw(n))==e['sha256']
 symbols=[]
 for line in raw('symbols.txt').decode().splitlines():
  m=re.fullmatch(r'([0-9a-f]+) ([0-9a-f]+) ([A-Za-z]) (.*)',line);assert m
  symbols.append({'start':int(m[1],16),'bytes':int(m[2],16),'type':m[3],'name':m[4]})
 assert symbols==a['function_symbols'] and len(symbols)==3
 cayley=next(s for s in symbols if 'cayley_cancellation' in s['name'])
 layout=doc('elf-layout.json');assert layout['library_sha256']==DSO
 expected={'checked-trapped':('0x11fcecb','f20f5ec1','gram_indicator::<f64>','gram-indicator.disassembly.txt'),
           'forced-qp-trapped':('0x2523cc3','f20f5ec8','oneloop::native::dilog::series::','oneloop-series.disassembly.txt')}
 assert set(x['case'] for x in a['cases'])==set(expected)
 for case in a['cases']:
  name=case['case'];worker=r['workers'][name]
  assert worker['exit_code']==case['exit_code']==-8 and worker['progress']['process_reaped']
  stdout=raw('inputs/'+name+'.stdout.log').decode();pc=int(re.search(r'^#3\s+(0x[0-9a-f]+)',stdout,re.M)[1],16)
  assert pc==int(case['reported_PC'],16)
  mappings=[]
  for line in raw('inputs/'+name+'.maps.txt').decode().splitlines():
   f=line.split(None,5)
   if len(f)==6:
    start,end=[int(x,16) for x in f[0].split('-')]
    if start<=pc<end:mappings.append((start,end,int(f[2],16),f[5],line))
  assert len(mappings)==1;start,end,offset,path,line=mappings[0]
  assert r['runtime_sha256_after'][path]==DSO
  assert case['mapped_file']['literal_line']==line and case['mapped_file']['start']==start
  segments=[p for p in layout['PT_LOAD'] if p['offset']<=offset<p['offset']+p['filesz'] and p['flags']&1]
  assert len(segments)==1;p=segments[0]
  bias=start-(p['vaddr']+offset-p['offset']);relative=pc-bias
  assert bias==int(case['ELF_load_bias'],16) and relative==int(case['ELF_address'],16)
  owner=[s for s in symbols if s['start']<=relative<s['start']+s['bytes']]
  assert len(owner)==1 and owner[0]==case['symbol']
  assert relative-owner[0]['start']==int(case['symbol_offset'],16)
  assert not cayley['start']<=relative<cayley['start']+cayley['bytes']
  want,code,function,asm=expected[name]
  assert case['ELF_address']==want and case['instruction_hex']==code and function in owner[0]['name']
  row=re.search(r'^\s*'+want[2:]+r':\s+((?:[0-9a-f]{2}\s+)+)\s*(.*)$',raw(asm).decode(),re.M);assert row
  assert ''.join(row[1].split())==code and ' '.join(row[2].split())==case['instruction']
  assert owner[0]['name'] in raw(name+'.addr2line.txt').decode()
  control=raw('inputs/'+name+'.control.log').decode()
  assert 'CALL_ENTER WARMUP 2 2' in control and 'CALL_RETURN WARMUP 2 2' not in control
  assert not case['actual_register_operands_or_exception_subtype_available']
 gram=raw('sources/gram.rs').decode().splitlines();dilog=raw('sources/oneloop-dilog.rs').decode().splitlines()
 assert gram[723].strip()=='let w = largest / cnorm2(&delta).to_f64().sqrt();'
 assert dilog[73].strip()=='let ratio = 4.0 * std::f64::consts::PI * std::f64::consts::PI / modulus;'
 assert a['oneloop_committed_source_confirmation']['HEAD']==a['source_rationale']['oneloop_revision']=='bd703d09a78f7849a6b86a07096e6f222fc01193'
 assert a['oneloop_committed_source_confirmation']['source_matches_git_show_HEAD']
 assert a['oneloop_committed_source_confirmation']['source_sha256']==sha(raw('sources/oneloop-dilog.rs'))
 assert b'bd703d09a78f7849a6b86a07096e6f222fc01193' in raw('sources/bridge-Cargo.lock')
 assert a['oneloop_ratio_numerator']['binary64_value']==39.47841760435743
 assert all(c['process_reaped'] and c['returncode']==0 for c in a['read_only_tool_commands'])
 return {'verified':True,'cases':2,'new_DIVSD_sites':['0x11fcecb','0x2523cc3'],'both_outside_guarded_Cayley_symbol':True,'exception_subtype_or_register_values_qualified':False,'native_artifact_read_loaded_or_executed_by_verifier':False}
if __name__=='__main__':print(json.dumps(verify(),sort_keys=True))
