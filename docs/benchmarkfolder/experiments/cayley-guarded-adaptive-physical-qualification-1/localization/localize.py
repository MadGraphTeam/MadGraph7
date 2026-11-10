#!/usr/bin/env python3
"""Static read-only decoding of already closed SIGFPE evidence."""
from pathlib import Path
import gzip
import hashlib
import json
import re
import shutil
import struct
import subprocess
import sys
sys.dont_write_bytecode=True
ROOT=Path.cwd().resolve()
OUT=ROOT/'docs/benchmarkfolder/experiments/cayley-guarded-adaptive-physical-qualification-1/localization'
RUN=ROOT/'.tirust-build/cayley-guarded-adaptive-physical-proposal-2/run1'
BUILD=ROOT/'.tirust-build/cayley-guarded-adaptive-build-proposal-3/attempt1'
DSO=BUILD/'runtime/libmadloop_tirust_adaptive_private.so'
EXPECTED='69722452c234f494f0aa436fa24bbabfeae59100c03b05a1f0b76336cc17886b'

def sha(raw):return hashlib.sha256(raw).hexdigest()
def save(name, value):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def main():
 assert not (OUT/'analysis.json').exists()
 copies={}
 def copy(path,name):
  raw=path.read_bytes();raw.decode('utf8')
  target=OUT/name;target.parent.mkdir(parents=True,exist_ok=True)
  target.write_bytes(gzip.compress(raw,mtime=0) if target.suffix=='.gz' else raw)
  copies[name]={'path':str(path.relative_to(ROOT)),'sha256':sha(raw),'bytes':len(raw),'encoding':'gzip-utf8' if target.suffix=='.gz' else 'utf8'}
  return raw
 runtime=json.loads(copy(RUN/'execution.json','inputs/runtime-execution.json.gz'))
 build=json.loads(copy(BUILD/'execution.json','inputs/build-execution.json.gz'))
 assert build['complete'] and runtime['execution_complete']
 assert build['private_library_not_loaded']['sha256']==runtime['guarded_library_sha256']==EXPECTED
 assert runtime['source_index_sha256']=='7d9a40657b9df702fe1bdd1b3f302363813b1ba4a9514a2bcc88467f371dcd1c'
 copy(ROOT/'docs/benchmarkfolder/experiments/cayley-guarded-adaptive-physical-proposal-2/source-index.json','inputs/physical-source-index.json')
 copy(ROOT/'.tirust-build/cayley-guarded-adaptive-physical-proposal-2/binding1.json','inputs/binding1.json')
 gram=ROOT/'.tirust-build/cayley-guarded-adaptive-build-proposal-3/sources/pole-development/crates/iregi-rust/src/direct/exec/gram.rs'
 copy(gram,'sources/gram.rs')
 oneloop=ROOT/'.tirust-build/cargo-home/git/checkouts/oneloopmaster-1e046f4c0b16c2f9/bd703d0/src/native/dilog.rs'
 copy(oneloop,'sources/oneloop-dilog.rs')
 copy(ROOT/'.tirust-build/cayley-guarded-adaptive-build-proposal-3/sources/adaptive-qp-rescue-prototype3/Cargo.lock','sources/bridge-Cargo.lock')
 native=DSO.read_bytes();assert sha(native)==EXPECTED and len(native)==69162336
 header=struct.unpack_from('<16sHHIQQQIHHHHHH',native)
 assert header[0][:7]==b'\x7fELF\x02\x01\x01' and header[1:4]==(3,62,1)
 phoff,phsize,phnum=header[5],header[9],header[10]
 loads=[]
 for n in range(phnum):
  p=struct.unpack_from('<IIQQQQQQ',native,phoff+n*phsize)
  if p[0]==1:loads.append({'type':'PT_LOAD','flags':p[1],'offset':p[2],'vaddr':p[3],'filesz':p[5],'memsz':p[6],'align':p[7]})
 def file_offset(address):
  matches=[p['offset']+address-p['vaddr'] for p in loads if p['vaddr']<=address<p['vaddr']+p['filesz']]
  assert len(matches)==1;return matches[0]
 commands=[]
 def tool(label,argv):
  binary=Path(shutil.which(argv[0])).resolve()
  completed=subprocess.run([str(binary),*argv[1:]],cwd=ROOT,capture_output=True,timeout=30)
  assert completed.returncode==0,(label,completed.returncode,completed.stderr.decode())
  output=completed.stdout.decode('utf8')
  commands.append({'label':label,'argv':[str(binary),*argv[1:]],'tool_sha256':sha(binary.read_bytes()),'returncode':completed.returncode,'process_reaped':True,'role':'Read-only ELF symbolization/disassembly; no artifact loading/execution'})
  return output
 symbols=tool('symbols',['nm','-an','-S','-C',str(DSO)])
 wanted=('iregi_rust::direct::exec::gram::gram_indicator::<f64>','iregi_rust::direct::exec::gram::cayley_cancellation::<f64>','oneloop::native::dilog::series::<numerica::domains::float::multiprecision::Float>')
 selected=[]
 for line in symbols.splitlines():
  match=re.match(r'([0-9a-f]+) ([0-9a-f]+) ([A-Za-z]) (.*)',line)
  if match and match[4] in wanted:selected.append({'start':int(match[1],16),'bytes':int(match[2],16),'type':match[3],'name':match[4]})
 assert len(selected)==3
 (OUT/'symbols.txt').write_text('\n'.join(line for line in symbols.splitlines() if any(line.endswith(name) for name in wanted))+'\n')
 for row in selected:
  label='cayley' if 'cayley_cancellation' in row['name'] else 'gram-indicator' if 'gram_indicator' in row['name'] else 'oneloop-series'
  (OUT/(label+'.disassembly.txt')).write_text(tool(label,['objdump','-d','-C','--start-address='+hex(row['start']),'--stop-address='+hex(row['start']+row['bytes']),str(DSO)]))
 findings=[]
 for case in ('checked-trapped','forced-qp-trapped'):
  stdout=copy(RUN/case/'stdout.log','inputs/'+case+'.stdout.log').decode('utf8')
  maps=copy(RUN/case/'native-maps.txt','inputs/'+case+'.maps.txt').decode('utf8')
  controls=copy(RUN/case/'checked-control.log','inputs/'+case+'.control.log').decode('utf8')
  worker=runtime['workers'][case]
  assert worker['exit_code']==-8 and worker['progress']['process_reaped']
  assert not (RUN/case/'mass-observer.hex').exists()
  pc=int(re.search(r'^#3\s+(0x[0-9a-f]+)',stdout,re.M)[1],16)
  matched=[]
  for line in maps.splitlines():
   fields=line.split(None,5)
   if len(fields)==6:
    start,end=[int(v,16) for v in fields[0].split('-')]
    if start<=pc<end:matched.append({'start':start,'end':end,'permissions':fields[1],'file_offset':int(fields[2],16),'path':fields[5],'literal_line':line})
  assert len(matched)==1;mapping=matched[0]
  runtime_key=mapping['path'];assert runtime['runtime_sha256_after'][runtime_key]==EXPECTED
  compatible=[p for p in loads if p['offset']<=mapping['file_offset']<p['offset']+p['filesz'] and p['flags']&1]
  assert len(compatible)==1;segment=compatible[0]
  bias=mapping['start']-(segment['vaddr']+mapping['file_offset']-segment['offset'])
  relative=pc-bias
  owner=[s for s in selected if s['start']<=relative<s['start']+s['bytes']]
  assert len(owner)==1
  offset=file_offset(relative);instruction=native[offset:offset+4].hex()
  symbolization=tool(case,['addr2line','-f','-C','-e',str(DSO),hex(relative)])
  (OUT/(case+'.addr2line.txt')).write_text(symbolization)
  assert owner[0]['name'] in symbolization
  assert 'CALL_ENTER WARMUP 2 2' in controls and 'CALL_RETURN WARMUP 2 2' not in controls
  findings.append({'case':case,'signal':'SIGFPE','exit_code':-8,'backtrace_frame':3,'reported_PC':hex(pc),'mapped_file':mapping,'ELF_load_bias':hex(bias),'ELF_address':hex(relative),'ELF_file_offset':hex(offset),'instruction_hex':instruction,'instruction':'divsd %xmm1,%xmm0' if case=='checked-trapped' else 'divsd %xmm0,%xmm1','symbol':owner[0],'symbol_offset':hex(relative-owner[0]['start']),'stage':'WARMUP point2 before return; observed mass journal absent','not_in_cayley_cancellation_symbol':True,'actual_register_operands_or_exception_subtype_available':False})
 assert findings[0]['ELF_address']=='0x11fcecb' and findings[0]['instruction_hex']=='f20f5ec1'
 assert findings[1]['ELF_address']=='0x2523cc3' and findings[1]['instruction_hex']=='f20f5ec8'
 constant_address=0x380f608;constant=struct.unpack_from('<d',native,file_offset(constant_address))[0]
 save('analysis.json',{'schema':'guarded-physical-static-trap-localization-v1','scope':'Closed stack/map evidence plus literal ELF static decoding, without a numerical rerun or debugger','library_sha256':EXPECTED,'library_bytes':len(native),'native_ELF_payload_retained':False,'native_artifact_executed_or_loaded':False,'function_symbols':selected,'cases':findings,'oneloop_ratio_numerator':{'ELF_address':hex(constant_address),'binary64_value':constant,'interpretation':'Literal MOVSD constant feeding the first DIVSD; source expression4*pi*pi/modulus'},'source_rationale':{'checked':'gram_indicator w=largest/sqrt(cnorm2(delta).to_f64()) at gram.rs:724. This is a distinct unguarded normalization from cayley_cancellation below.','forced_qp':'oneloop series calls series_terms(bits,modulus) when modulus!=0; series_terms computes4*pi*pi/modulus at oneloop-dilog.rs:74. The helper is inlined into the MPFR series symbol, but this instruction is binary64.','limits':'No register/MXCSR/siginfo capture or core dump is available; zero operands, overflow and exact exception subtype are not established. Source association is inferred from symbol plus instruction/constant and retained source; optimized ELF has no source line table.','oneloop_revision':'bd703d09a78f7849a6b86a07096e6f222fc01193'},'copies':copies,'read_only_tool_commands':commands})
 save('elf-layout.json',{'library_sha256':EXPECTED,'PT_LOAD':loads})
 shutil.copyfile(Path(__file__),OUT/'localize.py')
 print(json.dumps({'cases':findings,'oneloop_ratio_numerator':constant,'native_executed_or_loaded':False},indent=2))
if __name__=='__main__':main()
