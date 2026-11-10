"""Offline PC translation/symbolization; never execute an inspected artifact."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
RUN=ROOT/'.tirust-build/adaptive-qp-checked-trap-proposal-1/run1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (HERE/'symbolization.json').exists()
    tools={name:Path(shutil.which(name)) for name in ('addr2line','objdump')}
    record={'scope':'Read-only offline artifact symbolization; no artifact loading, rerun or debugger attachment',
            'tools':{name:{'path':str(path),'sha256':sha(path)} for name,path in tools.items()},'arms':{}}
    for arm in ('original','adaptive'):
        directory=RUN/(arm+'-trapped');maps=(directory/'native-maps.txt').read_text().splitlines()
        text=(directory/'stdout.log').read_text();frames=[]
        for number,pcword in re.findall(r'^#(\d+)\s+0x([0-9a-f]+)',text,re.M):
            if not 3<=int(number)<=16:continue
            pc=int(pcword,16)
            row=next(line.split(None,5) for line in maps if int(line.split()[0].split('-')[0],16)<=pc<int(line.split()[0].split('-')[1],16))
            path=Path(row[5]);data=path.read_bytes();offset=pc-int(row[0].split('-')[0],16)+int(row[2],16)
            start=struct.unpack_from('<Q',data,32)[0];size,count=struct.unpack_from('<HH',data,54)
            loads=[struct.unpack_from('<IIQQQQQQ',data,start+i*size) for i in range(count)]
            load=next(v for v in loads if v[0]==1 and v[2]<=offset<v[2]+v[5]);virtual=load[3]+offset-load[2]
            argv=[str(tools['addr2line']),'-f','-C','-e',str(path),hex(virtual)]
            result=subprocess.run(argv,capture_output=True,text=True,check=True,timeout=30)
            frame={'number':int(number),'pc':hex(pc),'mapping_line':' '.join(row),'object':str(path),
                   'object_sha256':hashlib.sha256(data).hexdigest(),'file_offset':offset,'elf_virtual_address':virtual,
                   'bytes_at_pc':data[offset:offset+16].hex(),'addr2line_argv':argv,'addr2line_stdout':result.stdout,'addr2line_stderr':result.stderr}
            if number=='3':
                argv=[str(tools['objdump']),'-d','--start-address='+hex(virtual-16),'--stop-address='+hex(virtual+24),str(path)]
                result=subprocess.run(argv,capture_output=True,text=True,check=True,timeout=30)
                frame.update(objdump_argv=argv,objdump_stdout=result.stdout,objdump_stderr=result.stderr)
            frames.append(frame)
        record['arms'][arm]={'maps_sha256':sha(directory/'native-maps.txt'),'stdout_sha256':sha(directory/'stdout.log'),'frames':frames}
    source=ROOT.parent/'references/TIRust/crates/iregi-rust/src/direct/exec/gram.rs'
    (HERE/'reference-gram.rs').write_bytes(source.read_bytes())
    record['reference_source']={'origin':str(source),'sha256':sha(source),
                                'scope':'Pinned reference source witness; no vendor/reference edits'}
    record['instruction_at_both_fault_PCs']='DIVSD %xmm1,%xmm0; bytesf20f5ec1'
    record['localized_path']='Shared TIRust DP f64 Cayley-conditioning heuristic, before observed adaptive QP'
    record['exact_hardware_exception_identified']=False
    record['limitation']='No siginfo/register values were captured. Divide instruction is proven; division-by-zero is supported by masked named flags but is an inference.'
    (HERE/'symbolization.json').write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'offline_symbolization_complete':True,'artifact_executed':False,'arms':2,'frames':sum(len(a['frames']) for a in record['arms'].values())}))


if __name__=='__main__':main()
