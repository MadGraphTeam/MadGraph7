#!/usr/bin/env python3
"""Record static allocator call sites, without executing any numerical code."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE=Path(__file__).resolve().parent
meta=json.loads((HERE/'adapter-scratch-build/build-metadata.json').read_text())
rows=[]
for variant,output in [('original',meta['source']),('scratch',meta['output'])]:
    process=Path(output)/'SubProcesses/P0_uux_z'
    for filename,names in [
        ('TIRUST_interface.o',['ml5_0_tirust_tensor_diagnostic_','ml5_0_mp_tirust_tensor_diagnostic_']),
        ('TIR_interface.o',['ml5_0_tirloop_','ml5_0_switch_order_','ml5_0_mp_switch_order_'])]:
        obj=process/filename
        disassembly=subprocess.check_output(['objdump','-dr',str(obj)],text=True)
        for name in names:
            block=disassembly.split('<'+name+'>:',1)[1]
            block=re.split(r'\n[0-9a-f]+ <',block)[0]
            calls=[line.strip() for line in block.splitlines() if 'R_X86_64_PLT32' in line]
            row={'variant':variant,'object':str(obj),'object_sha256':hashlib.sha256(obj.read_bytes()).hexdigest(),
                 'routine':name,'relocated_calls':calls,
                 'malloc_calls':sum(bool(re.search(r'\tmalloc-',line)) for line in calls),
                 'free_calls':sum(bool(re.search(r'\tfree-',line)) for line in calls),
                 'pack_calls':[line for line in calls if '_pack' in line or '_unpack' in line]}
            rows.append(row)
            if filename=='TIRUST_interface.o':
                expected=7 if variant=='original' else 0
                assert row['malloc_calls']==row['free_calls']==expected
                assert not row['pack_calls']
            print(variant,name,row['malloc_calls'],row['free_calls'])
result={'purpose':__doc__,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Static call sites, not executed allocation counts or timing estimates',
        'objects':rows}
(HERE/'adapter-scratch-build/allocation-sites.json').write_text(json.dumps(result,indent=2)+'\n')
