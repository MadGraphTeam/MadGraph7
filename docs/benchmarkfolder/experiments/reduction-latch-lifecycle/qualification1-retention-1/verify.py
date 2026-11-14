#!/usr/bin/env python3
"""Pure capsule replay: four plain traces, exact input hashes, runtime receipts."""
from pathlib import Path
import gzip
import hashlib
import importlib.util
import json
import struct

HERE=Path(__file__).resolve().parent

def need(ok,message):
    if not ok:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read_json(path):return json.loads(path.read_text())
def verify(root=HERE):
    root=Path(root);index=read_json(root/'evidence-index.json')
    for name,entry in index['files'].items():
        path=root/name;need(path.is_file() and not path.is_symlink(),'Payload kind changed')
        data=path.read_bytes();need(len(data)==entry['bytes'] and sha(data)==entry['sha256'],'Payload changed: '+name)
    capture=read_json(root/'capture.json');logical={}
    for name,entry in capture['original_files'].items():
        stored=(root/name).read_bytes();data=gzip.decompress(stored) if entry['encoding']=='gzip' else stored
        need(len(data)==entry['logical_bytes'] and sha(data)==entry['logical_sha256'],'Logical original changed')
        logical[entry['original_path']]=data
    actual=root/'actual';terminal=read_json(actual/'terminal.json');build=read_json(actual/'build.json')
    need(terminal['passed'] is True and terminal['stage_within_cap'] is True,'Stage did not pass')
    need(terminal['elapsed_seconds_including_final_guards']<=600,'Stage cap exceeded')
    need(capture['actual_terminal_code']==0 and terminal['workers_completed']==4,'Closure shape')
    need(sha(gzip.decompress((actual/'link.map.gz').read_bytes()))==build['link_map_sha256'],'Link map differs')
    need(capture['excluded_binaries']['diagnostic']['sha256']==build['candidate_sha256']==terminal['candidate_executable_sha256'],'Candidate identity differs')
    for number in (1,2):
        receipt=read_json(actual/f'compile{number}.json');need(receipt['returncode']==0 and not receipt['native'],'Compile failed or misclassified')
    spec=importlib.util.spec_from_file_location('retained_latch_analyze',root/'analyze.py')
    analyzer=importlib.util.module_from_spec(spec);spec.loader.exec_module(analyzer)
    meta=read_json(root/'selected-inputs.json')
    points={p:b''.join(struct.pack('<d',float.fromhex(x)) for row in meta['points'][str(p)]['binary64_hex'] for x in row) for p in (1,93,39)}
    runtime=read_json(actual/'runtime-proof.json');need(runtime['passed'] and runtime['library_count']==15,'Runtime alias proof')
    expected={x['resolved_path']:x['sha256'] for x in runtime['libraries']}
    need(len(expected)==15 and capture['runtime_alias_nodes']=={x['soname']:x['resolved_path'] for x in runtime['libraries']},'Provider alias identity differs')
    parsed={};total=0
    for arm,mode in [('original',4),('candidate',4),('original',5),('candidate',5)]:
        folder=actual/f'{arm}-mode{mode}'
        worker=read_json(folder/'worker.json');loader=read_json(folder/'loader.json')
        need(worker['native'] and worker['returncode']==0 and worker['process_reaped'],'Worker did not close')
        need(loader['returncode']==0 and not loader['native'],'Separate loader trace failed')
        need(worker['seconds_cap']<=90 and worker['process_wall_seconds']<=90,'Worker cap exceeded')
        exe=build['original_executable'] if arm=='original' else build['candidate_executable']
        checksum=build['original_sha256'] if arm=='original' else build['candidate_sha256']
        need(worker['argv'][-3:]==['9',str(mode),'0'] and worker['argv'][-7]==exe and worker['argv'][-6]==checksum,'Worker identity differs')
        for name,hash_value in read_json(folder/'inputs.json').items():need(name in logical and sha(logical[name])==hash_value,'Input byte guard differs')
        for name,hash_value in read_json(folder/'outputs.json').items():need(sha((folder/name).read_bytes())==hash_value,'Output receipt differs')
        maps=read_json(folder/'mapped-runtime.json')
        need(maps['expected_runtime_sha256']==maps['mapped_files_sha256_after_exit']==expected,'Mapped provider receipts differ')
        need(maps['executable']==exe and set(maps['mapped_elf_paths'])==set(expected)|{exe},'Observed mapped closure differs')
        need(maps['cpu_allowed_list']=='12','CPU affinity differs')
        need(maps['maps_sha256']==sha((folder/'maps.txt').read_bytes()),'Map snapshot differs')
        text=(folder/'loader.log').read_text();need('not found' not in text,'Missing loader provider')
        import re
        names=set(re.findall(r'/[^\s()]+',text))
        # Original aliases may be spelled literally; each retained resolved alias must map exactly.
        aliases=capture['runtime_alias_nodes'];alias_dir=runtime['alias_directory']
        resolved={aliases[Path(x).name] if x.startswith(alias_dir+'/') else x for x in names}
        need(resolved==set(expected),'Loader closure differs')
        value=analyzer.parse_worker((folder/'trace.txt').read_text(),(folder/'worker.log').read_text(),9,mode,0,points)
        need(json.loads(json.dumps(value))==read_json(folder/'parsed.json'),'Retained parser result differs')
        need(value['public_calls']<=12,'Per-worker public bound')
        parsed[arm,mode]=value;total+=value['public_calls']
    need(total==terminal['public_calls']==20 and total<=48,'Public count differs')
    for mode in (4,5):need(parsed['original',mode]['public_raw']==parsed['candidate',mode]['public_raw'],'Public byte parity differs')
    return {'passed':True,'native_executed':False,'workers':4,'public_calls':total,'diagnostic_calls':12,'answer_order_columns':24,'byte_parity_modes':[4,5],'points':[1,93,39],'runtime_provider_receipts':15,'host_checked':False,'accuracy_or_performance_acceptance':False}
if __name__=='__main__':print(json.dumps(verify(),sort_keys=True,indent=2))
