#!/usr/bin/env python3
"""One-shot text-only retention of already closed profiling evidence; no native calls."""
from pathlib import Path
import gzip
import hashlib
import json
import shutil

ROOT=Path.cwd();HERE=Path(__file__).resolve().parent
BASE=ROOT/'docs/benchmarkfolder/experiments'

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def retain(source,target):
    raw=source.read_bytes();raw.decode('utf-8')
    if b'\0' in raw:raise ValueError('Binary/NUL payload refused: '+str(source))
    if target.exists():raise ValueError('Retention path already exists: '+str(target))
    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
def indexed(source,target,index_name):
    index=read(source/index_name)
    for name,item in index['files'].items():
        path=source/name
        if sha(path)!=item['sha256'] or path.stat().st_size!=item['bytes']:raise ValueError('Changed closed payload: '+str(path))
        retain(path,target/name)
    retain(source/index_name,target/index_name)

def package():
    science=BASE/'fortran-qp-transport-profile-1'
    controllers=[BASE/f'fortran-qp-transport-profile-controller-{i}' for i in (1,2,3)]
    indexed(science,HERE/'science','index.json')
    for i,controller in enumerate(controllers,1):indexed(controller,HERE/f'controller{i}','source-index.json')
    indexed(controllers[0]/'build',HERE/'failure-build1','failure-index.json')
    indexed(controllers[1]/'build',HERE/'build2','build-index.json')
    indexed(controllers[1]/'mode-0',HERE/'failure-mode0','failure-index.json')
    for name in ('map-retention.json','root-build-review.json'):retain(controllers[1]/'build'/name,HERE/'build2'/name)
    map_bytes=(controllers[1]/'build/private-link.map').read_bytes();map_bytes.decode('utf-8')
    if b'\0' in map_bytes:raise ValueError('Map is not text')
    (HERE/'build2/private-link.map.gz').write_bytes(gzip.compress(map_bytes,compresslevel=9,mtime=0))
    for mode in range(3):
        source=controllers[2]/f'mode-{mode}';terminal=read(source/'terminal.json')
        if terminal['status']!='passed' or terminal['returncode']!=0 or terminal['cleanup']['survivors']:raise ValueError('Unclosed mode')
        if sha(source/'result.json')!=terminal['result_sha256']:raise ValueError('Changed closed result')
        for path in sorted(source.rglob('*')):
            if path.is_file():retain(path,HERE/f'mode-{mode}'/path.relative_to(source))
    for name in ('tirust_benchmark_worker.py','tirust_benchmark_warmup.py'):retain(ROOT/'tools'/name,HERE/'protocol'/name)
    campaign=ROOT/'docs/benchmarkfolder/campaigns/entry-latch-cache-guard-1'
    lane=campaign/'experiments/numerical/full100-warmup-qcd_n3_uux_zgg-1/candidate-9'
    retain(lane/'raw/observation.json',HERE/'original-warmup-reference.json')
    retain(lane/'candidate-inputs.json',HERE/'original-candidate-inputs.json')
    for name in ('dat','json'):retain(campaign/f'points/qcd_n3_uux_zgg.{name}',HERE/f'inputs/points.{name}')
    retain(campaign/'manifest.json',HERE/'inputs/manifest.json')
    case=ROOT/'.benchmark-work/tirust/campaigns/entry-latch-cache-guard-1/outputs/candidate/qcd_n3_uux_zgg'
    for name in ('param_card.dat','MadLoopParams.dat'):retain(case/'Cards'/name,HERE/'inputs'/name)
    from analyze import analyze
    (HERE/'analysis.json').write_text(json.dumps(analyze(HERE),indent=2,sort_keys=True,allow_nan=False)+'\n')

if __name__=='__main__':package()
