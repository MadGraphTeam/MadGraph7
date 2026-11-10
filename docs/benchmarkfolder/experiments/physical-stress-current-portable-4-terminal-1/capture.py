"""Append-once capture after root-confirmed native4 terminal; pure filesystem work."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
HERE = Path(__file__).resolve().parent
REPORT = HERE.with_name('physical-stress-current-qualification-4')
PARENT = HERE.with_name('physical-stress-current-portable-4-preflight-1')
POLICY = HERE.with_name('physical-stress-current-terminal-retention-proposal-1')

def sha(raw): return hashlib.sha256(raw).hexdigest()
def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute:
        print(json.dumps({'capture':False,'native_execution':False}));return
    assert Path.cwd()==ROOT and not (HERE/'evidence-index.json').exists()
    source=importlib.util.spec_from_file_location('terminal_portable_verifier',HERE/'verify_evidence.py')
    verifier=importlib.util.module_from_spec(source);source.loader.exec_module(verifier)
    parent=verifier.Packet(lambda name:(PARENT/name).read_bytes())
    assert sha(parent.index_raw)==verifier.PARENT_INDEX
    stage=json.loads((REPORT/'stage-run.json').read_bytes())
    assert stage['status']=='failed' and stage['exception']=="TimeoutError('Lock wait expired before execution')"
    assert not any(p.is_symlink() for p in REPORT.rglob('*'))
    original={str(p.relative_to(REPORT)):{'sha256':file_sha(p),'size':p.stat().st_size} for p in sorted(REPORT.rglob('*')) if p.is_file()}
    directories=['.']+sorted(str(p.relative_to(REPORT)) for p in REPORT.rglob('*') if p.is_dir())
    index={'schema_version':1,'files':{},'compressed_text':{},'dependencies':{'preparation':{'sibling':PARENT.name,
           'index_sha256':verifier.PARENT_INDEX,'verifier_sha256':verifier.PARENT_VERIFIER}}}
    for name in ('capture.py','verify_evidence.py','README.md'):
        raw=(HERE/name).read_bytes();index['files'][name]={'kind':'file','sha256':sha(raw),'size':len(raw)}
    def append(name,raw):
        raw.decode('utf-8');assert b'\0' not in raw
        if len(raw)>65536 and Path(name).suffix!='.py':
            logical=name;name='compressed/'+name+'.gz'
            index['compressed_text'][logical]={'path':name,'raw_bytes':len(raw),'raw_sha256':sha(raw)}
            raw=gzip.compress(raw,compresslevel=6,mtime=0)
        destination=HERE/name;destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.open('xb') as stream:stream.write(raw)
        index['files'][name]={'kind':'file','sha256':sha(raw),'size':len(raw)}
    def save(name,value):append(name,(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())
    old=parent.load('retention.json')['closed_files'];closed={}
    for name,item in original.items():
        if name in old:
            assert item==old[name];closed[name]={**item,'storage':'preparation'}
        else:
            raw=(REPORT/name).read_bytes();assert sha(raw)==item['sha256']
            append('closed/'+name,raw);closed[name]={**item,'storage':'own'}
    assert set(old)<=set(original)
    policy=json.loads((POLICY/'source-index.json').read_bytes())
    append('policy/source-index.json',(POLICY/'source-index.json').read_bytes())
    for name,item in policy['files'].items():
        raw=(POLICY/name).read_bytes();assert sha(raw)==item['sha256'] and len(raw)==item['size'];append('policy/'+name,raw)
    mapped={};workers={}
    for path in sorted((REPORT/'runs').glob('*/*/specification.json')):
        spec=json.loads(path.read_bytes());rid=spec['request']['case']+'/'+spec['request']['label'];cwd=Path(spec['cwd']);output=cwd.parents[1]
        scientific={name:file_sha(output/name) for name in spec['scientific_sources']}
        cards={name:file_sha(output/name) for name in spec['numerical_cards']}
        assert scientific==spec['scientific_sources'] and cards==spec['numerical_cards']
        assert file_sha(output/'Cards/param_card.dat')==spec['physical_card_sha256']
        journals={}
        for journal in sorted((cwd/'worker-raw').rglob('*')):
            if not journal.is_file():continue
            assert not journal.is_symlink();name=str(journal.relative_to(cwd));raw=journal.read_bytes()
            retained='journals/'+rid+'/'+name;append(retained,raw)
            journals[name]={'retained':retained,'sha256':sha(raw),'size':len(raw)}
        workers[rid]={'scientific_sources':scientific,'numerical_cards':cards,
                      'physical_card_sha256':file_sha(output/'Cards/param_card.dat'),'journals':journals}
        for mapfile in sorted(path.parent.glob('maps-*.txt')):
            for line in mapfile.read_text().splitlines():
                fields=line.split(maxsplit=5)
                if len(fields)!=6 or not fields[5].startswith('/'):continue
                spelling=fields[5];assert not spelling.endswith(' (deleted)');actual=Path(spelling)
                if spelling not in mapped:
                    stat=actual.stat()
                    with actual.open('rb') as stream:header=stream.read(64)
                    mapped[spelling]={'resolved':str(actual.resolve()),'sha256':file_sha(actual),'size':stat.st_size,
                                     'inode':stat.st_ino,'device':[os.major(stat.st_dev),os.minor(stat.st_dev)],'header64_hex':header.hex()}
                item=mapped[spelling]
                assert item['inode']==int(fields[4]) and item['device']==[int(v,16) for v in fields[3].split(':')]
                if item['header64_hex'].startswith('7f454c46'):assert spec['expected_runtime'][item['resolved']]==item['sha256']
    work=ROOT/'.tirust-build/physical-stress-current-4'
    worker_dirs=sorted(str(p.relative_to(work/'workers')) for p in (work/'workers').glob('*/*') if p.is_dir())
    save('postclosure-bindings.json',{'scope':'Read-only postclosure identity check, not a new numerical observation',
          'mapped_files':mapped,'workers':workers,'worker_root':str(work/'workers'),'worker_directories':worker_dirs})
    save('root-terminal-observation.json',{'kind':'root-reported-tool-terminal-observation','session_id':29078,'exit_code':1,
          'observed_utc':'2026-11-10T12:56:26Z','source':'Parent root reported actual tools terminal observation; no synthetic machine transcript.',
          'exception':stage['exception']})
    save('retention.json',{'preparation_index_sha256':verifier.PARENT_INDEX,'source_policy_index_sha256':verifier.POLICY_INDEX,
          'capture_sha256':file_sha(Path(__file__)),'original_closed_root':str(REPORT),'closed_files':closed,'closed_directories':directories,
          'original_files_unchanged':True,'capture_native_execution':False})
    assert original=={str(p.relative_to(REPORT)):{'sha256':file_sha(p),'size':p.stat().st_size} for p in sorted(REPORT.rglob('*')) if p.is_file()}
    assert directories==['.']+sorted(str(p.relative_to(REPORT)) for p in REPORT.rglob('*') if p.is_dir())
    def reader(name):
        if name=='evidence-index.json':return (json.dumps(index,sort_keys=True,indent=2)+'\n').encode()
        if name.startswith('preparation/'):return (PARENT/name[len('preparation/'):]).read_bytes()
        return (HERE/name).read_bytes()
    proof=verifier.verify(read_bytes=reader,derive_summary=True)
    save('qualification-summary.json',proof['summary'])
    verifier.verify(read_bytes=reader)
    with (HERE/'evidence-index.json').open('xb') as stream:stream.write((json.dumps(index,sort_keys=True,indent=2)+'\n').encode())
    proof=verifier.verify(HERE)
    print(json.dumps({'verified':proof['verified'],'index_sha256':proof['index_sha256'],
                      'preparation_index_sha256':proof['preparation_index_sha256'],
                      'state_counts':proof['summary']['state_counts'],'case_coverage':proof['summary']['case_coverage']},indent=2,sort_keys=True))


if __name__=='__main__':main()
