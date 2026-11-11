"""Pure RUN-only ancestry, full retained-byte inventory and frozen-tool pins."""
import hashlib
import json
from pathlib import Path

ROOT=Path.cwd();HERE=Path(__file__).resolve().parent
PARENT=ROOT/'.tirust-build/fortran-qp-output-bulk-unit-controller-3'
BUILD=ROOT/'.tirust-build/fortran-qp-output-bulk-unit-work-3/build'
RUN=ROOT/'.tirust-build/fortran-qp-output-bulk-unit-work-3/run'
SOURCE_SHA='511225ea5bb4523cbf84012ea424a0710ef6f86b3ec6964ba8e8b41ea08a76d6'
BUILD_SHA='d4eb1386ce923a16d70666637ff3677091449b30a812b6ebcb90b421c0b3cf9f'
RUN_SHA='4e70b14200b037f0acf05723f05a29078f26e95bf2871b3be4fc97a4421a815f'
SCIENTIFIC=['provider.c','driver.f90','corpus.py','corpus.json','analyze.py','observer-mtirust.f90','forced-mtirust.f90']


def read(p):return json.loads(Path(p).read_text())
def identity(p):
    b=Path(p).read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def inventory(base):return {str(p.relative_to(base)):identity(p) for p in sorted(base.rglob('*')) if p.is_file()}
def write_new(p,value):
    text=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if p.exists():assert p.read_text()==text
    else:p.write_text(text)


def main():
    assert identity(PARENT/'source-index.json')['sha256']==SOURCE_SHA
    parent=read(PARENT/'source-index.json')
    for name,want in parent['files'].items():assert identity(PARENT/name)==want,name
    assert identity(BUILD/'terminal.json')['sha256']==BUILD_SHA
    assert identity(RUN/'terminal.json')['sha256']==RUN_SHA
    terminal=read(RUN/'terminal.json')
    assert terminal['status']=='failed' and terminal['returncode']==1 and not terminal['cleanup']['survivors']
    assert read(BUILD/'terminal.json')['status']=='passed'
    corpus=read(HERE/'corpus.json')['cases']
    scientific={}
    for name in SCIENTIFIC:
        scientific[name]=identity(PARENT/name)
        assert identity(HERE/name)==scientific[name],name
    prefix=[c['id'] for c in corpus if (RUN/c['id']).is_dir()]
    assert len(prefix)==88 and prefix[-1]=='options-f0-p3'
    endpoint=next(i for i,c in enumerate(corpus) if c['id']==prefix[-1])+1
    assert prefix==[c['id'] for c in corpus[:endpoint] if c['execution']!='lexical-only']
    lexical=[c['id'] for c in corpus if c['execution']=='lexical-only']
    pending=[c['id'] for c in corpus[endpoint:] if c['execution']!='lexical-only']
    assert len(lexical)==14 and len(pending)==47
    manifest={'schema':'madloop-tirust-bulk-run-only-continuation-inputs-v1',
              'source3_root':str(PARENT.relative_to(ROOT)),'source3_index':identity(PARENT/'source-index.json'),
              'source3_files':parent['files'],'build_root':str(BUILD.relative_to(ROOT)),
              'build_terminal':identity(BUILD/'terminal.json'),'build_files':inventory(BUILD),
              'failed_run_root':str(RUN.relative_to(ROOT)),'failed_run_terminal':identity(RUN/'terminal.json'),
              'failed_run_files':inventory(RUN),'scientific_files':scientific,
              'prefix_case_ids':prefix,'pending_case_ids':pending,'lexical_only_case_ids':lexical,
              'last_failed_case':'options-f0-p3','retained_original_comparisons':87,
              'retained_executable_invocations':264,'retained_public_mock_calls':1320,
              'native_rejection_correction':'finite=false,poles=none =>ERROR2/native message; exact source-defined Mode::new/guard/reject semantics',
              'old_failed_run_status_preserved':'failed','new_compiler_invocations':0,'new_native_executions':0}
    write_new(HERE/'continuation-inputs.json',manifest)
    pin=read(PARENT/'input-pins.json')
    for key,checksum in pin['files'].items():
        path=Path(key) if Path(key).is_absolute() else ROOT/key
        assert identity(path)['sha256']==checksum,key
    for base,names in [(PARENT,{**parent['files'],'source-index.json':identity(PARENT/'source-index.json')}),
                       (BUILD,manifest['build_files']),(RUN,manifest['failed_run_files'])]:
        for name,want in names.items():pin['files'][str((base/name).relative_to(ROOT))]=want['sha256']
    pin['schema']='madloop-tirust-bulk-unit-run-only-inputs-v1'
    pin['compiled_source_index_sha256']=SOURCE_SHA
    pin['reviewed_build_sha256']=BUILD_SHA;pin['reviewed_failed_run_sha256']=RUN_SHA
    pin['source3_failed_run_unchanged']=True
    write_new(HERE/'input-pins.json',pin)
    print(json.dumps({'protected_files':len(pin['files']),'continuation_inputs':identity(HERE/'continuation-inputs.json'),
                      'prefix':len(prefix),'pending':len(pending),'lexical':len(lexical),'new_compiler_invocations':0,'new_native_executions':0},sort_keys=True))


if __name__=='__main__':main()
