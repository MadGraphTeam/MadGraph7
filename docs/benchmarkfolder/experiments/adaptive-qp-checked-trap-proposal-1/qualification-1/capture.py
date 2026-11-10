"""Capture the root-confirmed terminal build/run text; native bytes hash-only."""
import gzip
import hashlib
import json
from pathlib import Path
import stat

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent;ROOT=HERE.parents[4]
WORK=ROOT/'.tirust-build/adaptive-qp-checked-trap-proposal-1'
SOURCE_HASH='32909d9a7b4b2102f35eeddc547d2f45c7560f903253b0571bf77b5628220a29'


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    assert Path.cwd().resolve()==ROOT and sha((SOURCE/'source-index.json').read_bytes())==SOURCE_HASH
    assert not (HERE/'payload-inventory.json').exists()
    source=json.loads((SOURCE/'source-index.json').read_text());prepared=json.loads((SOURCE/'prepared.json').read_text())
    build=json.loads((WORK/'build1/build.json').read_text());run=json.loads((WORK/'run1/execution.json').read_text())
    assert build['complete'] and run['execution_complete'] and len(run['workers'])==7
    assert all(row['progress']['process_reaped'] for row in run['workers'].values())
    # Preserve any content-addressed text from the recorded pure-capture draft;
    # exact byte equality below is mandatory before reusing a payload.
    payload=HERE/'payloads';payload.mkdir(exist_ok=True)
    text={};native={}
    def retain(key,path,*,hash_only=False):
        assert not path.is_symlink() or hash_only
        data=path.read_bytes();info={'origin':str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                                   'sha256':sha(data),'bytes':len(data),'mode':stat.S_IMODE(path.stat().st_mode)}
        if path.is_symlink():info['resolved_dependency_target']=str(path.resolve())
        try:decoded=data.decode('utf-8')
        except UnicodeDecodeError:decoded=None
        if hash_only or decoded is None or data.startswith((b'\x7fELF',b'!<arch>')):
            info['retained_as']='native_or_dependency_identity_only';native[key]=info;return
        archive='payloads/'+sha(data)+'.gz';compressed=gzip.compress(data,compresslevel=9,mtime=0)
        destination=HERE/archive
        if not destination.exists():destination.write_bytes(compressed)
        else:assert destination.read_bytes()==compressed
        info.update(archive=archive,archive_sha256=sha(compressed));text[key]=info
    for name in [*source['sources'],'source-index.json']:retain('source/'+name,SOURCE/name)
    for kind in ('build1','run1'):
        for path in sorted((WORK/kind).rglob('*')):
            if path.is_file():retain(kind+'/'+str(path.relative_to(WORK/kind)),path)
    # Explicit dependency identities preserve recorded provenance; the portable
    # verifier will not reopen these historical absolute paths.
    dependencies={**prepared['dependencies'],**build['actual_link_inputs']}
    for name,digest in dependencies.items():
        path=ROOT/name if not name.startswith('/') else Path(name);assert sha(path.read_bytes())==digest
        retain('dependency/'+name,path,hash_only=True)
    for name in ('adaptive-qp-physical-replay-qualification-1','adaptive-qp-physical-replay-independent-8-audit-1'):
        path=SOURCE.parent/name/'evidence-index.json';retain('ancestor/'+name+'/evidence-index.json',path)
    for arm in ('original','adaptive'):
        for path,digest in run['runtime_sha256_after'][arm].items():
            if not Path(path).is_relative_to(WORK/'run1'):
                assert sha(Path(path).read_bytes())==digest;retain('external-runtime/'+Path(path).name,Path(path),hash_only=True)
    (HERE/'payload-inventory.json').write_text(json.dumps(text,sort_keys=True,indent=2)+'\n')
    (HERE/'native-identities.json').write_text(json.dumps(native,sort_keys=True,indent=2)+'\n')
    receipt={'scope':'One root-confirmed closed build88244 and runtime39482; no native rerun',
             'original_source_index_sha256':SOURCE_HASH,'original_build_sha256':sha((WORK/'build1/build.json').read_bytes()),
             'original_run_sha256':sha((WORK/'run1/execution.json').read_bytes()),'text_records':len(text),
             'unique_payloads':len(list(payload.glob('*.gz'))),'native_identity_records':len(native),
             'artifact_executed_by_capture':False,'originals_modified':False}
    (HERE/'capture-receipt.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    print(json.dumps(receipt,sort_keys=True))


if __name__=='__main__':main()
