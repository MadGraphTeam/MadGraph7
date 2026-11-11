"""Read-only source/scientific/compiled/validation pins for RUN-only glue."""
import hashlib,json
from pathlib import Path
ROOT=Path.cwd();HERE=Path(__file__).resolve().parent
SCI=ROOT/'.tirust-build/rank6-ir-triangle-controller-1'
OWNER=ROOT/'.tirust-build/rank6-ir-triangle-validation-controller-1'
VALIDATED=ROOT/'.tirust-build/rank6-ir-triangle-validation-work-1/validate'
COMPILED=ROOT/'.tirust-build/rank6-ir-triangle-work-1/build'
def identity(p):
    b=Path(p).read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def main():
    require=lambda b:None if b else (_ for _ in ()).throw(ValueError('Unexpected frozen ancestor'))
    require(identity(OWNER/'source-index.json')['sha256']=='73f311c041c176fe80c00b53f51935d8283521672717247ee73d26c855ee0640')
    require(identity(VALIDATED/'terminal.json')['sha256']=='9c6d48c1db7e725db8a7579d9d6dc1ea623c5f22f4603232ebf3540042f4715c')
    old=json.loads((OWNER/'input-pins.json').read_text());files=dict(old['protected_files'])
    index=json.loads((OWNER/'source-index.json').read_text())
    for name,w in index['files'].items():files[str(OWNER/name)]=w
    files[str(OWNER/'source-index.json')]=identity(OWNER/'source-index.json')
    validated={}
    for p in sorted(VALIDATED.rglob('*')):
        if p.is_file():validated[str(p.relative_to(VALIDATED))]=identity(p);files[str(p)]=validated[str(p.relative_to(VALIDATED))]
    for p,w in files.items():require(identity(p)==w)
    value={'schema':'rank6-RUN-only-input-pins-v1','protected_files':files,'validated_files':validated,'failed_build_files':old['failed_build_files'],
      'scientific_source_sha256':'4f496ebf3071e8d405e1e2e72b4bcff3ab7b210ef2338740cdbe314b0cfcb2b0','validation_source_sha256':'73f311c041c176fe80c00b53f51935d8283521672717247ee73d26c855ee0640','validation_terminal_sha256':'9c6d48c1db7e725db8a7579d9d6dc1ea623c5f22f4603232ebf3540042f4715c','native_tensor_calls':0,'compiler_calls':0,'loader_queries':0}
    with (HERE/'input-pins.json').open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'protected_files':len(files),'validated_files':len(validated),'input_pins':identity(HERE/'input-pins.json')}))
if __name__=='__main__':main()
