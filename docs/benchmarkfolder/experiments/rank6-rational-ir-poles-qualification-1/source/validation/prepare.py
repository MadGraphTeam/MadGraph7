"""Freeze old failed BUILD/source inputs; read-only data preparation only."""
import hashlib,json
from pathlib import Path
ROOT=Path.cwd();HERE=Path(__file__).resolve().parent
ORIGINAL=ROOT/'.tirust-build/rank6-ir-triangle-controller-1'
FAILED=ROOT/'.tirust-build/rank6-ir-triangle-work-1/build'
def identity(p):
    b=Path(p).read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def main():
    index=json.loads((ORIGINAL/'source-index.json').read_text())
    assert identity(ORIGINAL/'source-index.json')['sha256']=='4f496ebf3071e8d405e1e2e72b4bcff3ab7b210ef2338740cdbe314b0cfcb2b0'
    assert identity(FAILED/'terminal.json')['sha256']=='e1d442dbc86c04606a5aba1ee5f83614b00c64d6ac9ef0a07558277fc5f2ad77'
    old=json.loads((ORIGINAL/'input-pins.json').read_text())
    files={str(ORIGINAL/name):w for name,w in index['files'].items()};files[str(ORIGINAL/'source-index.json')]=identity(ORIGINAL/'source-index.json')
    for p,h in old['files'].items():
        w=identity(p);assert w['sha256']==h,p;files[p]=w
    closed={}
    for p in sorted(FAILED.rglob('*')):
        if p.is_file():closed[str(p.relative_to(FAILED))]=identity(p);files[str(p)]=closed[str(p.relative_to(FAILED))]
    assert all(identity(p)==w for p,w in files.items())
    value={'schema':'rank6-validation-only-input-pins-v1','original_source_sha256':'4f496ebf3071e8d405e1e2e72b4bcff3ab7b210ef2338740cdbe314b0cfcb2b0','original_failed_terminal_sha256':'e1d442dbc86c04606a5aba1ee5f83614b00c64d6ac9ef0a07558277fc5f2ad77','protected_files':files,'failed_build_files':closed,'native_main_calls':0,'compiler_calls':0,'loader_queries':0}
    with (HERE/'input-pins.json').open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'protected_files':len(files),'failed_build_files':len(closed),'input_pins':identity(HERE/'input-pins.json')}))
if __name__=='__main__':main()
