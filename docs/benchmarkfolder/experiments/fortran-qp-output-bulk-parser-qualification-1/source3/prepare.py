"""Pure source3 pins, preserving source2's failed BUILD without executing tools."""
import hashlib
import json
from pathlib import Path

ROOT=Path.cwd();HERE=Path(__file__).resolve().parent
PARENT=ROOT/'.tirust-build/fortran-qp-output-bulk-unit-controller-2'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert digest(PARENT/'source-index.json')=='ad63b2e1128c70085fb282b1a74e1b84aa282c7e5be95514a79a558845817fe4'
    pin=json.loads((PARENT/'input-pins.json').read_text())
    for name,checksum in pin['files'].items():
        p=Path(name) if Path(name).is_absolute() else ROOT/name
        assert digest(p)==checksum,name
    profile=pin.pop('expected_runtime')
    pin['profile_runtime_reference_only']=profile
    pin['unit_runtime_policy']='Pure DT_NEEDED/RUNPATH graph from newly linked ELF, constrained to prebound files; exact subsequent actual PT_INTERP listing required'
    pin['schema']='madloop-tirust-bulk-unit-inputs-v2'
    ancestor=json.loads((HERE/'failure-ancestry.json').read_text())
    assert ancestor['terminal']['sha256']=='874557356d762b3b51fa5bdb7f33df23375d14a46f6ea38f4a6b991e972de64c'
    for name,identity in ancestor['files'].items():
        path=ROOT/ancestor['failed_work']/name
        assert path.stat().st_size==identity['bytes'] and digest(path)==identity['sha256'],name
        pin['files'][str(path.relative_to(ROOT))]=identity['sha256']
    pin['files'][str((PARENT/'source-index.json').relative_to(ROOT))]=digest(PARENT/'source-index.json')
    target=HERE/'input-pins.json'
    text=json.dumps(pin,sort_keys=True,indent=2)+'\n'
    if target.exists():assert target.read_text()==text
    else:target.write_text(text)
    print(json.dumps({'input_pins_sha256':digest(target),'protected_files':len(pin['files']),
                      'compiler_invocations':0,'loader_queries':0,'native_executions':0},sort_keys=True))


if __name__=='__main__':main()
