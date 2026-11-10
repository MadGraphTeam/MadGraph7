"""Reuse the frozen two-command CMS supervisor for a separate common fixture.

Only source verification is admitted initially. Build/run still require root.
"""
import hashlib
import json
from pathlib import Path
import runpy
import sys

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
ROOT=PARENT.parents[4]
CMS=PARENT/'cms-width1/controller.py'
CMS_SHA='fcaa871f6c5b6bc880b0c9c315d151ac27c04e63751d46e7309065e2d32aeeb4'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def module():
    assert sha(CMS)==CMS_SHA
    original=runpy.run_path(str(CMS))
    source=CMS.read_text()
    changes={
        "('point.dat','nominal/param_card.dat','nominal/ident_card.dat')":"('nominal/param_card.dat','nominal/ident_card.dat')",
        'two captures and15 private CMS tensor rows only; no timing/physical-amplitude qualification':
        'zero physical captures and30 independent common-input tensor rows; no timing/physical-amplitude qualification'}
    for before,after in changes.items():
        assert source.count(before)==1;source=source.replace(before,after)
    ns={'__name__':'common_input_owned_controller','__file__':str(CMS)}
    exec(compile(source,str(CMS),'exec'),ns)
    work=ROOT/'.tirust-build/generated-rank6-poles-1/common-input1'
    ns.update(HERE=HERE,WORK=work,BUILD=work/'build1',RUN=work/'run1',INDEX=HERE/'source-index.json')
    def verify():
        assert Path.cwd()==ROOT
        index=json.loads((HERE/'source-index.json').read_text())
        for name,digest in index['files'].items():assert sha(HERE/name)==digest,name
        for name,digest in index['dependencies'].items():assert sha(ROOT/name)==digest,name
        runpy.run_path(str(HERE/'verify_fixture.py'))['verify']()
        return original['verify']()
    ns['verify']=verify
    return ns

if __name__=='__main__':
    assert len(sys.argv)<=2
    stage=sys.argv[1] if len(sys.argv)==2 else 'verify'
    assert stage in ('verify','build','run')
    module()[stage]()
