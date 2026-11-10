"""Retain text and hash native files from the one completed private attempt."""
import difflib
import gzip
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
ATTEMPT=ROOT/'.tirust-build/adaptive-qp-physical-replay-proposal-1/attempt1'
ORIGINAL=ROOT/'.tirust-build/qp-mass-general-implementation/combined-qualification/amplitude/runs/guarded-qp-9'
PROPOSAL=HERE.with_name('adaptive-qp-physical-replay-proposal-1')


def sha(data):return hashlib.sha256(data).hexdigest()


def save(path,value):
    assert not path.exists(),str(path)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def main():
    assert Path.cwd().resolve()==ROOT
    payload=HERE/'payloads';payload.mkdir(exist_ok=False)
    inventory={};natives={}
    def retain(key,path):
        data=path.read_bytes();assert not path.is_symlink()
        origin=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        record={'origin':origin,'sha256':sha(data),'bytes':len(data)}
        if data.startswith(b'\x7fELF'):
            record['retained_as']='hash_only_native_ELF';natives[key]=record;return
        data.decode('utf-8')
        name='payloads/'+sha(data)+'.gz';destination=HERE/name
        compressed=gzip.compress(data,compresslevel=9,mtime=0)
        if not destination.exists():destination.write_bytes(compressed)
        else:assert destination.read_bytes()==compressed
        record.update(archive=name,archive_sha256=sha(compressed));inventory[key]=record
    for key,base in [('attempt',ATTEMPT),('original',ORIGINAL),('proposal',PROPOSAL)]:
        for path in sorted(base.rglob('*')):
            if path.is_file():retain(key+'/'+str(path.relative_to(base)),path)
    for key,path in [
        ('frozen/journal-parser.py',HERE.parent/'ew-qp-mass-prototype/observer-native-design/analyze.py'),
        ('frozen/completion-validator.py',ORIGINAL.parents[1]/'analyze.py'),
        ('ancestry/mass-build.json',ORIGINAL.parents[2]/'build.json'),
        ('ancestry/mass-prepared.json',ORIGINAL.parents[2]/'prepared.json'),
        ('ancestry/mass-amplitude-prepared.json',ORIGINAL.parents[1]/'prepared.json'),
        ('ancestry/mass-evidence-index.json',HERE.parent/'ew-qp-mass-prototype/combined-qualification-1/evidence-index.json'),
        ('ancestry/release-fortran-evidence-index.json',HERE.parent/'adaptive-qp-release-fortran-qualification-v2/runtime1/evidence-index.json'),
        ('diagnosis-source/ew_mass_snapshot.f',HERE.parent/'ew-qp-mass-prototype/observer-native-design/ew_mass_snapshot.f'),
        ('diagnosis-source/ew_mass_observer.f90',HERE.parent/'ew-qp-mass-prototype/observer-native-design/ew_mass_observer.f90'),
        ('diagnosis-source/check_observed.f',HERE.parent/'ew-qp-mass-prototype/observer-native-design/attempt2/check_observed.f'),
        ('diagnosis-source/adapters.f',ORIGINAL.parents[1]/'src/adapters.f'),
        ('diagnosis-source/loop_matrix_standalone.inc',ORIGINAL.parents[1]/'templates/loop_matrix_standalone.inc'),
    ]:retain(key,path)
    # The interpreter is not copied into the attempt, so retain its identity too.
    execution=json.loads((ATTEMPT/'execution.json').read_text())
    for path in execution['expected_runtime_sha256']:
        if not Path(path).is_relative_to(ATTEMPT):retain('external-runtime/'+Path(path).name,Path(path))
    save(HERE/'payload-inventory.json',inventory);save(HERE/'native-identities.json',natives)
    patch=''.join(difflib.unified_diff((PROPOSAL/'analyze.py').read_text().splitlines(True),
                                    (HERE/'offline-successor/analyze.py').read_text().splitlines(True),
                                    fromfile='frozen-proposal/analyze.py',tofile='offline-successor/analyze.py'))
    assert patch
    (HERE/'classification-only.patch').write_text(patch)
    save(HERE/'capture-receipt.json',{'native_executed_by_capture':False,'loader_or_build_executed':False,
                                    'text_records':len(inventory),'unique_text_payloads':len(list(payload.glob('*.gz'))),
                                    'native_files_hash_only':len(natives),
                                    'original_failed_execution_sha256':sha((ATTEMPT/'execution.json').read_bytes()),
                                    'original_failed_analysis_sha256':sha((ATTEMPT/'analysis.json').read_bytes()),
                                    'corrected_offline_analysis_sha256':sha((HERE/'corrected-analysis.json').read_bytes())})
    print(json.dumps({'text_records':len(inventory),'native_hash_records':len(natives),'native_launched':False}))


if __name__=='__main__':main()
