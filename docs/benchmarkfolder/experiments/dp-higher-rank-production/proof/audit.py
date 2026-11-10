"""Read-only replay of the live production source/consumer/table proof."""
from pathlib import Path
import argparse, hashlib, json, datetime, sys
HERE=Path(__file__).resolve().parent
REPO=next(path for path in HERE.parents if (path/'madgraph/loop/dp_terminal_proof.py').is_file())
sys.path.insert(0,str(REPO))
from madgraph.loop import dp_terminal_proof as proof
p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--expected',default=str(HERE/'expected-producers.json'));p.add_argument('--prefix',default='ML5_0_');p.add_argument('--output');a=p.parse_args()
directory=Path(a.directory).resolve()
expected_bytes=Path(a.expected).read_bytes()
expected={int(key):value for key,value in json.loads(expected_bytes).items()}
export,units,roots=proof._inventory(directory,None,())
expanded={unit:proof._expanded(unit,export,roots) for unit in units}
paths=set(units)|{row.origin for rows in expanded.values() for row in rows}
paths|={path for path in export.rglob('*') if path.is_file() and path.name in ('makefile','make_opts','makeinc.inc','aloha_file.inc')}
repo_paths=[REPO/name for name in (
    'madgraph/loop/dp_terminal_proof.py',
    'Template/loop_material/StandAlone/SubProcesses/makefile',
    'madgraph/models/template_files/fortran/makefile_standalone',
    'aloha/template_files/Makefile_F',
    'madgraph/iolibs/template_files/loop_optimized/polynomial.inc')]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def pins():return ({str(path.relative_to(export)):sha(path) for path in sorted(paths)},
                  {str(path.relative_to(REPO)):sha(path) for path in repo_paths})
before=pins()
passed=proof._proof(directory,a.prefix,expected,None,())
if pins()!=before or Path(a.expected).read_bytes()!=expected_bytes:raise SystemExit('Proof inputs changed during audit')
result=dict(schema_version=1,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=passed,
    process=str(directory),expected_producers=len(expected),compiled_units=len(units),
    source_include_build_files=len(paths),source_sha256=before[0],repository_input_sha256=before[1],
    expected_sha256=hashlib.sha256(expected_bytes).hexdigest(),audit_script_sha256=sha(Path(__file__)),unchanged_during_audit=True,
    scope='Production higher-rank consumer/storage/table proof; arithmetic, compiler, IEEE, and numerical qualification are separate')
serialized=json.dumps(result,indent=2,sort_keys=True)+'\n'
if a.output:Path(a.output).write_text(serialized)
print(json.dumps({key:value for key,value in result.items() if key not in ('source_sha256','repository_input_sha256')},sort_keys=True))
