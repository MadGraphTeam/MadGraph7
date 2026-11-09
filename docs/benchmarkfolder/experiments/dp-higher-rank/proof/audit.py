"""Read-only full-source proof invocation; writes its JSON only when requested."""
from pathlib import Path
from collections import Counter
import argparse, hashlib, importlib.util, json, datetime

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('higher_proof',HERE/'dp_terminal_proof.py')
proof=importlib.util.module_from_spec(spec);spec.loader.exec_module(proof)
p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--expected');p.add_argument('--output');a=p.parse_args()
directory=Path(a.directory).resolve()
expected=proof.collect_expected(directory)
if a.expected:
    supplied={int(k):v for k,v in json.loads(Path(a.expected).read_text()).items()}
    if supplied != expected: raise SystemExit('Expected producer map differs from source inventory')
export,units,roots=proof._inventory(directory,None,())
expanded={u:proof._expanded(u,export,roots) for u in units}
paths=set(units)|{row.origin for rows in expanded.values() for row in rows}
def pins(): return {str(path.relative_to(export)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}
before=pins()
passed=proof._proof(directory,'ML5_0_',expected,None,())
after=pins()
if before != after: raise SystemExit('Source closure changed during proof')
result=dict(schema_version=1,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=passed,
    process=str(directory),expected_producers=len(expected),ranks=dict(Counter('%dx%d'%(x['loop_mother_rank'],x['vertex_rank']) for x in expected.values())),
    compiled_units=len(units),source_and_include_files=len(paths),source_sha256=before,
    proof_sha256=hashlib.sha256((HERE/'dp_terminal_proof.py').read_bytes()).hexdigest(),
    unchanged_during_audit=True,scope='Private rank1/2/3 x rank1 four-component consumer/storage/table proof; arithmetic and IEEE kernel qualification are separate')
text=json.dumps(result,indent=2,sort_keys=True)+'\n'
if a.output: Path(a.output).write_text(text)
print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},sort_keys=True))
