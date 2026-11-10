import gzip,hashlib,json
from pathlib import Path
h=Path(__file__).resolve().parent;m=json.loads((h/'retention.json').read_bytes());raws={}
for n,r in m['closed_files'].items():
 b=(h/r['stored']).read_bytes();b=gzip.decompress(b) if r['gzip'] else b
 assert len(b)==r['size'] and hashlib.sha256(b).hexdigest()==r['sha256'];raws[n]=b
for r in m['additional_closed_diagnostics'].values():assert hashlib.sha256((h/r['stored']).read_bytes()).hexdigest()==r['sha256']
s=json.loads(raws['stage-prepare.json']);assert s['status']=='failed' and not s['native_execution']
log=raws['execution/generate-baseline-qcd_n2_gg_ttx.txt'];assert b'Compiling CutTools' in log and b'Cannot open module file' in log and b'Error detected' in log
a=json.loads((h/'prerequisite-audit.json').read_bytes());assert a['products']['baseline']['CT_directory_is_symlink'] and not any(a['products']['baseline']['CT_required_copied'].values()) and all(a['products']['candidate']['CT_required_copied'].values())
assert a['CT_compiler_version']=='15.2.0'
print(json.dumps({'passed':True,'closed_files':len(raws),'private_dependency_compilation_failed':True,'MadLoop_amplitudes':0}))
