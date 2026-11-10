import gzip,hashlib,json
from pathlib import Path
h=Path(__file__).resolve().parent;m=json.loads((h/'retention.json').read_bytes());raws={}
for n,r in m['closed_files'].items():
 b=(h/r['stored']).read_bytes();b=gzip.decompress(b) if r['gzip'] else b
 assert len(b)==r['size'] and hashlib.sha256(b).hexdigest()==r['sha256'];raws[n]=b
for r in m['additional_closed_diagnostics'].values():assert hashlib.sha256((h/r['stored']).read_bytes()).hexdigest()==r['sha256']
s=json.loads(raws['stage-prepare.json']);assert s['status']=='failed' and s['native_execution'] is False
e=json.loads(raws['execution/generate-baseline-qcd_n2_gg_ttx.json']);assert e['returncode']==0 and not e['cleanup']['survivors'] and e['cleanup_exception'] is None
log=raws['execution/generate-baseline-qcd_n2_gg_ttx.txt'];assert b'os.mkdir(self.dir_path)' in log and b'FileNotFoundError' in log and b'interrupted in sub-command' in log and b'Process generated in' in log
print(json.dumps({'passed':True,'closed_files':len(raws),'CLI_dispatch_succeeded':True,'native_execution':False}))
