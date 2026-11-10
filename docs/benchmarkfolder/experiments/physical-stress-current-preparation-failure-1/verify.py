import gzip,hashlib,json
from pathlib import Path
h=Path(__file__).resolve().parent
m=json.loads((h/'retention.json').read_bytes())
raws={}
for n,r in m['closed_files'].items():
 b=(h/r['stored']).read_bytes();b=gzip.decompress(b) if r['gzip'] else b
 assert len(b)==r['size'] and hashlib.sha256(b).hexdigest()==r['sha256'];raws[n]=b
s=json.loads(raws['stage-prepare.json']);assert s['status']=='failed' and s['native_execution'] is False
e=json.loads(raws['execution/generate-baseline-qcd_n2_gg_ttx.json']);assert e['returncode']==2 and not e['cleanup']['survivors'] and e['cleanup_exception'] is None
assert b"No such file or directory" in raws['execution/generate-baseline-qcd_n2_gg_ttx.txt']
for source,e in m['entrypoints'].items():
 j=json.loads(raws['materialized-'+source+'.json']);assert j['files'][e['entrypoint']]['sha256']==e['sha256']
 assert 'bin/mg5_aMC' not in j['files'];assert hashlib.sha256((h/e['stored']).read_bytes()).hexdigest()==e['sha256']
print(json.dumps({'passed':True,'closed_files':len(raws),'actual_CLI_code_executed':False,'native_execution':False}))
