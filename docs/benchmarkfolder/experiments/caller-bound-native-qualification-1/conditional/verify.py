import hashlib,json
from pathlib import Path
import analyze
p=Path(__file__).resolve().parent
d=json.loads((p/'index.json').read_text())
assert {x.name for x in p.iterdir()}==set(d['files'])|{'index.json'}
for name,v in d['files'].items():
 q=p/name;assert not q.is_symlink();b=q.read_bytes();assert v=={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
out=analyze.main();assert out==json.loads((p/'results.json').read_text())
assert out['counts']=={'class_1':240,'class_2':120,'conditional_certificates':60,'conditional_zero_vetoes':40,'ordered_pairs':360,'original_zero_branches':280}
assert all(not r['conditional_certificate'] for r in out['rows'] if r['pair_class']==1)
assert len([r for r in out['critical_group3_pair24'] if r['original_zero_branch'] and r['conditional_certificate']])==4
print(json.dumps({'passed':True,'ordered_pairs':360,'conditional_zero_vetoes':40,'four_causal_cases_conditionally_admitted':True,'native_executions':0,'environment_or_producer_proved_by_old_trace':False},sort_keys=True))
