"""Pure hash/corpus verification; no compiler or native call."""
import json
from pathlib import Path
import control,prepare
p=Path(__file__).resolve().parent
control.source_verified()
c,e,t=prepare.corpus()
assert json.loads(json.dumps(c))==json.loads((p/'cases.json').read_text())
assert e==json.loads((p/'expected.json').read_text()) and t==json.loads((p/'tokens.json').read_text())
a,b=prepare.render(c,t)
assert a==(p/'inputs.txt').read_text() and b==(p/'tokens.txt').read_text()
print(json.dumps({'passed':True,'cases':len(c),'base_route_modes':104,'synthetic_token_cases':len(t),'compiled':False,'native_executions':0},sort_keys=True))
