"""Adjudicate new CMS rows with retained controls, never merge raw attempts."""
from decimal import localcontext
import hashlib
import json
from pathlib import Path
import runpy

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
ROOT=PARENT.parents[4]
OLD_WORK=ROOT/'.tirust-build/generated-rank6-poles-1'
NEW_WORK=OLD_WORK/'cms-width1/run1'

def analyze(new_work=NEW_WORK):
    assert hashlib.sha256((PARENT/'analyze.py').read_bytes()).hexdigest()=='d3e6583162f3e3efe74bf933cda34e82eb0d8f606f752c475d15c6a3ac06eb4e'
    ns=runpy.run_path(str(PARENT/'analyze.py'))
    # Function globals belong to the returned namespace under runpy.
    ns=ns['analyze'].__globals__
    original_records=ns['parse_records'];original_observer=ns['parse_observer']
    rows=json.loads((HERE/'schedule.json').read_text())
    fixtures=json.loads((HERE/'fixture.json').read_text())
    fresh_captures,fresh_cases=original_records(new_work/'records.txt','ew',rows)
    fresh_native=original_observer(new_work/'observer.txt',[r['action'] for r in fresh_captures+fresh_cases])
    expected=fixtures['mass_squared_promoted_binary128']
    for row in fresh_cases:
        assert row['fixture']==105 and row['entry']['m'][2]==expected
        assert ns['binary128'](row['entry']['m'][2][2:])<0
    def records(path,model,schedule):
        captures,cases=original_records(path,model,schedule)
        if model=='ew':
            # Inputs, actual group identity, and all captured coefficient/status
            # rows must reproduce the original fresh process before replacement.
            assert captures==fresh_captures
            replacement={r['action']:r for r in fresh_cases}
            cases=[replacement.get(r['action'],r) for r in cases]
        return captures,cases
    def observer(path,expected_actions):
        values=original_observer(path,expected_actions)
        if path.parent.name=='ew':values.update(fresh_native)
        return values
    ns['parse_records']=records;ns['parse_observer']=observer
    with localcontext() as context:
        context.prec=100
        result=ns['analyze']([OLD_WORK/'ew',OLD_WORK/'qcd'])
    result['combined_analysis']=True
    result['new_native_scope']={'ew_captures':2,'cms_rows':15,'tirust_rows':12,'collier_rows':3,
                               'fixture':105,'expected_mass_squared_words':expected}
    result['scope']='Separate private positive-width CMS replay plus unchanged retained controls. Original zero-width failures remain archived; no physical-amplitude or timing qualification.'
    return result

if __name__=='__main__':print(json.dumps(analyze(),indent=2,sort_keys=True))
