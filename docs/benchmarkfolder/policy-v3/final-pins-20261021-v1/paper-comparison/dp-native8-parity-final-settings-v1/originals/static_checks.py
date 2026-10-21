"""Ordinary Python-only ABI/source/parser checks; no compiler or engine."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path.cwd()
HERE = ROOT/'.benchmark-work/paper-dp-stability-20261021-v1/nrs8-qcd1-parity-final-settings-v1'
guard = (HERE/'guard.c').read_text()
header = (ROOT/'vendor/ninjars/ninjars.h').read_text()
tree = ast.parse((HERE/'run.py').read_text())
ast.parse((HERE/'ninjars_dp_stability_frozen_b0fa.py').read_text())
checks = []

def checked(name, condition):
    if not condition:
        raise ValueError(name)
    checks.append(name)

def macro(name):
    lines = guard.splitlines()
    for i,line in enumerate(lines):
        if line.startswith('#define '+name+' '):
            text = line[len('#define '+name+' '):]
            while text.rstrip().endswith('\\'):
                text = text.rstrip()[:-1]+' '+lines[i+1]
                i += 1
            return text
    raise ValueError('Missing macro '+name)

def arguments(text):
    return [re.sub(r'\s+', '', item) for item in text.split(',')]

for symbol in ('context_create','evaluate_dp','evaluate_dp_compensated','evaluate_mp','context_destroy'):
    original = re.search(r'int32_t ninjars_'+symbol+r'\((.*?)\);',header,re.S)[1]
    wrapped = re.search(r'int32_t __wrap_ninjars_'+symbol+r'\((.*?)\)\s*\{',guard,re.S)[1]
    if wrapped == 'DP_ARGUMENTS':
        wrapped = macro('DP_ARGUMENTS')
    checked('exact_c_header_signature_'+symbol,arguments(original)==arguments(wrapped))
dp_signature = re.search(r'int32_t ninjars_evaluate_dp\((.*?)\);',header,re.S)[1]
names = [re.search(r'(\w+)\s*$', item)[1] for item in dp_signature.split(',')]
checked('native_dp_all_arguments_forwarded_in_order',arguments(macro('DP_VALUES'))==names)
checked('create_forwarded_unchanged','__real_ninjars_context_create(version, output)' in guard)
checked('destroy_forwarded_unchanged','__real_ninjars_context_destroy(context)' in guard)
checked('native_statuses_returned_unchanged',guard.count('return status;')==3)
checked('no_process_pointer_logging','%p' not in guard and '(uintptr_t)' not in guard)
checked('forbidden_compensated_entry','return forbidden(2, "FORBIDDEN_COMPENSATED_DP_ENTRY");' in guard)
checked('forbidden_mp_entry','return forbidden(3, "FORBIDDEN_MP_ENTRY");' in guard)
checked('no_real_compensated_or_mp_forward','__real_ninjars_evaluate_mp' not in guard and '__real_ninjars_evaluate_dp_compensated' not in guard)
tools = next(node for node in tree.body if isinstance(node,ast.Assign) and
             any(isinstance(name,ast.Name) and name.id=='TOOLS' for name in node.targets))
checked('selector_only_8',ast.literal_eval(tools.value)==(8,))
checked('cpu12_fence',"os.sched_getaffinity(0) != {12}" in (HERE/'run.py').read_text())
checked('bounded_each_command_180s','proc.wait(timeout=180)' in (HERE/'run.py').read_text())
checked('no_qualification_calls',not any(isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
    and node.func.attr in ('qualify','qualify_campaign','build_manifest','summary','validate_manifest')
    for node in ast.walk(tree)))

fn = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='parse_guard')
ns = dict(re=re,Counter=Counter)
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(HERE/'run.py'),'exec'),ns)
parse_guard = ns['parse_guard']
fixtures = HERE/'synthetic-static-fixtures'
fixtures.mkdir(exist_ok=False)

def line(event,counts,status=None):
    text = event+(' status='+str(status) if status is not None else '')
    return text+''.join(' '+name+'='+str(value) for name,value in zip(
        ('create','dp','compensated','mp','destroy','nonnull'),counts))

valid = [line('GUARD_BEGIN',[0,0,0,0,0,0]),line('CREATE_RETURN',[1,0,0,0,0,0],0),
    line('NATIVE_DP_RETURN',[1,1,0,0,0,0],0),line('NATIVE_DP_RETURN',[1,2,0,0,0,0],1),
    line('NATIVE_DP_RETURN',[1,3,0,0,0,0],2),line('DESTROY_RETURN',[1,3,0,0,1,1],0),
    line('GUARD_FINAL',[1,3,0,0,1,1])]
path = fixtures/'valid-native-errors-retained.log'
path.write_text('\n'.join(valid)+'\n')
parsed = parse_guard(path)
checked('native_statuses_0_1_2_preserved',parsed['native_per_call_status_counts']=={'0':1,'1':1,'2':1}
        and parsed['native_error_returns_retained']==2)
invalid = {
    'mp_from_start':[valid[0],line('FORBIDDEN_MP_ENTRY',[0,0,0,1,0,0])],
    'compensated_from_start':[valid[0],line('FORBIDDEN_COMPENSATED_DP_ENTRY',[0,0,1,0,0,0])],
    'missing_final':valid[:-1],
    'missing_begin':valid[1:],
    'counter_mismatch':valid[:2]+[line('NATIVE_DP_RETURN',[1,9,0,0,0,0],0)]+valid[3:],
    'lifecycle_failure':valid[:-2]+[line('DESTROY_RETURN',[1,3,0,0,1,1],2),valid[-1]],
    'hidden_mp_count':valid[:-1]+[line('GUARD_FINAL',[1,3,0,1,1,1])],
}
for name,rows in invalid.items():
    path = fixtures/(name+'.log')
    path.write_text('\n'.join(rows)+'\n')
    try:
        parse_guard(path)
    except ValueError:
        checked('reject_'+name,True)
    else:
        raise ValueError('Accepted synthetic invalid guard '+name)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

proof = dict(status='passed-static-source-and-synthetic-parser-only',checks=checks,
    count=len(checks),numerical_calls=0,compiler_calls=0,build_calls=0,
    sources={str(p.relative_to(ROOT)):sha(p) for p in (HERE/'run.py',HERE/'guard.c',
        HERE/'ninjars_dp_stability_frozen_b0fa.py',Path(__file__).resolve(),
        ROOT/'vendor/ninjars/ninjars.h',ROOT/'tools/ninjars_dp_stability_driver.f')},
    fixtures={str(p.relative_to(ROOT)):sha(p) for p in sorted(fixtures.iterdir())},
    note='All guard statuses in these checks are synthetic; no actual native reduction has run')
with (HERE/'static-validation.json').open('x') as stream:
    json.dump(proof,stream,indent=2,allow_nan=False);stream.write('\n')
print(json.dumps(dict(status=proof['status'],checks=len(checks)),indent=2))
