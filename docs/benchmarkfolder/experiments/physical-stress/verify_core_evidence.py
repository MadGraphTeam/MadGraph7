from pathlib import Path
import collections
import datetime
import hashlib
import json
import math
import re

ROOT = Path.cwd()
REPORT = ROOT / 'docs/benchmarkfolder/experiments/physical-stress'
DEST = ROOT / '.tirust-build/physical-stress-audit/final.json'
issues, notes, identities, cache = [], [], {}, {}
counts = collections.Counter()

def load(path):
    return json.loads(Path(path).read_text())

def sha(path):
    path = Path(path)
    st = path.stat()
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns)
    if key not in cache:
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
        cache[key] = h.hexdigest()
    return cache[key]

def keep(path):
    identities[str(path.relative_to(ROOT))] = {'sha256': sha(path), 'bytes': path.stat().st_size}

def check(ok, text):
    if not ok:
        issues.append(text)

def card(path):
    lines = path.read_text().splitlines()
    return {line[1:].strip().lower(): next(v.strip() for v in lines[i+1:] if v.strip() and not v.lstrip().startswith('!'))
            for i, line in enumerate(lines) if line.startswith('#')}

def matches(value, expected):
    if isinstance(expected, bool):
        return value.lower() == ('.true.' if expected else '.false.')
    if isinstance(expected, (int, float)):
        return float(value.lower().replace('d', 'e')) == expected
    return value == str(expected)

config, plan = load(REPORT / 'cases.json'), load(REPORT / 'plan.json')
for case in plan['cases']:
    cid, n = case['id'], case['nfinal'] + 2
    pdir = REPORT / 'points' / cid
    meta = load(pdir / 'selected.json')
    data = (pdir / 'selected.dat').read_text().splitlines()
    check(data[0] == '2' and len(data) == 1 + 2*n, f'{cid}: selected DAT shape')
    check(sha(pdir / 'selected.dat') == meta['sha256'], f'{cid}: selected DAT SHA')
    for i, parent in enumerate(meta['parents']):
        src = ROOT / parent['source']; srcmeta = load(src.with_suffix('.json'))
        idx = parent['source_point_index']; start = 1 + (idx - 1)*n
        check(data[1+i*n:1+(i+1)*n] == src.read_text().splitlines()[start:start+n], f'{cid}: exact parent DAT rows {i}')
        check(meta['points'][i] == srcmeta['points'][idx-1], f'{cid}: exact parent point metadata {i}')
        check(sha(src) == parent['sha256'] == srcmeta['sha256'], f'{cid}: parent SHA {i}')
    counts['derived_point_streams_checked'] += 1

initial = load(ROOT / '.tirust-build/physical-stress-audit/evidence-check-initial.json')
for name, identity in initial['record_identities'].items():
    check(sha(ROOT / name) == identity['sha256'], f'Initial audit record changed: {name}')
counts['initial_record_identities_preserved'] = len(initial['record_identities'])

intervals, records, successful = [], {}, []
for section in ('runs', 'diagnostics'):
    for resultpath in sorted((REPORT / section).glob('*/*/result.json')):
        run = resultpath.parent; rec = load(resultpath); spec = load(run / 'specification.json')
        cid, source, backend = rec['case']['id'], rec['source'], rec['backend']
        isdiag = section == 'diagnostics'; trace = 'trace' in run.name
        group = 'trace' if trace else 'corrected_qp' if isdiag else 'primary' if rec['precision_request'] == 'auto' else 'initial_qp'
        counts[group + '_' + rec['status']] += 1; counts[group + '_total'] += 1
        records[(cid, run.name)] = (run, rec)
        for path in run.iterdir():
            if path.is_file() and path.suffix in ('.json', '.dat', '.f', '.py'):
                keep(path)
        for path in (run/'raw').glob('*'):
            if path.is_file(): keep(path)
        check(rec['points_sha256'] == sha(REPORT/'points'/cid/'selected.dat'), f'{run}: point binding')
        check(rec['performance_eligible'] is False, f'{run}: performance qualification')
        if isdiag or rec['precision_request'] == 'qp':
            check(rec['stability_checked'] is False, f'{run}: QP falsely checked')
        if 'runner_sha256' in rec:
            check(sha(run/'runner.py') == rec['runner_sha256'], f'{run}: runner binding')
        inp_path = run/f'{source}-inputs.json'
        if not inp_path.exists():
            check(rec['status'] not in ('complete', 'unchecked-qp-diagnostic'), f'{run}: successful without inputs')
            continue
        inp = load(inp_path); saved = card(run/'MadLoopParams.dat')
        original = load(REPORT/'inputs'/source/cid/'compilation.json')
        generation = load(REPORT/'inputs'/source/cid/'generation.json')
        check(inp['compilation'] == original, f'{run}: original compilation changed')
        check(inp['generation'] == generation, f'{run}: original generation changed')
        check(inp['points_sha256'] == rec['points_sha256'], f'{run}: input point binding')
        check(inp['param_card_sha256'] == generation['parameter_card_sha256'], f'{run}: physical card pin')
        check(sha(inp['resolved_parameter_card']) == inp['param_card_sha256'], f'{run}: resolved physical card pin')
        if (run/'param_card.dat').exists():
            check(sha(run/'param_card.dat') == inp['param_card_sha256'], f'{run}: retained physical card')
        if 'settings' in rec:
            check(rec['settings'] == inp['settings'], f'{run}: result/input settings mismatch')
        missing = []
        for key, value in inp['settings'].items():
            if key not in saved:
                missing.append(key)
            else:
                check(matches(saved[key], value), f'{run}: saved setting mismatch {key}')
        allowed = {'tirustcachemaxbytes','tirustcomputefinite','tirustglobalcache','useqpintegrandfortirust'} if source=='baseline' else set()
        check(set(missing) <= allowed, f'{run}: missing card settings {missing}')
        expected = dict(config['madloop_parameters'])
        if source == 'candidate':expected.update(config['candidate_parameters'])
        if rec['precision_request'] == 'qp':expected.update(CTModeRun=4, CTModeInit=1 if isdiag else 4)
        expected['MLReductionLib'] = str(backend)
        for key, value in expected.items():
            check(inp['settings'][key.lower()] == value, f'{run}: requested setting differs {key}')
        for groupname in ('libraries','archives','metadata'):
            for name, digest in inp['runtime'][groupname].items():
                check(sha(name) == digest, f'{run}: changed runtime input {name}')
        if not isdiag:
            intervals.append({'source':source,'case':cid,'run':run.name,
                'start':inp_path.stat().st_mtime,'end':resultpath.stat().st_mtime})
        else:
            proof=load(run/'diagnostic-compilation.json')
            check(proof == inp['diagnostic_compilation'], f'{run}: compilation proof mismatch')
            check(proof['numerical_objects_before'] == proof['numerical_objects_after'], f'{run}: numerical objects changed')
            check(proof['link_inputs_before'] == proof['link_inputs_after'] == original['link_inputs_before'] == original['link_inputs_after'], f'{run}: numerical libraries changed')
            check(set(proof['sources']) == set(original['sources']), f'{run}: source inventory changed')
            changed=[name for name in proof['sources'] if proof['sources'][name] != original['sources'].get(name)]
            check(len(changed)==1 and Path(changed[0]).name=='check_sa.f', f'{run}: non-driver source changes {changed}')
            check(sha(run/'check_sa.f') == proof['driver_sha256'] == proof['sources'][changed[0]], f'{run}: driver hash')
            check(sha(run/'unchecked-worker.f') == proof['template_sha256'], f'{run}: template hash')
            if not trace:
                originaltemplate=(REPORT/'protocol-sources/tirust_benchmark_worker.f').read_text()
                check((run/'unchecked-worker.f').read_text()==originaltemplate.replace('FORCE_STABILITY_CHECK(.TRUE.)','FORCE_STABILITY_CHECK(.FALSE.)'), f'{run}: template changes beyond forcing')
            output=Path(inp['resolved_parameter_card']).parents[1]
            sub=next((output/'SubProcesses').glob('P*_*'))
            check(sha(sub/'check')==proof['executable_sha256'],f'{run}: diagnostic executable hash')
            check(sha(output/'compilation.json')==rec['original_compilation_sha256'],f'{run}: original compilation digest')
            for name,digest in proof['numerical_objects_after'].items():
                check(sha(output/name)==digest,f'{run}: retained numerical object {name}')
            counts['driver_only_proofs_checked'] += 1
        raw=run/'raw'; parsed=[]
        if (raw/'values.dat').exists():
            for line in (raw/'values.dat').read_text().splitlines():
                f=line.split();check(len(f)==8,f'{run}: malformed raw numerical row')
                if len(f)==8:
                    parsed.append({'point':int(f[0]),'order':int(f[1]),'status':int(f[2]),'coefficients':list(map(float,f[3:7])),'accuracy':float(f[7])})
        categoryrows=[x for category in ('accepted','unchecked','rejected') for x in rec['rows'][category]]
        check(len(parsed)==len(categoryrows),f'{run}: discarded raw rows')
        recorded={(x['point'],x['order']):x for x in categoryrows}
        for row in parsed:
            other=recorded.get((row['point'],row['order']))
            check(other is not None and row['status']==other['status'],f'{run}: raw status binding')
            if other and 'coefficients' in other:
                check(row['coefficients']==other['coefficients'] and row['accuracy']==other['accuracy'],f'{run}: raw coefficient binding')
        if rec['status'] in ('complete','unchecked-qp-diagnostic'):
            dims=list(map(int,(raw/'dimensions.dat').read_text().split()))
            check(len(dims)==3 and min(dims)>=0 and dims[0]==max(dims[1:]),f'{run}: dimensions')
            check([(r['point'],r['order']) for r in parsed]==[(p,o) for p in (1,2) for o in range(dims[0]+1)],f'{run}: incomplete numerical rows')
            check(all(math.isfinite(x) for row in parsed for x in row['coefficients']+[row['accuracy']]),f'{run}: nonfinite successful rows')
            if isdiag:
                check(all(row['status']==120 and row['accuracy']==-1 for row in parsed),f'{run}: falsely unchecked QP')
                check(list(map(int,(raw/'timed_status.dat').read_text().split()))==[120,2],f'{run}: QP call histogram')
                check(not rec['rows']['accepted'] and not rec['rows']['rejected'],f'{run}: QP row categories')
            successful.append((run,rec,dims))

overlaps=[]
for i,a in enumerate(intervals):
    for b in intervals[i+1:]:
        if (a['source'],a['case'])==(b['source'],b['case']) and max(a['start'],b['start'])<min(a['end'],b['end']):
            overlaps.append([a,b])
check(not overlaps,'Same-source/case worker intervals overlap')

summary=load(REPORT/'summary.json')
for cmp in summary['comparisons']:
    left=records[(cmp['case'],cmp['reference'])];right=records[(cmp['case'],cmp['candidate'])]
    rows=[]
    for run,rec in (left,right):
        check(rec['status'] in ('complete','unchecked-qp-diagnostic'),f'{run}: failed run in comparison')
        dims=list(map(int,(run/'raw/dimensions.dat').read_text().split()))
        point=1 if cmp['role']=='ordinary-control' else 2
        selected=[r for category in ('accepted','unchecked') for r in rec['rows'][category] if r['point']==point]
        check([(r['point'],r['order']) for r in selected]==[(point,o) for o in range(dims[0]+1)],f'{run}: incomplete comparison orders')
        rows.append(selected)
    check(cmp['stability_checked_both']==(left[1]['stability_checked'] and right[1]['stability_checked']),f'{cmp}: incorrect stability claim')
    passed=True
    for a,b in zip(*rows):
        scale=max(abs(a['coefficients'][1]),abs(b['coefficients'][1]),1e-99)
        for col,(x,y) in enumerate(zip(a['coefficients'],b['coefficients'])):
            ratio=abs(x-y)/abs(x+y) if x+y else 0 if x==y==0 else math.inf
            passed &= ratio<cmp['tolerance'] or (col>=2 and max(abs(x),abs(y))/scale<cmp['tolerance'])
    check(cmp['passed']==passed,f'{cmp["case"]}/{cmp["role"]}: comparison outcome mismatch')
counts['complete_order_comparisons_checked']=len(summary['comparisons'])

def signature(recipe):
    return ([(x['sha256'],x['version']) for x in recipe['compiler']],recipe['flags'])
for proofpath in (REPORT/'diagnostics').glob('*/*/diagnostic-compilation.json'):
    run=proofpath.parent;proof=load(proofpath);inp=load(run/'candidate-inputs.json')
    original=inp['compilation'];left=original['effective_compiler_flags'];right=proof['effective_compiler_flags']
    check(left['environment']==right['environment'] and set(left['scopes'])==set(right['scopes']),f'{run}: compiler environment/scopes')
    for scope in left['scopes']:
        l=left['scopes'][scope]['recipes'];r=right['scopes'][scope]['recipes']
        check(set(l)==set(r) and all(signature(l[k])==signature(r[k]) for k in l),f'{run}: effective compiler flags changed')
    rendered=(run/'unchecked-worker.f').read_text()
    for token,key in (('@PREFIX@','prefix'),('@NEXTERNAL@','nexternal'),('@NBORN@','nborn')):
        rendered=rendered.replace(token,str(original['worker_render_parameters'][key]))
    check(rendered==(run/'check_sa.f').read_text(),f'{run}: driver render binding')

adjudication=load(REPORT/'adjudication.json')
for cmp in adjudication['comparisons']:
    point=1 if cmp['role']=='ordinary-control' else 2;pair=[];complete=[]
    for label in (cmp['reference'],cmp['candidate']):
        run,rec=records[(cmp['case'],label)];dims=list(map(int,(run/'raw/dimensions.dat').read_text().split()))
        rows=[]
        for line in (run/'raw/values.dat').read_text().splitlines():
            f=line.split()
            if int(f[0])==point:rows.append((int(f[1]),list(map(float,f[3:7]))))
        check([x[0] for x in rows]==list(range(dims[0]+1)),f'{run}: incomplete adjudication orders')
        check(all(math.isfinite(x) for _,vals in rows for x in vals),f'{run}: nonfinite adjudication')
        complete.append(rec['status']=='complete' and rec['stability_checked']);pair.append(rows)
    check(cmp['both_complete_checked_runs']==all(complete) and cmp['diagnostic_only']==(not all(complete)),f'{cmp["case"]}: adjudication qualification')
    expected=[];passed=True
    for (order,a),(_,b) in zip(*pair):
        scale=max(abs(a[1]),abs(b[1]),1e-99)
        for col,(x,y) in enumerate(zip(a,b)):
            ratio=abs(x-y)/abs(x+y) if x+y else 0 if x==y==0 else math.inf
            accepted=ratio<cmp['tolerance'] or (col>=2 and max(abs(x),abs(y))/scale<cmp['tolerance'])
            passed &= accepted;expected.append((point,order,col,x,y,accepted))
    actual=[(r['point'],r['order'],r['column'],r['reference'],r['candidate'],r['passed']) for r in cmp['residuals']]
    check(expected==actual and cmp['passed']==passed,f'{cmp["case"]}/{cmp["role"]}: adjudication raw binding/outcome')
counts['diagnostic_only_adjudication_comparisons_checked']=len(adjudication['comparisons'])

traceplan=load(REPORT/'trace-plan.json');traceobs=load(REPORT/'trace-observation.json')
check(sha(REPORT/'protocol-sources/ordinary-trace-worker.f')==traceplan['ordinary_trace_template_sha256'],'Original trace template binding')
for name,digest in traceplan['files'].items():check(sha(REPORT/'protocol-sources'/name)==digest,f'Trace protocol binding {name}')
for name,digest in traceobs['source_and_result_sha256'].items():check(sha(REPORT/name)==digest,f'Trace source/result binding {name}')
tracerun=REPORT/traceobs['run'];timing=list(map(float,(tracerun/'raw/timing.dat').read_text().split()))
check(timing==[traceobs['cold_initialization_seconds'],traceobs['two_point_traversal_seconds']],'Trace raw timing binding')
check(all(0<x<traceobs['phase_timeout_seconds'] for x in timing),'Trace phase deadline statement')
warm=[line.split() for line in (tracerun/'raw/warmup.dat').read_text().splitlines() if not line.startswith('#')]
check({int(r[0]) for r in warm}=={0,1,2} and all(int(r[3])==120 and float(r[-1])==-1 for r in warm),'Trace warmup status statement')

indexpath=REPORT/'evidence-index.json';index=load(indexpath);indexed=set(index['files']);index_issues=[]
for name,identity in index['files'].items():
    p=Path(name)
    if p.is_absolute() or '..' in p.parts or not (REPORT/p).is_file():index_issues.append(name);continue
    if sha(REPORT/p)!=identity['sha256'] or (REPORT/p).stat().st_size!=identity['bytes']:index_issues.append(name)
unindexed=sorted(str(p.relative_to(REPORT)) for p in REPORT.rglob('*') if p.is_file() and p!=indexpath and str(p.relative_to(REPORT)) not in indexed)
check(not index_issues and not unindexed,'Pre-audit packaging index mismatch or uncovered file')
index_snapshot={'sha256':sha(indexpath),'files':len(indexed),'mismatches':index_issues,'unindexed':unindexed,
                'note':'Index before adding this audit artifact; owner will regenerate after packaging it.'}

incompatible=load(REPORT/'fixed-qp-protocol-audit.json')
for request in incompatible['requests']:
    check(sha(REPORT/request['run']/'result.json')==request['record_sha256'],f'{request["run"]}: incompatible failure record modified')

binary=[];secret=[];pycache=[]
patterns={'private_key':re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
          'github_token':re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b'),
          'aws_access_key':re.compile(rb'\bAKIA[0-9A-Z]{16}\b')}
for path in REPORT.rglob('*'):
    if '__pycache__' in path.parts or path.suffix in ('.pyc','.pyo'):pycache.append(str(path.relative_to(REPORT)))
    if not path.is_file():continue
    data=path.read_bytes()
    if data.startswith((b'\x7fELF',b'!<arch>\n',b'MZ')) or path.suffix in ('.o','.a','.so','.pyc'):
        binary.append(str(path.relative_to(REPORT)))
    for name,pattern in patterns.items():
        if pattern.search(data):secret.append({'path':str(path.relative_to(REPORT)),'pattern':name})
check(not binary,'Binary artifacts in evidence tree')
check(not pycache,'Python cache artifacts in evidence tree')
check(not secret,'Recognizable credential material in evidence tree')
notes.append('Baseline settings metadata includes four irrelevant TIRust Python defaults absent from original baseline Fortran cards; applicable settings match.')
notes.append('Credential scan covers private-key headers and recognized GitHub/AWS token patterns; it is not a claim to recognize every possible secret format.')
notes.append('Packaging index checked before adding this audit artifact; its final post-addition index will be different.')
value={'schema':'madloop-physical-stress-final-core-audit-v1','snapshot_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'scope':'Read-only audit of core physical stress evidence. No builds or numerical executions; no stability/performance qualification of forced-QP diagnostics.',
       'counts':dict(counts),'issues':issues,'notes':notes,'intervals':intervals,'same_source_case_overlaps':overlaps,
       'binary_artifacts':binary,'python_cache_artifacts':pycache,'recognizable_sensitive_material':secret,
       'packaging_index_snapshot':index_snapshot,
       'record_identities':identities,'audit_script_sha256':sha(Path(__file__)),'unique_files_hashed':len(cache)}
DEST.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:value[k] for k in ('snapshot_utc','counts','issues','binary_artifacts','python_cache_artifacts','recognizable_sensitive_material')},indent=2))
print(json.dumps({'audit_path':str(DEST),'sha256':sha(DEST),'bytes':DEST.stat().st_size,'record_identities':len(identities)}))
