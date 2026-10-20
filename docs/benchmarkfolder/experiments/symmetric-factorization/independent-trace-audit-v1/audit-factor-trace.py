from pathlib import Path
import sys,json,hashlib,importlib.util
R=Path.cwd();C=R/'docs/benchmarkfolder/experiments/symmetric-factorization';D=C/'qcd3-factor-trace-v1';W=R/'.benchmark-work/experiments/symmetric-factorization/qcd3-factor-trace-v1';P=R/'.benchmark-work/policy-v2/generation-abi3-terminal-fixed/outputs/candidate/qcd-3'
sys.path[:0]=[str(C),str(R/'tools')]
import run_process as run
import ninjars_benchmark as bench
import ninjars_benchmark_policy_v2 as policy
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((D/'result.json').read_text());assert r['complete'] and not r['acceptance_eligible']
for n in ['run_process.py','process_prototype.py','check_fallback.py']:assert (C/n).read_bytes()==(D/n).read_bytes(),n
s=importlib.util.spec_from_file_location('retained_factor',D/'factor.py');f=importlib.util.module_from_spec(s);s.loader.exec_module(f);f.HERE=C;f.PREVIOUS=C.parent/'symmetric-accumulation';run.process_prototype.factor=f
pm=bench.generated_source_manifest(P);assert pm==r['source_manifest'];assert pm==json.loads((D/'source-inputs/compiled-sources.json').read_text())
assert (P/'generation.json').read_bytes()==(D/'source-inputs/generation.json').read_bytes();assert json.loads((P/'generation.json').read_text())==r['source_identity']
assert (D/'points.dat').read_bytes()==(R/'docs/benchmarkfolder/points/qcd-3.dat').read_bytes();assert sha(D/'points.dat')==r['points_sha256']
assert (D/'original-MadLoopParams.dat').read_bytes()==(D/'factor-MadLoopParams.dat').read_bytes()
pd=next((W/'original').glob('SubProcesses/P*_*'));changed,proof=run.process_prototype.transform(pd,trace=True);assert proof==json.loads((D/'factor-proof.json').read_text())
fd=next((W/'factor').glob('SubProcesses/P*_*'))
for n,t in changed.items():assert (fd/n).read_text()==t,n
rows={};libraries=policy.library_hashes(P)
for v in ['original','factor']:
 o=W/v;d=next(o.glob('SubProcesses/P*_*'));m=bench.generated_source_manifest(o)
 assert m==json.loads((D/(v+'-compiled-sources.json')).read_text())
 assert m['files_sha256']==r['builds'][v]['compiled_sources_sha256'];assert sha(d/'check')==r['builds'][v]['executable_sha256'];assert policy.library_hashes(o)==libraries
 assert (o/'Cards/param_card.dat').read_bytes()==(P/'Cards/param_card.dat').read_bytes()==(D/'source-inputs/Cards/param_card.dat').read_bytes();assert (o/'Cards/MadLoopParams.dat').read_bytes()==(D/(v+'-MadLoopParams.dat')).read_bytes()
 actual=next(p for p in [d/'param_card.dat']+[d.joinpath(*(['..']*level),'Cards/param_card.dat') for level in range(6)] if p.is_file());assert actual.read_bytes()==(P/'Cards/param_card.dat').read_bytes()
 delta=policy.delta(pm['files'],m['files']);expected={str((d/n).relative_to(o)) for n in changed} if v=='factor' else set();assert delta==expected,(v,delta)
 raw=D/(v+'-values.dat');rows[v]=run.rows(raw);assert rows[v]==r['measurements'][v]['rows'];assert sha(raw)==r['measurements'][v]['values_sha256'];assert [(x['point'],x['order']) for x in rows[v]]==[(p,o) for p in range(1,101) for o in range(2)]
 assert list(map(int,(D/(v+'-dimensions.dat')).read_text().split()))==[1,1,1]
 counts=dict((str(a),int(b)) for a,b in (line.split() for line in (D/(v+'-timed_status.dat')).read_text().splitlines()));assert counts=={'216':98,'328':2}==r['measurements'][v]['timed_status_counts']
comparison=run.compare(rows['original'],rows['factor'],r['policy']);assert comparison==r['comparison'];assert comparison['all_printed_coefficients_identical'] and not comparison['accuracy_or_status_changes'];assert (D/'original-values.dat').read_bytes()==(D/'factor-values.dat').read_bytes()
counts=[dict(zip(['index','fast','fallback'],map(int,line.split()))) for line in (D/'factor-factor_trace.dat').read_text().splitlines()];assert counts==r['guard_counts'];assert [c['index'] for c in counts]==list(range(1,len(proof['helpers'])+1));assert all(c['fast']>=0 and c['fallback']>=0 for c in counts);assert {k:sum(c[k] for c in counts) for k in ['fast','fallback']}==r['guard_totals']=={'fast':6528,'fallback':0}
driver=(fd/'check_sa.f').read_text();assert driver.index('FACTOR_COUNTS=0')<driver.index('CALL SYSTEM_CLOCK(T0)');assert driver.index('ELAPSED=DBLE(T1-T0)/DBLE(RATE)')<driver.index("FILE='factor_trace.dat'")
print(json.dumps({'passed':True,'raw_point_order_rows_per_variant':200,'printed_coefficients_exact':800,'status_counts':{'216':98,'328':2},'guard_totals':r['guard_totals'],'retained_factor_sha256':sha(D/'factor.py'),'source_transform_reproduced':True,'source_card_library_executable_bindings_verified':True,'diagnostic_only':True,'physics_evaluations_started':False,'builds_started':False},indent=2))
