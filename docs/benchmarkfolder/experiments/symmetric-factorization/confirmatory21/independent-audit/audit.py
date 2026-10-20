#!/usr/bin/env python3
"""Independent retained-data/live-source audit. No builds or amplitude calls."""
from pathlib import Path
from collections import Counter
import hashlib,importlib.util,json,math,random,sys
ROOT=Path(__file__).resolve().parents[6]
BASE=ROOT/'docs/benchmarkfolder/experiments/symmetric-factorization'
HERE=Path(__file__).resolve().parent
REPORT=HERE.parent/'qcd3-factor-confirmatory21-v1'
MONITOR=REPORT.with_name(REPORT.name+'-monitor')
WORK=ROOT/'.benchmark-work/experiments/symmetric-factorization'/REPORT.name
PARENT=ROOT/'.benchmark-work/policy-v2/generation-abi3-terminal-fixed/outputs/candidate/qcd-3'
sys.path[:0]=[str(BASE),str(ROOT/'tools')]
import process_prototype
import ninjars_benchmark as bench
import ninjars_benchmark_policy_v2 as policy
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())

def audit():
 result=read(REPORT/'result.json');assert result['complete'] and not result['acceptance_eligible']
 assert result['cpu']==208 and result['policy']['required_repetitions']==result['policy']['passes']==21
 assert result['policy']==read(REPORT/'predeclared-policy.json')
 assert result['completed_campaign_seal_sha256']==sha(BASE/'campaign-seal.json')
 assert result['prior_seven_result_sha256']==sha(BASE/'qcd3-factor-quiet-v1/result.json')
 for name in ['process_prototype.py','check_fallback.py']:
  assert (REPORT/name).read_bytes()==(BASE/name).read_bytes()
 spec=importlib.util.spec_from_file_location('retained_confirmation_factor',REPORT/'factor.py')
 factor=importlib.util.module_from_spec(spec);spec.loader.exec_module(factor)
 factor.HERE=BASE;factor.PREVIOUS=BASE.parent/'symmetric-accumulation';process_prototype.factor=factor
 before=bench.generated_source_manifest(PARENT);assert before==result['source_manifest']==read(REPORT/'source-inputs/compiled-sources.json')
 assert read(PARENT/'generation.json')==result['source_identity']==read(REPORT/'source-inputs/generation.json')
 assert (REPORT/'points.dat').read_bytes()==(ROOT/'docs/benchmarkfolder/points/qcd-3.dat').read_bytes()
 assert sha(REPORT/'points.dat')==result['points_sha256']
 op=next((WORK/'original').glob('SubProcesses/P*_*'));fp=next((WORK/'factor').glob('SubProcesses/P*_*'))
 changed,proof=process_prototype.transform(op,trace=False)
 assert proof==read(REPORT/'factor-proof.json')
 for name,text in changed.items():assert (fp/name).read_text()==text,name
 assert 'check_sa.f' not in changed
 assert 'FACTOR_COUNTS' not in (fp/'polynomial.f').read_text()
 source_libraries=policy.library_hashes(PARENT)
 source_metadata={k:sha(v) for k,v in policy.runtime_metadata_files(PARENT).items()}
 physical=(PARENT/'Cards/param_card.dat').read_bytes()
 assert physical==(REPORT/'source-inputs/Cards/param_card.dat').read_bytes()
 assert (REPORT/'original-MadLoopParams.dat').read_bytes()==(REPORT/'factor-MadLoopParams.dat').read_bytes()
 snapshots={};times={};rows={}
 for variant in ['original','factor']:
  output=WORK/variant;directory=next(output.glob('SubProcesses/P*_*'));manifest=bench.generated_source_manifest(output)
  assert manifest==read(REPORT/(variant+'-compiled-sources.json'))
  assert manifest==read(BASE/'qcd3-factor-quiet-v1'/(variant+'-compiled-sources.json'))
  assert manifest['files_sha256']==result['builds'][variant]['compiled_sources_sha256']
  assert sha(directory/'check')==result['builds'][variant]['executable_sha256']
  assert policy.library_hashes(output)==source_libraries
  assert {k:sha(v) for k,v in policy.runtime_metadata_files(output).items()}==source_metadata
  assert policy.delta(before['files'],manifest['files'])==({str((directory/name).relative_to(output)) for name in changed} if variant=='factor' else set())
  assert (output/'Cards/param_card.dat').read_bytes()==physical
  assert (output/'Cards/MadLoopParams.dat').read_bytes()==(REPORT/(variant+'-MadLoopParams.dat')).read_bytes()
  card=next(p for p in [directory/'param_card.dat']+[directory.joinpath(*(['..']*n),'Cards/param_card.dat') for n in range(6)] if p.is_file())
  assert card.read_bytes()==physical
  for p in (BASE/'qcd3-factor-quiet-v1/generated-sources'/variant).iterdir():assert (directory/p.name).read_bytes()==p.read_bytes()
  assert [r['repetition'] for r in result['measurements'][variant]]==list(range(21))
  snapshots[variant]={'executable_sha256':sha(directory/'check'),'generated_sources_sha256':manifest['files_sha256']}
  times[variant]=[]
  for rep,record in enumerate(result['measurements'][variant]):
   stem=f'{variant}-{rep}';values=REPORT/(stem+'-values.dat');parsed=[]
   assert sha(values)==record['values_sha256'] and record['returncode']==0
   for line in values.read_text().splitlines():
    f=line.split();assert len(f)==8
    r={'point':int(f[0]),'order':int(f[1]),'status':int(f[2]),'coefficients':list(map(float,f[3:7])),'accuracy':float(f[7])}
    bench.validate_status(r['status'],6);assert r['accuracy']>=0 and all(math.isfinite(v) for v in r['coefficients']+[r['accuracy']]);parsed.append(r)
   assert parsed==record['rows'];assert [(r['point'],r['order']) for r in parsed]==[(pt,o) for pt in range(1,101) for o in range(2)]
   assert list(map(int,(REPORT/(stem+'-dimensions.dat')).read_text().split()))==record['answer_dimensions']==[1,1,1]
   counts={}
   for line in (REPORT/(stem+'-timed_status.dat')).read_text().splitlines():
    code,count=line.split();assert code not in counts;counts[code]=int(count)
   assert counts==record['timed_status_counts']=={'216':98,'328':2}
   assert dict(Counter(str(r['status']) for r in parsed if r['order']==0))==counts
   timing=list(map(float,(REPORT/(stem+'-timing.dat')).read_text().split()));assert len(timing)==2 and all(math.isfinite(x) and x>0 for x in timing)
   assert timing[0]==record['cold_initialization_and_warm_pass_seconds'];assert timing[1]/100==record['seconds_per_point'];times[variant].append(timing[1]/100);rows[variant,rep]=parsed
 assert [c['repetition'] for c in result['comparisons']]==list(range(21))
 for rep in range(21):
  assert (REPORT/f'original-{rep}-values.dat').read_bytes()==(REPORT/f'factor-{rep}-values.dat').read_bytes()
  assert rows['original',rep]==rows['original',0]==rows['factor',rep]
  stored=result['comparisons'][rep];assert stored['passed'] and stored['coefficient_count']==stored['printed_binary64_identical_coefficients']==800
  assert not stored['accuracy_or_status_changes'] and stored['added_qp_rescues']==0
 # Independent implementation of the declared fixed-length paired percentile bootstrap.
 median=lambda xs:sorted(xs)[len(xs)//2]
 ratio=median(times['factor'])/median(times['original']);rng=random.Random(410083);bootstrap=[]
 for _ in range(10000):
  indices=[int(rng.random()*21) for __ in range(21)]
  bootstrap.append(median([times['factor'][i] for i in indices])/median([times['original'][i] for i in indices]))
 bootstrap.sort();ci=[bootstrap[249],bootstrap[9749]]
 assert ratio==result['paired_statistics']['runtime_ratio'] and ci==result['paired_statistics']['ratio_confidence_interval_95']
 assert result['paired_statistics']['measurements']['baseline']['samples']==times['original']
 assert result['paired_statistics']['measurements']['candidate']['samples']==times['factor']
 assert ci[1]<1 and result['paired_statistics']['status']=='passed'
 context=read(MONITOR/'result.json');topology=read(MONITOR/'topology.json');assert context['returncode']==0 and topology['evaluation_cpu']==208 and topology['siblings']==[208]
 actual=[]
 for line in (MONITOR/'run.log').read_text().splitlines():
  f=line.split()
  if len(f)>4 and f[0] in ['original','factor'] and f[2:5]==['100','points','complete']:actual.append((f[0],int(f[1])))
 expected=[(v,n) for n in range(21) for v in (['original','factor'] if n%2==0 else ['factor','original'])]
 assert actual==expected
 samples=[json.loads(line) for line in (MONITOR/'samples.jsonl').read_text().splitlines()];assert len(samples)==context['samples']
 assert all(b['monotonic_seconds']>=a['monotonic_seconds'] for a,b in zip(samples,samples[1:]))
 return {'passed':True,'source_equivalent_to_prior_seven_pairs':True,'all_runtime_libraries_sha256':source_libraries,'all_runtime_metadata_sha256':source_metadata,'builds':snapshots,'pairs':21,'point_order_rows_per_variant_per_pair':200,'compared_coefficients':16800,'exact_coefficients':16800,'all_accuracy_and_status_identical':True,'per_traversal_status_counts':{'216':98,'328':2},'raw_ratio':ratio,'independent_paired_bootstrap_95':ci,'bootstrap_seed':410083,'bootstrap_resamples':10000,'alternating_order_verified':True,'all_samples_retained':True,'earlier_seven_samples_excluded':True,'physical_parameter_card_sha256':sha(PARENT/'Cards/param_card.dat'),'result_sha256':sha(REPORT/'result.json'),'scope':'Independent retained-data and live-source audit only. Supports this isolated Ninja6 QCD3 prototype gain; not final NinjaRS acceptance or other-process coverage.','physics_evaluations_started':False,'builds_started':False}

if __name__=='__main__':print(json.dumps(audit(),indent=2))
