"""Copy only closed, hash-bound four-lepton preparation text after timing ends."""
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT=Path.cwd()
REPORT=ROOT/'docs/benchmarkfolder/experiments/current-source-four-lepton-timing-1'
ORIGIN='.tirust-build/current-source-four-lepton-preparation-1'
EXTRA={
 'run.py':'b424cd69bc69afb85be1ce0b643b9b6881b32a29ea53058c1caff53ef3bc9747',
 'specification.json':'40bbd3d9f2908e856b1a91f7b25bf2b2bd988d915dbd865aa97bc47f127a7135',
 'report/commands/candidate-qcd_n4_uux_epem_mupmum.mg5':'b1ecb775d964cd8b1ffe30a15d29e29f9ad467663922b5378d1fc7486e82fd1e',
 'work/logs/generate-candidate-qcd_n4_uux_epem_mupmum.log':'fc70ccabcdc0a3f1cec902720aa76ba4dd99d074b2776ed3e2753cb2d7cf25ee',
 'work/logs/compile-candidate-qcd_n4_uux_epem_mupmum-MODEL.log':'b57b0b3778cea9d9c6632eef26ccab48591cf2a9b780b43f66034832b392989a',
 'work/logs/compile-candidate-qcd_n4_uux_epem_mupmum-DHELAS.log':'f85316b4c71f2526f8d875bdc6017e1dbb97a71a9befad49c15f7e46cc64678e',
 'work/logs/compile-candidate-qcd_n4_uux_epem_mupmum.log':'3b7659b5503b0e55f29fee2cc975e33566da18186e1e69864a8fa425ad1408f0',
 'work/logs/masses-candidate-qcd_n4_uux_epem_mupmum.log':'bf26fd3d9e8ab9eff3080c43c479e51ae9c65aed72163b0368b2ea2afaf063cc',
}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def require(value,message):
    if not value:raise ValueError(message)
def main():
    require(not (REPORT/'index.json').exists(),'Timing packet already sealed')
    rows=read(REPORT/'summary.json')['cases'];require(len(rows)==1 and rows[0]['status']=='complete' and rows[0]['returncode']==0 and not rows[0]['cleanup']['survivors'],'Timing not closed')
    original=read(REPORT/'preparation.json')['originals']['qcd_n4_uux_epem_mupmum']
    files=dict(original['shared']['closed_preparation_inputs'])
    files.update({ORIGIN+'/'+name:checksum for name,checksum in EXTRA.items()})
    output=Path(original['original_output']);generation=read(output/'generation.json');compilation=read(output/'compilation.json')
    require(sha(output/'generation.json')==original['generation_sha256'] and sha(output/'compilation.json')==original['compilation_sha256'],'Original generation/build changed')
    generated=('SubProcesses/global_specs.inc','SubProcesses/'+original['subprocess']+'/loop_CT_calls_1.f','SubProcesses/'+original['subprocess']+'/loop_matrix.f')
    for name in generated:files[str((output/name).relative_to(ROOT))]=compilation['sources'][name]
    destination=REPORT/'origin-preparation';destination.mkdir()
    for name,checksum in files.items():
        path=ROOT/name;require(not Path(name).is_absolute() and '..' not in Path(name).parts and sha(path)==checksum,'Closed preparation input changed')
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    rank=int(re.search(r'parameter\(OVERALLMAXRANK\s*=\s*(\d+)\)',(output/generated[0]).read_text(),re.I).group(1))
    calls=list(map(int,re.findall(r'\bCALL\s+ML5_0_LOOP_(\d+)\s*\(',(output/generated[1]).read_text(),re.I)))
    topology={'overall_max_rank':rank,'max_denominators':max(calls),'loop_call_groups':len(calls),'denominator_call_counts':{str(n):calls.count(n) for n in sorted(set(calls))}}
    require(topology=={'overall_max_rank':3,'max_denominators':4,'loop_call_groups':12,'denominator_call_counts':{'2':2,'3':8,'4':2}},'Unexpected actual loop topology')
    value={'files':files,'generated_sources':list(generated),'source_output':str(output.relative_to(ROOT)), 'topology':topology,'native_executions':0,'amplitude_calls_during_original_preparation':0}
    (REPORT/'origin-preparation.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    shutil.copy2(Path(__file__),REPORT/'origin-preparation-retain.py')
    print(json.dumps({'files':len(files),'topology':topology,'native_executions':0},sort_keys=True))
if __name__=='__main__':main()
