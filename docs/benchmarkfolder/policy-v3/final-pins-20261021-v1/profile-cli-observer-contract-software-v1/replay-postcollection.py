from contextlib import contextmanager
import copy
import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import sys
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
helper_path=ROOT/'.benchmark-work/profile-cli-physical-card-observer-fix-v1/ninjars_profile_cli_v3.py'
assert hashlib.sha256(helper_path.read_bytes()).hexdigest()=='b0681842604b651eac164448f0ca5b19cc61ae02c16704ceb4e9e926516c8ff7'
helper_spec=importlib.util.spec_from_file_location('private_profile_cli_observer_consumer',helper_path)
cli=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(cli)
cli.ROOT=ROOT
BASE = Path(__file__).resolve().parent
CELL = ROOT/'docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/profile-cli-expanded-v3/run-v1/qcd-1/6/0'
original = json.loads((CELL/'timing-metadata.json').read_text())
metadata = json.loads((BASE/'metadata-derived-diagnostic.json').read_text())
actual_output = Path(original['export_dir']); actual_dir = Path(original['subprocess_dir'])
specification = json.loads((CELL/'input-specification.json').read_text())
case = copy.deepcopy(specification['cases']['qcd-1'])
area = BASE/'postcollection-candidate-v2'; area.mkdir()
item = {'case':'qcd-1','backend':6,'repetition':0,'npoints':20}
spec = {'phase':cli.PHASE,'ready':True,'repetitions':3,'initial_stability_points':20,'schedule':[item],
        'sources':{'producer':cli.bound(Path(cli.__file__)),
                   'madgraph/various/process_checks.py':cli.bound(ROOT/'madgraph/various/process_checks.py')},
        'cases':{'qcd-1':case},'report':cli.policy.relative(area/'report'),
        'work':cli.policy.relative(area/'work'),'timeout_seconds':1}
path = area/'specification.json'; path.write_text(json.dumps(spec,indent=2)+'\n')
handoff = area/'root-handoff.json'; handoff.write_text(json.dumps({'root_authorized':True,'phase':cli.PHASE,
    'cpu':207,'specification_sha256':cli.workers.digest(path)})+'\n')
# Source/include/build files only: no archive/object/executable is read by this function.
manifest = cli.bench.generated_source_manifest(actual_output)
(BASE/'actual-generated-source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
libraries = {name.removeprefix('lib/'):record['sha256'] for name,record in metadata['runtime_files'].items()
             if name.startswith('lib/') and (Path(name).suffix in ('.a','.so','.dylib') or '.so.' in Path(name).name)
             and 'sha256' in record}
archive_records = [case['ninja_archive'],case['ninjars_archive']]
real_checked = cli.checked
archive_skips = []
def checked(record):
    if record in archive_records:
        archive_skips.append(record['path'])
        p=cli.policy.path(record['path'])
        cli.require(p.is_file() and p.stat().st_size==record['bytes'],'Archive size changed')
        return p
    return real_checked(record)
@contextmanager
def owned(path):
    yield

def replay_process(command, **kwargs):
    work=kwargs['cwd']; output=work/actual_output.name
    shutil.copytree(CELL/'timing-stage-inputs',output/'SavedRunInfos')
    (output/'SavedRunInfos/profile_timing_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (output/'Cards').mkdir();shutil.copyfile(actual_output/'Cards/param_card.dat',output/'Cards/param_card.dat')
    directory=output/'SubProcesses'/actual_dir.name; (directory/'MadLoop5_resources').mkdir(parents=True)
    for name in ('check','check_sa.f','loop_matrix.f','SavedStabilityRun_Ninja_0.pkl'):
        shutil.copyfile(actual_dir/name,directory/name)
    shutil.copyfile(actual_dir/'MadLoop5_resources/MadLoopParams.dat',directory/'MadLoop5_resources/MadLoopParams.dat')
    shutil.copyfile(CELL/'profile.log',work/'profile_optimized_replayed.log')
    kwargs['stdout'].write((CELL/'stdout.log').read_text())
    return mock.Mock(wait=mock.Mock(return_value=0))

with mock.patch.object(cli,'checked',side_effect=checked), \
        mock.patch.object(cli.os,'sched_getaffinity',return_value={207}), \
        mock.patch.object(cli.ownership,'owned_runtime',owned), \
        mock.patch.object(cli.ownership,'profile_environment',return_value={name:'1' for name in cli.pair.THREADS}), \
        mock.patch.object(cli.workers.subprocess,'Popen',side_effect=replay_process) as process, \
        mock.patch.object(cli.bench,'generated_source_manifest',return_value=manifest), \
        mock.patch.object(cli.policy,'library_hashes',return_value=libraries):
    result=cli.execute_one(path,0,handoff,True)
report=area/'report/qcd-1/6/0/result.json'
observed=cli.report_snapshot(report)
receipt={'status':'PASS-execute_one-postcollection-and-report_snapshot',
    'scope':'Private diagnostic reconstruction/replay only; originalfailedv3result remainscompleteFalse andisnotacceptedtiming',
    'native_calls':0,'Popen_mock_calls':process.call_count,'archive_bytes_hashed':0,
    'archives_checked_by_retained_pins_and_size_only':archive_skips,
    'source_manifest_files':len(manifest['files']),'source_manifest_total_bytes':sum(r['bytes'] for r in manifest['files'].values()),
    'cached_dp_points':observed['stability']['dp_points'],'cached_qp_points':observed['stability']['qp_points'],
    'adaptive_base_points':observed['adaptive_base_points'],'metadata_stage_count':observed['metadata_stage_count'],
    'source_guard_checks':'Actual current producer/process sources checked before/after replay; originalv3sources unchanged',
    'selected_archive_expected':case['ninja_archive']['sha256'],'selected_archives_from_retained_runtime':{name:sha for name,sha in libraries.items() if Path(name).name=='libninja.a'},
    'source':cli.bound(report),'original_result_complete':json.loads((CELL/'result.json').read_text())['complete']}
(BASE/'actual-postcollection-result.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
