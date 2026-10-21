"""Reuse two unmeasured HEL builds; compile the DP construction variants only."""
from pathlib import Path
import datetime
import os
import sys
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_paper_numerator_v3 as num

WORK = ROOT / ".benchmark-work/paper-numerator-compatibility-v1"
PUBLIC = ROOT / "docs/benchmarkfolder/policy-v3/paper-numerator-compile-proof-v1"
PUBLIC.mkdir(parents=True, exist_ok=False)
specpath = ROOT / "docs/benchmarkfolder/policy-v3/paper-timing-software-proof-v1/specification-held-v2.json"
spec = num.read(specpath)
prior = num.read(ROOT / "docs/benchmarkfolder/policy-v3/paper-timing-compile-proof-v1/status.json")
assert num.workers.digest(Path(num.__file__)) == "2d46a9495f3a817e7438e0c246639f524ab29093b4fe05988769d1624f60410f"
stat = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
status = {"scope": "Two compile/link-only probes; no generated executable, Worker, SETPARA, reduction or MP call",
    "parent_scope": "Prior unmeasured ordinary HEL compile outputs; cloned selector8→6 runtime card only. Not completed Ninja6 timing parents and never valid numerator collection inputs.",
    "pid": os.getpid(), "pgid": os.getpgrp(), "start_ticks": int(stat[19]), "affinity": sorted(os.sched_getaffinity(0)),
    "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "status": "incomplete", "completed": [],
    "controller": num.supplement.bound(Path(__file__)), "producer": num.supplement.bound(Path(num.__file__))}
assert status['affinity'] == [6]
num.bench.dump(PUBLIC / 'status.json', status)
try:
    with num.ownership.owned_runtime(PUBLIC), num.workers.isolated_harness(WORK, PUBLIC):
        for cid in ('qcd-1', 'supplement-gg-hh'):
            source = num.policy.path(next(x['output'] for x in prior['completed'] if x['case'] == cid))
            plain = WORK / 'ordinary-unmeasured' / cid
            saved = num.references.subprocess
            try:
                num.references.subprocess = num.workers.subprocess
                num.references.clone_output(source, plain)
            finally: num.references.subprocess = saved
            objects = [(p, num.MadLoopParam(str(p))) for p in num.cards(plain)]
            for path, card in objects:
                assert str(card['MLReductionLib']) == '8' and card['CTModeRun'] == -1
                before = {str(k).lower(): v for k,v in dict(card).items()}
                if path.is_symlink():
                    assert path.resolve().is_relative_to(plain)
                    content=path.read_bytes();path.unlink();path.write_bytes(content)
                card['MLReductionLib']='6';card.write(str(path), template=str(ROOT/'Template/loop_material/StandAlone/Cards/MadLoopParams.dat'))
                after={str(k).lower():v for k,v in dict(num.MadLoopParam(str(path))).items()}
                assert after == {**before, 'mlreductionlib': '6'}
            case = spec['cases'][cid]
            state = num.full.target_state(plain, 6, case['origin']['reducer_runtime'], WORK/'parent-assets')
            parent = {'output': num.policy.relative(plain), 'state': state, 'filter': None,
                'scope': 'unmeasured compatibility source only; not a scientific ordinary parent'}
            report = WORK / 'cases' / cid
            directory, build = num.prepare(parent, case, WORK/'construction'/cid, report, compile_only=True)
            assert build['compile_only'] and not build['ready_for_numerical']
            assert not list((WORK/'construction'/cid/'SubProcesses').rglob('*HelFilter.dat'))
            status['completed'].append({'case':cid,'private_report':num.policy.relative(report),
                'build':num.supplement.bound(report/'build.json'),'compile_only':True,'ready_for_numerical':False,
                'executable_sha256':build['state']['executable_sha256'],'helicity_filters_created':0})
            num.bench.dump(PUBLIC/'status.json',status)
    status['status']='complete'
except BaseException as error:
    status['status']='failed';status['error']=str(error);raise
finally:
    status['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    num.bench.dump(PUBLIC/'status.json',status)
