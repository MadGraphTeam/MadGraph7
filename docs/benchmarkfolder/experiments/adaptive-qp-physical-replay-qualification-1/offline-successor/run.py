"""Apply the additive classification correction to frozen journals offline."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ('old-journal','new-journal','old-console','new-console','parser','result'):
        cli.add_argument('--'+name,type=Path,required=True)
    args=cli.parse_args()
    assert not args.result.exists(),'Append-only offline result; preserve an existing record'
    parser=load('adaptive_physical_unchanged_frozen_parser',args.parser)
    analysis=load('adaptive_physical_classification_successor',Path(__file__).with_name('analyze.py'))
    old=parser.parse(args.old_journal,'candidate',9);new=parser.parse(args.new_journal,'candidate',9)
    result=analysis.compare(old,new,args.old_console.read_text(),args.new_console.read_text())
    result['offline_input_sha256']={key:sha(getattr(args,key)) for key in ('old_journal','new_journal','old_console','new_console','parser')}
    result['offline_analyzer_sha256']=sha(Path(__file__).with_name('analyze.py'))
    result['native_execution_performed_by_this_comparison']=False
    args.result.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'invariants_passed':result['invariants_passed'],'former_refusals_now_served':result['refusals_now_served'],
                      'no_recorded_native_omissions':result['observed_sums_have_no_recorded_native_omissions'],
                      'physical_accuracy_qualified':False,'native_launched':False},sort_keys=True))
    if not result['invariants_passed']:raise SystemExit(1)
