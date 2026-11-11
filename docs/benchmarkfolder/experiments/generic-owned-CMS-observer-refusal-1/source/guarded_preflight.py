"""Authenticated read-only preflight; all process/native/tool/write APIs denied."""
from pathlib import Path
import argparse,datetime,fcntl,fractions,importlib.util,json,resource,signal,sys
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import run_pure_tests as guard

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--phase',choices=('build','run'),required=True);parser.add_argument('--reviewed-build');args=parser.parse_args()
    if args.phase=='run' and not args.reviewed_build:raise ValueError('separate exact observed BUILD review required')
    sys.addaudithook(guard.audit)
    import controller
    b=controller.configure();b.EXPECTED_SOURCE=args.source;b.REVIEWED_BUILD=args.reviewed_build;b.PHASE=args.phase
    result=b.preflight()
    if guard.ATTEMPTS:raise RuntimeError('forbidden preflight API attempts')
    print(json.dumps({'preflight':result,'forbidden_api_attempts':guard.ATTEMPTS,'native_admitted':False},indent=2,sort_keys=True))
if __name__=='__main__':main()
