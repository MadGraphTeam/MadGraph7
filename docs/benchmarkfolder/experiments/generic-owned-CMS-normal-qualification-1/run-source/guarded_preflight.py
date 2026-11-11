"""Pure preflight only: no compiler/loader/native/process/write APIs."""
from pathlib import Path
import argparse,datetime,fcntl,fractions,importlib.util,json,resource,signal,sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_pure_tests as guard

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);args=parser.parse_args()
    sys.addaudithook(guard.audit)
    import controller
    b=controller.configure();b.EXPECTED_SOURCE=args.source;b.REVIEWED_BUILD='be89129beb16a82d65624748aa6948d1d955358d910efc9956845ae6efff98f2';b.PHASE='run'
    result=b.preflight()
    if guard.ATTEMPTS:raise RuntimeError('forbidden preflight API attempts')
    print(json.dumps({'preflight':result,'forbidden_api_attempts':guard.ATTEMPTS,'native_admitted':False},indent=2,sort_keys=True))
if __name__=='__main__':main()
