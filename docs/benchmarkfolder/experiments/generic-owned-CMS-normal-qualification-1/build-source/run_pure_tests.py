"""Guard source-only tests against process/native/write operations."""
from pathlib import Path
import ast,copy,dataclasses,hashlib,importlib.util,json,os,shlex,shutil,struct,sys,unittest
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];ATTEMPTS=[]
def audit(event,args):
    def deny():ATTEMPTS.append(event);raise RuntimeError('pure source boundary '+event)
    if event.startswith(('subprocess.','ctypes.','os.exec','os.spawn','socket.')) or event in ('os.system','os.fork','os.posix_spawn'):deny()
    if event=='import' and len(args)>1 and args[1] and str(args[1]).endswith(('.so','.pyd')):deny()
    if event=='open':
        mode,flags=args[1:3]
        if (isinstance(mode,str) and any(x in mode for x in 'wax+')) or isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):deny()
        if not isinstance(args[0],int):
            path=Path(os.fsdecode(args[0])).resolve()
            if not path.is_relative_to(ROOT) and not path.is_relative_to('/nix/store'):deny()
    if event in ('os.mkdir','os.remove','os.rmdir','os.rename','os.link','os.symlink','os.chmod','os.truncate'):deny()
if __name__=='__main__':
    sys.dont_write_bytecode=True;sys.path.insert(0,str(HERE));sys.addaudithook(audit)
    import test_build
    suite=unittest.defaultTestLoader.loadTestsFromModule(test_build)
    r=unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({'status':'pass' if r.wasSuccessful() and not ATTEMPTS else 'fail','tests':r.testsRun,'forbidden_api_attempts':ATTEMPTS,'compiler_calls':0,'loader_calls':0,'native_calls':0,'RUN_admitted':False},sort_keys=True))
    raise SystemExit(0 if r.wasSuccessful() and not ATTEMPTS else 1)
