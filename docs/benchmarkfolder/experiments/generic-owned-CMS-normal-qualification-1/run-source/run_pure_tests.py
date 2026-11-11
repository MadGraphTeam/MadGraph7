"""RUN source/RAW64/protocol controls only; all process/native/write denied."""
from pathlib import Path
import fractions
import ast,copy,dataclasses,hashlib,importlib.util,json,math,os,selectors,shlex,shutil,struct,subprocess,sys,time,types,unittest
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];ATTEMPTS=[]
def audit(event,args):
    def deny():ATTEMPTS.append(event);raise RuntimeError('pure RUN source boundary '+event)
    if event.startswith(('subprocess.','ctypes.','os.exec','os.spawn','socket.')) or event in ('os.system','os.fork','os.posix_spawn'):deny()
    if event=='import' and len(args)>1 and args[1] and str(args[1]).endswith(('.so','.pyd')):deny()
    if event=='open':
        mode,flags=args[1:3]
        if (isinstance(mode,str) and any(x in mode for x in 'wax+')) or isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):deny()
        if not isinstance(args[0],int):
            path=Path(os.fsdecode(args[0])).resolve()
            if not path.is_relative_to(ROOT) and not path.is_relative_to('/nix/store'):deny()
    if event in ('os.mkdir','os.remove','os.rmdir','os.rename','os.link','os.symlink','os.chmod','os.truncate','os.mkfifo'):deny()
if __name__=='__main__':
    sys.dont_write_bytecode=True;sys.path.insert(0,str(HERE));sys.addaudithook(audit)
    import test_run,test_analysis
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(test_run),unittest.defaultTestLoader.loadTestsFromModule(test_analysis)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({'status':'pass' if result.wasSuccessful() and not ATTEMPTS else 'fail','tests':result.testsRun,'forbidden_api_attempts':ATTEMPTS,'compiler_calls':0,'loader_queries':0,'native_main_calls':0,'native_admitted':False},sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() and not ATTEMPTS else 1)
