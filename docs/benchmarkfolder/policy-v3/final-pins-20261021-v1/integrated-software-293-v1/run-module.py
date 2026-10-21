from pathlib import Path
import datetime,hashlib,importlib,json,os,subprocess,sys,time,unittest
source=Path(sys.argv[1]).resolve();name=sys.argv[2];output=Path(sys.argv[3]);sys.path.insert(0,str(source));os.chdir(source);started=time.monotonic();before={p:(source/p).exists() for p in ['.benchmark-work','.ninjars-build']};assert not any(before.values());commands=[];real_popen=subprocess.Popen
# This software proof may spawn Python control fixtures and read Git metadata.
# Actual compiler, Cargo, native reducer and shell commands fail closed.
def guarded_popen(args,*a,**kw):
 command=list(args) if not isinstance(args,str) else [args];base=Path(str(command[0])).name
 allowed=(not isinstance(args,str) and not kw.get('shell') and (base.startswith('python') or base=='git' or (base=='fc-list' and len(command)==2 and (command[1]=='--help' or command[1].startswith('--format=')))))
 commands.append({'command':[str(v) for v in command],'allowed':allowed})
 if not allowed:raise RuntimeError('Software-only proof rejected non-software subprocess: '+str(command))
 return real_popen(args,*a,**kw)
subprocess.Popen=guarded_popen
module=importlib.import_module(name);filename=Path(module.__file__).resolve();assert filename.is_relative_to(source);checksum=hashlib.sha256(filename.read_bytes()).hexdigest();suite=unittest.defaultTestLoader.loadTestsFromModule(module);result=unittest.TextTestRunner(verbosity=2).run(suite)
row={'module':name,'loaded_from':str(filename.relative_to(source)),'ignored_parents_before':before,'source_sha256':checksum,'source_unchanged':hashlib.sha256(filename.read_bytes()).hexdigest()==checksum,'tests_run':result.testsRun,'failures':[{'test':str(t),'traceback':e} for t,e in result.failures],'errors':[{'test':str(t),'traceback':e} for t,e in result.errors],'skipped':[{'test':str(t),'reason':reason} for t,reason in result.skipped],'passed':result.wasSuccessful(),'elapsed_seconds':time.monotonic()-started,'actual_subprocesses':commands,'nonsoftware_commands_allowed':False};output.write_text(json.dumps(row,indent=2,sort_keys=True)+'\n');sys.exit(0 if result.wasSuccessful() else 1)
