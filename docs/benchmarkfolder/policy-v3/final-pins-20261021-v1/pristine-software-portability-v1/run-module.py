import hashlib,importlib,json,pathlib,sys,time,unittest
source=pathlib.Path(sys.argv[1]).resolve();name=sys.argv[2];output=pathlib.Path(sys.argv[3]);sys.path.insert(0,str(source))
started=time.monotonic();before={n:(source/n).exists() for n in (".benchmark-work",".ninjars-build")};assert not any(before.values())
module=importlib.import_module(name);filename=pathlib.Path(module.__file__).resolve();assert filename.is_relative_to(source)
checksum=hashlib.sha256(filename.read_bytes()).hexdigest();suite=unittest.defaultTestLoader.loadTestsFromModule(module)
result=unittest.TextTestRunner(verbosity=1).run(suite)
record={"module":name,"loaded_from":str(filename),"private_source":str(source),"ignored_parents_before":before,"source_sha256":checksum,"source_unchanged":hashlib.sha256(filename.read_bytes()).hexdigest()==checksum,"tests_run":result.testsRun,"failures":[str(t) for t,_ in result.failures],"errors":[{"test":str(t),"last_line":e.strip().splitlines()[-1]} for t,e in result.errors],"skipped":[{"test":str(t),"reason":r} for t,r in result.skipped],"passed":result.wasSuccessful(),"elapsed_seconds":time.monotonic()-started}
output.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n");sys.exit(0 if result.wasSuccessful() else 1)
