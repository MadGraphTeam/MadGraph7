"""Run the FKS suite with explicit two-worker bounds and its existing bypass."""
from pathlib import Path
import json,sys,unittest
root=Path.cwd()
sys.path.insert(0,str(root))
import madgraph.interface.master_interface as commands
from tests import test_manager
from tests.unit_tests.iolibs import test_export_fks
original_init=commands.MasterCmd.__init__
def bounded_init(self,*args,**kwargs):
    original_init(self,*args,**kwargs)
    self.options['nb_core']=2
commands.MasterCmd.__init__=bounded_init
method='test_w_nlo_gen_gosam'
assert method in {item[0] for item in test_manager.to_bypass}
setattr(test_export_fks.TestFKSOutput,method,unittest.skip('Existing @bypass_for_py3: CLI forbids GoSam with low_mem_multicore_nlo_generation on Python3')(getattr(test_export_fks.TestFKSOutput,method)))
suite=unittest.defaultTestLoader.loadTestsFromModule(test_export_fks)
result=unittest.TextTestRunner(verbosity=2).run(suite)
summary={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':[(test.id(),reason) for test,reason in result.skipped],'nb_core':2,'successful':result.wasSuccessful(),'production_monkeypatch':False,'test_harness_override':'MasterCmd instances receive nb_core=2; existing Python3 bypass marked unittest.skip'}
(root/'.benchmark-work/final-regression-review-v1/fks-supported-result.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
sys.exit(0 if result.wasSuccessful() else 1)
