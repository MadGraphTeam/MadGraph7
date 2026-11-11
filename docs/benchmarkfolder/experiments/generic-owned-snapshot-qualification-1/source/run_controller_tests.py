"""Pure preflight/mocked-controller tests; no compiler/loader/executable admission."""
from pathlib import Path
import json
import sys
import unittest
from contextlib import ExitStack
from unittest.mock import patch
import test_controller
import test_proposal
sys.path.insert(0,str(test_proposal.corpus.PARENT));from source_guard import SourceAudit

if __name__=='__main__':
    here=Path(__file__).resolve().parent;audit=SourceAudit(here,here.parents[1]);sys.addaudithook(audit)
    controls=[]
    def denied(*args,**kwargs):
        controls.append(str(args[:2]));raise AssertionError('Native process/control API denied by pure tests')
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(module)
                             for module in (test_proposal,test_controller)]);count=suite.countTestCases()
    with ExitStack() as guards:
        for obj,names in ((test_controller.c.os,('kill','killpg','system','fork','forkpty','posix_spawn')),
                          (test_controller.c.resource,('setrlimit',)),
                          (test_controller.c.signal,('setitimer',))):
            for name in names:
                if hasattr(obj,name):guards.enter_context(patch.object(obj,name,denied))
        result=unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps(dict(status='pass' if result.wasSuccessful() else 'fail',tests=count,
        mocked_controller_and_synthetic_records_only=True,forbidden_api_attempts=audit.attempts,
        denied_control_attempts=controls,compiler_queries=0,loader_queries=0,native_executions=0,BUILD_admitted=False,RUN_admitted=False,
        full_caller_qualified=False,scientific_dispatch_enabled=False),sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() and not audit.attempts and not controls else 1)
