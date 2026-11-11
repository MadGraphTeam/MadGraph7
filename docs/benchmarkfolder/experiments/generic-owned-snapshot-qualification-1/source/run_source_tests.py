"""Guarded source/oracle/synthetic-replay tests; never invokes the Fortran driver."""
from pathlib import Path
import json
import sys
import unittest
import test_proposal
sys.path.insert(0,str(test_proposal.corpus.PARENT));from source_guard import SourceAudit

if __name__=='__main__':
    here=Path(__file__).resolve().parent;audit=SourceAudit(here,here.parents[1]);sys.addaudithook(audit)
    suite=unittest.defaultTestLoader.loadTestsFromModule(test_proposal);count=suite.countTestCases()
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps(dict(status='pass' if result.wasSuccessful() else 'fail',tests=count,
        synthetic_observations_only=True,forbidden_api_attempts=audit.attempts,compiler_calls=0,
        native_calls=0,full_caller_qualified=False,scientific_dispatch_enabled=False),sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() and not audit.attempts else 1)
