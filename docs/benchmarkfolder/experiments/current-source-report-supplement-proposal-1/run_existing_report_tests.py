"""Run existing report tests against proposed build code; their Typst is mocked."""
import importlib.util
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TEST = ROOT / "tests/unit_tests/various/test_tirust_report.py"
REFRESH_TEST = ROOT / "docs/benchmarkfolder/test_campaign_report_refresh.py"
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("proposed_report_compatibility", HERE / "proposed/build_report.py")
proposed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proposed)
spec = importlib.util.spec_from_file_location("proposed_refresh_compatibility", HERE / "proposed/campaign_report_refresh.py")
refresh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refresh)


class ProposedBuild:
    def pytest_collection_modifyitems(self, items):
        for item in items:
            if Path(item.module.__file__).resolve() == TEST:
                item.module.REPORT = proposed
            elif Path(item.module.__file__).resolve() == REFRESH_TEST:
                item.module.REFRESH = refresh


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", str(TEST), str(REFRESH_TEST)], plugins=[ProposedBuild()]))
