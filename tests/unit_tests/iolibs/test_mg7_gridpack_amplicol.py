"""The shell of a gridpack with [amplicol] processes: the AmpliCol artifacts and
the reweight data files of their ratio libraries."""

import os
import tempfile
import unittest

from madgraph.mg7 import bootstrap as mg7_bootstrap
from madgraph.various.banner import RunCardMG7

# Importing madgraph_process without a compiled madspace starts its build.
madgraph_process = None
if mg7_bootstrap.madspace_is_installed():
    try:
        import madgraph.mg7.madgraph_process as madgraph_process
    except Exception:
        madgraph_process = None


def _touch(path, text="x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


@unittest.skipIf(madgraph_process is None, "madspace not available")
class TestAmplicolGridpackShell(unittest.TestCase):

    def setUp(self):
        self.cwd = os.getcwd()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, self.cwd)
        # an output with one [amplicol] subprocess; a stray temporary file of the writer
        self.reweight = os.path.join("SubProcesses", "A1_x", "reweight")
        for name in ("reweight_p0.rwd", "flavor_switch_p0.json", "reweight_p0.h",
                     ".reweight_p0.rwd.123.tmp"):
            _touch(os.path.join(self.reweight, name), name)
        for name in ("lc", "full"):
            _touch(os.path.join("SubProcesses", "A1_x", name, "artifact.json"))
        for name in ("src", "backend", "lib"):
            _touch(os.path.join(name, "keep"))
        _touch(os.path.join("Cards", "param_card.dat"))
        self.subprocess_data = [{
            "backend": "amplicol",
            "artifact": os.path.join("SubProcesses", "A1_x", "lc"),
        }]

    def shell(self, include_source):
        run_card = RunCardMG7()
        for key, value in (("gridpack.include_source", include_source),
                           ("gridpack.include_madspace", False),
                           ("gridpack.include_madspace_source", False),
                           ("gridpack.include_madtrex", False),
                           ("gridpack.include_madtrex_source", False),
                           ("run.device", ["cpu"])):
            run_card.set(key, value, user=True)
        gridpack = os.path.join(self.tmp.name, "gridpack_%d" % include_source)
        os.mkdir(gridpack)
        madgraph_process.MadgraphProcess.write_gridpack_shell(
            gridpack, run_card, ["scalar"], [], self.subprocess_data, "regular")
        return gridpack

    def test_reweight_data_is_copied_with_and_without_the_sources(self):
        for include_source in (False, True):
            with self.subTest(include_source=include_source):
                reweight = os.path.join(self.shell(include_source), self.reweight)
                for name in ("reweight_p0.rwd", "flavor_switch_p0.json"):
                    with open(os.path.join(reweight, name)) as f:
                        self.assertEqual(f.read(), name)
                for name in ("lc", "full"):
                    self.assertTrue(os.path.isfile(os.path.join(
                        os.path.dirname(reweight), name, "artifact.json")))
                if not include_source:
                    # only the data files of the library, no stray temporary file
                    self.assertFalse(os.path.exists(
                        os.path.join(reweight, ".reweight_p0.rwd.123.tmp")))


if __name__ == "__main__":
    unittest.main()
