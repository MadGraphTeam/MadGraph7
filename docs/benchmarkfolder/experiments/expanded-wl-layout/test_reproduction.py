"""Source selection must survive replacement of the current candidate outputs."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("expanded_experiment", HERE / "run.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)

ORIGINAL = """      DO I=1,LCUT_SIZE
        DO J=1,OUT_SIZE
          DO K=0,0
            OUT(J,K,I)=(0D0,0D0)
          ENDDO
          DO K=1,IN_SIZE
            OUT(J,0,I)=OUT(J,0,I)+A(K,0,I)*B(J,0,K)
          ENDDO
        ENDDO
      ENDDO"""


class RetainedSourceTest(unittest.TestCase):
    @staticmethod
    def source(parent, text):
        path = parent / "case/SubProcesses/P0_case/polynomial.f"
        path.parent.mkdir(parents=True)
        path.write_text(text)
        return parent / "case"

    def test_archived_source_survives_optimized_candidate_regeneration(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            changed, _ = experiment.interchange(ORIGINAL)
            self.source(work / "outputs/candidate", changed)
            archive = self.source(work / "outputs/pre-final-candidate", ORIGINAL)
            with patch.object(experiment.bench, "WORK", work), \
                    patch.object(experiment, "HERE", work / "experiments"):
                selected, _ = experiment.choose_source("case")
                self.assertEqual(selected, archive)
                explicit, _ = experiment.choose_source("case", work / "outputs/pre-final-candidate")
                self.assertEqual(explicit, archive)

    def test_current_optimized_output_cannot_silently_become_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            changed, _ = experiment.interchange(ORIGINAL)
            self.source(work / "outputs/candidate", changed)
            with patch.object(experiment.bench, "WORK", work), \
                    patch.object(experiment, "HERE", work / "experiments"):
                with self.assertRaisesRegex(RuntimeError, "No retained pre-expansion"):
                    experiment.choose_source("case")


if __name__ == "__main__":
    unittest.main()
