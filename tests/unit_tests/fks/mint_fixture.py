"""Runtime modules for MINT tests that do not export a physical subprocess."""
from pathlib import Path


TEMPLATES = Path(__file__).resolve().parents[3] / "Template/NLO/SubProcesses"


def mint_runtime_modules(work):
    # The generic integrator imports the vector driver's storage. Its allocator
    # requires process-specific dimensions, and these tests never call it.
    driver = (TEMPLATES / "driver.f90").read_text()
    declarations = driver[:driver.index("public :: allocate_storage")]
    data = work / "driver_storage.f90"
    data.write_text(declarations + "\nend module driver_vec\n")
    return [str(data), str(TEMPLATES / "mc_decision_ledger.f90"),
            str(TEMPLATES / "mc_batch_trace.f90")]
