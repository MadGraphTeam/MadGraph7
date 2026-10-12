"""Loads the functions of the [custom_functions] run card section.

Shared by bin/generate_events of a process directory and of a gridpack. A path
ending in .so is a UMAMI library built by bin/init_custom_func, a path ending in
.py a Python module defining custom_<kind> and FRAMEWORK.
"""

import importlib.util
import logging
import os

logger = logging.getLogger("madevent")

# the arguments passed to custom_<kind> in a Python module
PYTHON_INPUTS = {
    "cut": ["momenta", "pdg_ids"],
    "scale": ["momenta", "pdg_ids"],
    "bias": ["momenta", "pdg_ids"],
    "pdf": ["beam_index", "pdg_id", "momentum_fraction", "scale"],
}

_modules = {}


def _import_module(path):
    """Import the module at ``path`` once per process."""
    path = os.path.abspath(path)
    if path not in _modules:
        spec = importlib.util.spec_from_file_location(
            f"mg7_custom_function_{len(_modules)}", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _modules[path] = module
    return _modules[path]


def load_custom_function(ms, contexts, device_types, kind, path):
    """Load the custom ``kind`` function at ``path`` into each context, in the
    same order everywhere. Returns the ``MatrixElementApi`` of each context."""
    if path.endswith(".so"):
        return [context.load_matrix_element(path, "") for context in contexts]
    if not path.endswith(".py"):
        raise RuntimeError(
            f"[custom_functions] {kind} = {path!r}: expected a .so library or a .py file"
        )

    module = _import_module(path)
    name = f"custom_{kind}"
    function = getattr(module, name, None)
    if not callable(function):
        raise RuntimeError(f"{path} does not define a function {name}")
    framework = getattr(module, "FRAMEWORK", "numpy")
    if framework not in ("numpy", "torch"):
        raise RuntimeError(f"{path}: FRAMEWORK must be 'numpy' or 'torch', not {framework!r}")
    if framework == "numpy" and any(device != "cpu" for device in device_types):
        raise RuntimeError(
            f"{path}: numpy functions run on the CPU only, use FRAMEWORK = 'torch' "
            "for GPU runs"
        )
    if framework == "torch" and "cpu" in device_types:
        import torch

        # madspace already splits CPU batches over its threads
        if torch.get_num_threads() != 1:
            logger.info("Limiting torch to one thread per madspace thread")
            torch.set_num_threads(1)

    python_function = ms.PythonFunction(
        function, PYTHON_INPUTS[kind], [kind], framework=framework, name=name
    )
    return [context.add_matrix_element(python_function) for context in contexts]
