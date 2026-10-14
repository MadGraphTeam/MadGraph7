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


def _python_function(path, name, device_types):
    """The function ``name`` of the Python module at ``path`` and its
    FRAMEWORK, checked against the devices of the run."""
    module = _import_module(path)
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

    return function, framework


def load_custom_function(ms, contexts, device_types, kind, path):
    """Load the custom ``kind`` function at ``path`` into each context, in the
    same order everywhere. Returns the ``MatrixElementApi`` of each context."""
    if path.endswith(".so"):
        return [context.load_matrix_element(path, "") for context in contexts]
    if not path.endswith(".py"):
        raise RuntimeError(
            f"[custom_functions] {kind} = {path!r}: expected a .so library or a .py file"
        )

    name = f"custom_{kind}"
    function, framework = _python_function(path, name, device_types)
    python_function = ms.PythonFunction(
        function, PYTHON_INPUTS[kind], [kind], framework=framework, name=name
    )
    return [context.add_matrix_element(python_function) for context in contexts]


def load_surrogate(
    ms, contexts, backends, device_types, path, subprocess, pdg_ids, param_card
):
    """Load the [surrogate] function for one subprocess into each context, and
    return the ``MatrixElementApi`` of each context.

    A .so path is a UMAMI library with the inputs of the madmatrix libraries,
    where ``{subprocess}`` and ``{device}`` are replaced, initialized with
    ``param_card``. A .py file defines
    ``custom_surrogate(momenta, pdg_ids, alpha_s)``, which returns |M|^2;
    ``pdg_ids[flavor_index]`` are the PDG ids of the subprocess for each
    flavor index of its matrix element."""
    if path.endswith(".so"):
        return [
            context.load_matrix_element(
                path.format(subprocess=subprocess, device=backend), param_card
            )
            for context, backend in zip(contexts, backends)
        ]
    if not path.endswith(".py"):
        raise RuntimeError(
            f"[surrogate] function = {path!r}: expected a .so library or a .py file"
        )
    function, framework = _python_function(path, "custom_surrogate", device_types)
    particle_count = len(pdg_ids[0])
    if framework == "torch":
        import torch

        tables = {}

        def surrogate(momenta, flavor_index, alpha_s):
            device = momenta.device
            if device not in tables:
                tables[device] = torch.tensor(pdg_ids, dtype=torch.int32, device=device)
            return function(momenta, tables[device][flavor_index.long()], alpha_s)

    else:
        import numpy as np

        table = np.array(pdg_ids, dtype=np.int32)

        def surrogate(momenta, flavor_index, alpha_s):
            return function(momenta, table[flavor_index], alpha_s)

    python_function = ms.PythonFunction(
        surrogate,
        ["momenta", "flavor_index", "alpha_s"],
        ["matrix_element"],
        framework=framework,
        particle_count=particle_count,
        name=f"custom_surrogate ({subprocess})",
    )
    return [context.add_matrix_element(python_function) for context in contexts]
