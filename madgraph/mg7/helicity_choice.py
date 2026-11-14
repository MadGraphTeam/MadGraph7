"""The good helicities of the MadMatrix libraries of an mg7 run, chosen from the
cards before the run.

A MadMatrix instance left to itself chooses its good helicities on the first 16
events it evaluates, as the Fortran standalone output does. In an mg7 run that
would be one choice per thread and per device, each on whichever events reach it
first. Instead, before the run, an instance of a CPU build of each library
chooses them on 16 RAMBO points of the collider of the run card
(madmatrix_choose_good_helicities: up to its centre-of-mass energy, from just
above the threshold of the process when the beams have PDFs, ISR or
beamstrahlung), and every instance of every context takes that choice
(madspace Context.load_matrix_element(helicity_choice=...),
madmatrix_set_helicity_choice). The choice then depends on the cards only: the
same on every thread and device (a GPU run takes the choice of a scalar CPU
build), and from one run, or gridpack, to the next.
"""

import ctypes
import os

UMAMI_SUCCESS = 0


def choose(library: str, param_card: str, parameters: dict, sqrt_s: float,
           rapidity: float, spread: bool):
    """The helicity choice (madmatrix_get_helicity_choice: the --hel_recycling
    flag, then the kept rows of the helicity table) that an instance of the CPU
    build `library`, initialized with `param_card` and `parameters` as the
    run's are, makes for a collider of centre-of-mass energy `sqrt_s` moving at
    `rapidity` in the lab, whose partonic energy varies with `spread`. None
    where the library cannot choose so (an output made before this existed, a
    phase space RAMBO cannot fill): the libraries then choose on their first
    events."""
    lib = ctypes.CDLL(os.path.abspath(library))
    try:
        choose_function = lib.madmatrix_choose_good_helicities
        get_choice = lib.madmatrix_get_helicity_choice
    except AttributeError:
        return None
    lib.umami_initialize.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    lib.umami_set_parameter.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double, ctypes.c_double]
    lib.umami_free.argtypes = [ctypes.c_void_p]
    choose_function.argtypes = [
        ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_int]
    get_choice.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]

    handle = ctypes.c_void_p()
    if lib.umami_initialize(ctypes.byref(handle), os.path.abspath(param_card).encode()) \
            != UMAMI_SUCCESS:
        raise RuntimeError(f"{library}: umami_initialize failed with {param_card}")
    try:
        for name, value in parameters.items():
            if lib.umami_set_parameter(handle, name.encode(), float(value), 0.) \
                    != UMAMI_SUCCESS:
                raise RuntimeError(f"{library} refuses the parameter {name}")
        if choose_function(handle, float(sqrt_s), float(rapidity), int(bool(spread))) \
                != UMAMI_SUCCESS:
            return None
        size = get_choice(handle, None, 0)
        if size < 1:
            return None
        choice = (ctypes.c_int * size)()
        get_choice(handle, choice, size)
        return list(choice)
    finally:
        lib.umami_free(handle)
