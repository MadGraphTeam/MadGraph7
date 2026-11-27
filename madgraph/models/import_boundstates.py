import os
import sys


def get_boundstates_ufo(model):
    """Return the Fock states (Boundstate objects) shipped by a UFO model.

    `model` is either a loaded model or the path of a UFO directory. The
    states are read from that UFO's own boundstates.py only, so nothing from
    a model imported earlier in the same session can leak in.

    Constituents named as in the UFO but renamed by the import (the MadGraph
    names of Model.pass_particles_name_in_mg_default: loop_sm's m-, tt- are
    mu-, ta- unless --modelname) are renamed with them (_rename_constituents).
    """
    states = _from_boundstates(model, lambda module: list(module.all_boundstates), [])
    if states and not isinstance(model, str):
        _rename_constituents(model, states)
    return states


def _ufo_particle_pdgs(path):
    """{name: pdg} of the particles and antiparticles of a UFO's particles.py,
    read from the source (the names before any renaming by the import)."""
    import ast
    filename = os.path.join(path, 'particles.py')
    pdgs = {}
    try:
        tree = ast.parse(open(filename).read(), filename)
    except Exception:
        return pdgs
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'Particle':
            kws = dict((kw.arg, kw.value) for kw in node.keywords)
            pdg = kws.get('pdg_code')
            if not isinstance(pdg, ast.Constant):
                continue
            for key, sign in (('name', 1), ('antiname', -1)):
                if isinstance(kws.get(key), ast.Constant):
                    pdgs.setdefault(kws[key].value, sign * pdg.value)
    return pdgs


def _rename_constituents(model, states):
    """The constituent names of the Fock states as the particles of the
    imported model are named: a name that is no particle of the model is
    looked up in the UFO's particles.py and replaced by the current name of
    that PDG code."""
    try:
        particles = model.get('particles')
        missing = set(name for state in states for name in (state.particles or [])
                      if not particles.find_name(name))
    except Exception:
        return
    if not missing:
        return
    try:
        path = model.get('modelpath') or model.__dict__.get('path')
    except Exception:
        path = model.__dict__.get('path')
    ufo = _ufo_particle_pdgs(path) if path else {}
    renamed = {}
    for name in missing:
        part = model.get_particle(ufo[name]) if name in ufo else None
        if part:
            renamed[name] = part.get_name()
    for state in states:
        if any(name in renamed for name in state.particles):
            state.particles = [renamed.get(name, name) for name in state.particles]


def get_boundstate_decays_ufo(model):
    """The decays of the bound states shipped by a UFO model, its
    boundstates.py's all_boundstate_decays: {pdg: (total width in GeV or
    None, {(sorted pdg codes of the products): branching fraction})}, or {}
    when the model has none."""
    return _from_boundstates(model, lambda module: dict(getattr(module, 'all_boundstate_decays', {})),
                             {})


def _from_boundstates(model, attribute, default):
    """attribute(module) of the boundstates.py of a model (a loaded model or
    the path of a UFO directory), default when there is none."""
    if isinstance(model, str):
        path = model
    else:
        # 'modelpath' is computed by Model.get from the version tag and is
        # never a key of the model: a test "'modelpath' in model" is always
        # false, and only restricted models (with their 'path' attribute)
        # found their Fock states, not 'import model sm_onia-full'
        try:
            path = model.get('modelpath')
        except Exception:
            path = None
        path = path or model.__dict__.get('path')
    if not path or not os.path.isfile(os.path.join(path, 'boundstates.py')):
        return default

    # boundstates.py does a top-level 'import object_library', and every UFO
    # ships its own object_library.py. Both names stay in sys.modules once
    # imported, which is how the states of one model used to survive into the
    # next. Import them from this UFO only, then restore whatever those names
    # referred to before.
    names = ('boundstates', 'object_library')
    saved = dict((n, sys.modules.pop(n)) for n in names if n in sys.modules)
    sys.path.insert(0, path)
    try:
        import boundstates
        return attribute(boundstates)
    except Exception:
        return default
    finally:
        sys.path.remove(path)
        for n in names:
            sys.modules.pop(n, None)
        sys.modules.update(saved)
