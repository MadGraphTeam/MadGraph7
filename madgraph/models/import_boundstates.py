import os
import sys


def get_boundstates_ufo(model):
    """Return the Fock states (Boundstate objects) shipped by a UFO model.

    `model` is either a loaded model or the path of a UFO directory. The
    states are read from that UFO's own boundstates.py only, so nothing from
    a model imported earlier in the same session can leak in.
    """
    return _from_boundstates(model, lambda module: list(module.all_boundstates), [])


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
