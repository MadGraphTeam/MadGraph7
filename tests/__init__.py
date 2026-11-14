to_bypass = []


def import_madspace():
    """The madspace module the mg7 runs use, or None if it cannot be imported.

    A bare ``import madspace`` takes whatever comes first on sys.path -- for
    instance an older madspace left in the virtualenv's site-packages -- and
    caches it in sys.modules for the rest of the test process. The mg7 runs
    use the bundled build (madspace/install, see mg7/bootstrap.py) instead,
    and a compiled madspace cannot be swapped for another one in the same
    process: the second extension fails to load against the libmadspace
    already loaded, and every later ``import madspace`` raises. So every mg7
    gate imports through here, the way the launcher does: the bundled build
    when it is installed (never installing it from a test), else whatever
    madspace is importable.
    """
    try:
        from madgraph.iolibs.template_files.mg7 import bootstrap
        installed = bootstrap.madspace_is_installed()
    except Exception:
        bootstrap, installed = None, False
    try:
        if installed:
            # Already installed, so this only prepends the install directory.
            bootstrap.ensure_madspace(interactive=False)
            try:
                import madspace
            finally:
                # As in launch.py: the install directory also carries
                # madspace's build dependencies, which would shadow ours.
                bootstrap.drop_install_path()
        else:
            import madspace
    except ImportError:
        return None
    return madspace


_mg5_options = None


def lhapdf_data_path(pdf_set):
    """The LHAPDF data directory holding *pdf_set*, or None.

    Resolved as bin/generate_events does (misc.resolve_lhapdf): from
    $LHAPDF_DATA_PATH, then from the MadGraph configuration (the 'lhapdf'
    option, HEPTools), so a gate does not skip a run that would work.
    """
    global _mg5_options
    import madgraph.various.misc as misc
    if _mg5_options is None:
        try:
            import madgraph.interface.master_interface as MGCmd
            _mg5_options = dict(MGCmd.MasterCmd().options)
        except Exception:
            _mg5_options = {}
    return misc.resolve_lhapdf(_mg5_options).find_set(pdf_set)
