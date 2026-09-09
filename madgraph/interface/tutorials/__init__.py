################################################################################
#
# Copyright (c) 2026 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Registry of the available tutorials.

Adding a tutorial means writing one module here and listing it in _MODULES; no
logger, no logging.conf entry and no edit to the interface is needed.  Plugins
register their own through register().
"""

from __future__ import absolute_import

import importlib
import logging

from madgraph.interface.tutorials.session import Step, Tutorial, TutorialSession

logger = logging.getLogger('madgraph')

# modules of this package holding a module-level `tutorial`, in menu order
_MODULES = ['lo', 'syntax', 'mg7', 'madevent', 'model', 'bsm',
            'standalone', 'nlo', 'madloop', 'checks', 'exercises']

_REGISTRY = []          # list of Tutorial, in menu order
_BY_NAME = {}           # name or alias (lowercased) -> Tutorial
_LOADED = False


def register(tutorial, override=False):
    """Add a Tutorial to the registry.  Used by _load() and by plugins."""

    for name in tutorial.names:
        key = name.lower()
        if key in _BY_NAME and not override:
            raise ValueError('tutorial name %r is already taken by %r'
                             % (name, _BY_NAME[key].name))
    existing = _BY_NAME.get(tutorial.name.lower())
    if existing is not None and existing in _REGISTRY:
        _REGISTRY[_REGISTRY.index(existing)] = tutorial
    else:
        _REGISTRY.append(tutorial)
    for name in tutorial.names:
        _BY_NAME[name.lower()] = tutorial
    return tutorial


def _load():
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    for name in _MODULES:
        try:
            module = importlib.import_module(
                'madgraph.interface.tutorials.%s' % name)
        except ImportError as error:
            # a tutorial module that is simply absent is fine (a plugin may add
            # it later); one that fails to import for any other reason is a bug
            # and must not be swallowed
            if getattr(error, 'name', None) != 'madgraph.interface.tutorials.%s' % name:
                raise
            logger.debug('tutorial %s not available: %s', name, error)
            continue
        register(module.tutorial)


def all_tutorials(include_hidden=False):
    """Every registered tutorial, in menu order."""

    _load()
    return [t for t in _REGISTRY if include_hidden or not t.hidden]


def names(include_aliases=False, include_hidden=False):
    """Names accepted by the `tutorial` command (menu order, aliases last)."""

    tutos = all_tutorials(include_hidden=include_hidden)
    out = [t.name for t in tutos]
    if include_aliases:
        for tuto in tutos:
            out.extend(tuto.aliases)
    return out


def get(name):
    """Look a tutorial up by name or alias.  None if unknown."""

    _load()
    return _BY_NAME.get(str(name).lower())


def start(name):
    """Return a fresh TutorialSession for `name`, or None if unknown."""

    tutorial = get(name)
    if tutorial is None:
        return None
    return TutorialSession(tutorial)


def see_also_block(names, indent='  '):
    """Render a "where to go next" list, dropping tutorials that do not exist.

    Lets a tutorial point at one that has not been written yet without ever
    advertising a dead end.
    """

    lines = []
    for name in names:
        tutorial = get(name)
        if tutorial is None:
            continue
        lines.append('%s* `tutorial %-10s %s' % (indent, tutorial.name + '`',
                                                 tutorial.description))
    if not lines:
        return ''
    return '\n'.join(lines)
