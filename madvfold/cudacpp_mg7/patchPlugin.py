#!/usr/bin/env python3
# Copyright (C) 2026 CERN.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created by A. Valassi (Nov 2026) for project MadVFold (arXiv:2609.23500).
"""Minimal changes to the cudacpp plugin, for its use in MadGraph7 (see installPlugin.sh)"""
import os, sys

def patch(path, *pairs):
    s = open(path).read()
    for old, new in pairs:
        if old not in s:
            sys.exit('ERROR! Cannot patch %s: not found: %s' % (path, old))
        s = s.replace(old, new)
    open(path, 'w').write(s)

P = sys.argv[1]
# MadtRex is not needed by MadVFold, and it does not load in MadGraph7
patch(os.path.join(P, '__init__.py'),
      ("    __import__('%s.trex'%PLUGIN_NAME)\n    trex = sys.modules['%s.trex'%PLUGIN_NAME]\n",
       "    # MadGraph7: no MadtRex\n"),
      ("                   'standalone_trex' : trex.TREX_ProcessExporter,\n", ""),
      ("    new_reweight = {'madtrex': trex.TREX_ReweightInterface}\n", "    new_reweight = {}\n"))
# MadGraph7 removed the GPU exporters which the plugin derives from: add them to the independent
# copy of the export_cpp module of MadGraph7 which the plugin loads (not to the module itself)
patch(os.path.join(P, 'output.py'),
      ("SPEC_EXPORTCPP.loader.exec_module(PLUGIN_export_cpp)\n",
       "SPEC_EXPORTCPP.loader.exec_module(PLUGIN_export_cpp)\n"
       "_gpufile = os.path.join(os.path.dirname(__file__), 'export_cpp_gpu.py') # MadGraph7\n"
       "exec(compile(open(_gpufile).read(), _gpufile, 'exec'), PLUGIN_export_cpp.__dict__)\n"))
# The v3.8.1 aloha_writers.py: the writers of MadGraph7 use ALOHAOBJ arguments, and no six in
# MadGraph7. The writer must also derive from the WriteALOHA of MadGraph7 to pass the check
# in its WriterFactory: this is after the v3.8.1 classes in the MRO, so it adds no method.
patch(os.path.join(P, 'model_handling.py'),
      ("import aloha.aloha_writers as aloha_writers\n",
       "import aloha.aloha_writers as MG7_aloha_writers # MadGraph7\n"
       "import importlib.util as _ilu\n"
       "_spec = _ilu.spec_from_file_location('%s.aloha_writers_mg5v381'%PLUGIN_NAME, "
       "os.path.join(os.path.dirname(__file__), 'aloha_writers_mg5v381.py'))\n"
       "aloha_writers = _ilu.module_from_spec(_spec)\n"
       "_spec.loader.exec_module(aloha_writers)\n"),
      ("from six import StringIO\n", "from io import StringIO\n"),
      ("class PLUGIN_ALOHAWriter(aloha_writers.ALOHAWriterForGPU):",
       "class PLUGIN_ALOHAWriter(aloha_writers.ALOHAWriterForGPU, MG7_aloha_writers.WriteALOHA):"))
