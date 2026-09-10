First steps
===========

Type ``tutorial`` at the MadGraph7 prompt for a guided walkthrough of the commands.

To start your first process, you can enter the following commands::

    generate p p > t t~
    output my_first_run
    launch

Graphical interface
-------------------

`MadBoard <https://github.com/MadGraphTeam/MadBoard>`_ is a separate package providing a
modern web interface to MadGraph, for setting up processes, following runs from the
browser and inspecting their results. You can install it with::

    pip install madboard

and launch it by running the ``madboard`` command within your MadGraph7 directory.
