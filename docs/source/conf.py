import os
import subprocess
import sys

# Import the locally built package (``python madspace/install.py --source``)
# rather than any system-wide madspace, so the Python API docs track the
# working tree. Override with MADSPACE_INSTALL if it lives elsewhere.
_local_install = os.environ.get(
    "MADSPACE_INSTALL",
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "madspace", "install")
    ),
)
if os.path.isdir(os.path.join(_local_install, "madspace")):
    sys.path.insert(0, _local_install)

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "madspace"
copyright = "2025, Theo Heimel"
author = "Theo Heimel"
release = "0.2.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    # "sphinx.ext.linkcode",
    "sphinx_autodoc_typehints",
    "breathe",
]

# The docstrings generated from the Doxygen comments use Google-style "Args:"
# blocks (see madspace/generate_docstrings.py); napoleon turns them into proper
# parameter field lists. Keep the numpy parser off to avoid ambiguity.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_theme_options = {
    "sidebar_hide_name": True,
    "light_logo": "logo-light.png",
    "dark_logo": "logo-dark.png",
}

autoclass_content = "both"
# add_module_names = False
typehints_fully_qualified = False

breathe_projects = {"madspace": "../build/doxygenxml"}
breathe_default_project = "madspace"
breathe_domain_by_extension = {"h": "cpp"}
breathe_default_members = ("members", "undoc-members")


def generate_doxygen_xml(app):
    build_dir = os.path.join(app.confdir, "..", "build")
    if not os.path.exists(build_dir):
        os.mkdir(build_dir)

    try:
        subprocess.call(["doxygen", "--version"])
        retcode = subprocess.call(["doxygen"], cwd=os.path.join(app.confdir, ".."))
        if retcode < 0:
            sys.stderr.write(f"doxygen error code: {-retcode}\n")
    except OSError as e:
        sys.stderr.write(f"doxygen execution failed: {e}\n")

    _reparent_included_members(os.path.join(app.confdir, "..", "build", "doxygenxml"))


def _reparent_included_members(xml_dir):
    """Point members that came from a #included ``*_mixin.inc`` file back at the
    header that includes them (their ``bodyfile``). Breathe only renders the
    description of a member whose ``location`` file matches the class header, so
    without this the generated FunctionBuilder instruction methods would show up
    as bare signatures.
    """
    import glob
    import xml.etree.ElementTree as ET

    for path in glob.glob(os.path.join(xml_dir, "*.xml")):
        try:
            tree = ET.parse(path)
        except ET.ParseError:
            continue
        changed = False
        for loc in tree.getroot().iter("location"):
            f = loc.get("file", "")
            body = loc.get("bodyfile", "")
            if f.endswith("_mixin.inc") and body and body != f:
                loc.set("file", body)
                changed = True
        if changed:
            tree.write(path, encoding="utf-8", xml_declaration=True)


def generate_class_pages(app):
    # docs/ holds generate_api_pages.py; confdir is docs/source
    sys.path.insert(0, os.path.abspath(os.path.join(app.confdir, "..")))
    from generate_api_pages import generate_api_pages

    generate_api_pages(
        source_dir=app.srcdir,
        xml_dir=os.path.join(app.confdir, "..", "build", "doxygenxml"),
    )


def setup(app):
    # Registration order is run order: XML first, then the pages that read it.
    app.connect("builder-inited", generate_doxygen_xml)
    app.connect("builder-inited", generate_class_pages)
