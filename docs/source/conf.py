# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# numpydoc ignore=GL08
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from __future__ import annotations

from enum import IntEnum, IntFlag
from typing import Any

from sphinx.ext.autodoc import ClassDocumenter

project = "rapidsmpf"
copyright = "2025-2026, NVIDIA Corporation"
author = "NVIDIA Corporation"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "numpydoc",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "myst_nb",
    "breathe",
]
# Breathe Configuration
breathe_projects = {"librapidsmpf": "../../cpp/doxygen/xml"}
breathe_default_project = "librapidsmpf"

templates_path = ["_templates"]
exclude_patterns = []
autosummary_generate = True

# MyST parser configuration
myst_heading_anchors = 3

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]
html_css_files = ["custom.css"]


html_theme_options = {
    "external_links": [],
    # https://github.com/pydata/pydata-sphinx-theme/issues/1220
    "icon_links": [],
    "github_url": "https://github.com/rapidsai/rapidsmpf",
    "twitter_url": "https://twitter.com/rapidsai",
    "show_toc_level": 2,
    "navbar_align": "right",
    "navigation_with_keys": True,
}


# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = "pydata_sphinx_theme"
html_logo = "_static/RAPIDS-logo-purple.png"

numpydoc_class_members_toctree = False


# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-default_role
default_role = "any"


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "dask-cuda": ("https://docs.rapids.ai/api/dask-cuda/stable/", None),
    "cudf": ("https://docs.rapids.ai/api/cudf/stable/", None),
    "dask": ("https://docs.dask.org/en/stable/", None),
    "distributed": ("https://distributed.dask.org/en/stable/", None),
    "ray": ("https://docs.ray.io/en/latest/", None),
}


class CythonIntEnumDocumenter(ClassDocumenter):
    """
    Custom autodoc documenter for Cython cpdef enum classes (IntEnum/IntFlag).
    Without this, autodoc renders inherited int methods (denominator, imag, etc.)
    instead of the actual enum members.
    """

    objtype = "enum"
    directivetype = "class"
    priority = 10 + ClassDocumenter.priority

    option_spec = dict(ClassDocumenter.option_spec)

    @classmethod
    def can_document_member(
        cls, member: Any, membername: str, isattr: bool, parent: Any
    ) -> bool:
        try:
            return issubclass(
                member, (IntEnum, IntFlag)
            ) and member.__module__.startswith("rapidsmpf")
        except TypeError:
            return False

    def add_content(self, more_content) -> None:
        doc_as_attr = self.doc_as_attr
        self.doc_as_attr = False
        super().add_content(more_content)
        self.doc_as_attr = doc_as_attr
        source_name = self.get_sourcename()
        enum_object: IntEnum = self.object

        self.add_line("", source_name)
        self.add_line(".. container:: enum-members", source_name)
        self.add_line("", source_name)

        for the_member_name in enum_object.__members__:
            self.add_line(f"   .. attribute:: {the_member_name}", source_name)
            self.add_line("", source_name)


def setup(app):
    app.registry.add_documenter("enum", CythonIntEnumDocumenter)

    # Prevent Sphinx from replacing native Cython modules with .pyi stubs.
    # When .pyi files are installed alongside .so files, Sphinx 8.2+ prefers
    # the stub, which causes autodoc to miss Cython module-level functions
    # (they lack docstrings in the stub and get skipped as undocumented).
    # The importer skips this lookup if the module already happens to be
    # imported which is why it only seems to exhibit as not finding docs
    # for some modules.
    import sphinx.ext.autodoc.importer as _importer

    _importer._find_type_stub_spec = lambda spec, modname: (spec, None)


nitpick_ignore_regex = [
    # Cython turns __call__ into a slot_wrapper that autodoc doesn't understand.
    ("py:obj", "rapidsmpf.memory.buffer_resource.LimitAvailableMemory.__call__"),
    # We're subclassing this from RMM, and sphinx can't find these methods.
    ("py:obj", "rapidsmpf.rmm_resource_adaptor.RmmResourceAdaptor.allocate"),
    ("py:obj", "rapidsmpf.rmm_resource_adaptor.RmmResourceAdaptor.deallocate"),
    ("py:obj", "rapidsmpf.rmm_resource_adaptor.RmmResourceAdaptor.get_upstream"),
]
