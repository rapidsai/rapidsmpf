# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# numpydoc ignore=GL08
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'rapidsmpf'
copyright = '2025, NVIDIA Corporation'
author = 'NVIDIA Corporation'

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

templates_path = ['_templates']
exclude_patterns = []
autosummary_generate = True

# MyST parser configuration
myst_heading_anchors = 3

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']
html_css_files = ['custom.css']


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


nitpick_ignore_regex = [
    # Ignore WARNING: py:class reference target not found: Table [ref.class] in unpack_and_concat
    ('py:class', 'Table'),
    # Ignore TypeVars being assumed to be classes
    # https://github.com/sphinx-doc/sphinx/issues/10974
    ("py:class", "DataFrameT"),
    ("py:class", "rapidsmpf.integrations.dask.core.DataFrameT"),
    # Unclear why this was causing a warning
    ("py:obj", "rapidsmpf.memory.buffer_resource.LimitAvailableMemory.__call__"),
    # autodoc fails to generate references for integer methods (real, image, etc.)
    # for IntEnums coming from Cython.
    ("py:obj", "rapidsmpf.communicator.communicator.LOG_LEVEL.*"),
    ("py:obj", "rapidsmpf.memory.buffer.MemoryType.*"),
    ("py:obj", "rapidsmpf.memory.scoped_memory_record.AllocType.*"),
    ("py:obj", "(denominator|imag|numerator|real)"),
    ('py:class', 'rmm.pylibrmm.stream.Stream'),
    ('py:class', 'rmm.pylibrmm.memory_resource.DeviceMemoryResource'),
    # We're subclassing this from RMM, and sphinx can't find these methods.
    ("py:obj", "rapidsmpf.rmm_resource_adaptor.RmmResourceAdaptor.allocate"),
    ("py:obj", "rapidsmpf.rmm_resource_adaptor.RmmResourceAdaptor.deallocate"),
    ("py:obj", "rapidsmpf.rmm_resource_adaptor.RmmResourceAdaptor.get_upstream"),
]
