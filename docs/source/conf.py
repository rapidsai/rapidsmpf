# Copyright (c) 2025, NVIDIA CORPORATION.
# numpydoc ignore=GL08
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'rapidsmp'
copyright = '2025, NVIDA Corporation'
author = 'NVIDA Corporation'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
]

templates_path = ['_templates']
exclude_patterns = []
autosummary_generate = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']


html_theme_options = {
    "external_links": [],
    # https://github.com/pydata/pydata-sphinx-theme/issues/1220
    "icon_links": [],
    "github_url": "https://github.com/rapidsai/rapidsmp",
    "twitter_url": "https://twitter.com/rapidsai",
    "show_toc_level": 2,
    "navbar_align": "right",
    "navigation_with_keys": True,
}
include_pandas_compat = True


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
}
