# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# numpydoc ignore=GL08
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from __future__ import annotations

import datetime
import glob
import os
import re
import xml.etree.ElementTree as ET

from enum import IntEnum, IntFlag
from typing import Any

from packaging.version import Version
from sphinx.ext.autodoc import ClassDocumenter

import rapidsmpf

project = "NVIDIA RapidsMPF"
copyright = f"2025-{datetime.datetime.today().year}, NVIDIA Corporation"
author = "NVIDIA Corporation"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
RAPIDSMPF_VERSION = Version(rapidsmpf.__version__)
# The short X.Y version.
version = f"{RAPIDSMPF_VERSION.major:02}.{RAPIDSMPF_VERSION.minor:02}"
# The full version, including alpha/beta/rc tags.
release = f"{RAPIDSMPF_VERSION.major:02}.{RAPIDSMPF_VERSION.minor:02}.{RAPIDSMPF_VERSION.micro:02}"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "numpydoc",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "myst_nb",
    "breathe",
]

# Disambiguate section anchors across documents
autosectionlabel_prefix_document = True

# Breathe Configuration
breathe_projects = {"librapidsmpf": "../../cpp/doxygen/xml"}
breathe_default_project = "librapidsmpf"


def clean_doxygen_xml(path: str) -> None:
    # Doxygen 1.9.1 misparses concepts and requires clauses in its XML output.
    return_types = {
        "rapidsmpf::BufferResource::reserve_or_fail": "MemoryReservation",
        "rapidsmpf::ContentDescription::ContentDescription": "",
        "rapidsmpf::owner_equal": "bool",
        "rapidsmpf::safe_cast": "To",
    }

    for filename in glob.glob(os.path.join(path, "*.xml")):
        tree = ET.parse(filename)
        changed = False
        for section in tree.findall(".//sectiondef"):
            for member in list(section.findall("./memberdef")):
                type_node = member.find("type")
                type_text = "".join(type_node.itertext()) if type_node is not None else ""
                if type_text == "concept":
                    section.remove(member)
                    changed = True
                    continue

                definition = member.find("definition")
                if type_text.startswith("requires ") and definition is not None:
                    qualified_name = "".join(definition.itertext()).rsplit(" ", 1)[-1]
                    if qualified_name in return_types:
                        return_type = return_types[qualified_name]
                        type_node.clear()
                        type_node.text = return_type
                        definition.clear()
                        definition.text = f"{return_type} {qualified_name}".lstrip()
                        changed = True

                args = member.find("argsstring")
                if args is not None and args.text is not None:
                    cleaned_args = re.sub(r"\) requires.*", ")", args.text)
                    if cleaned_args != args.text:
                        args.text = cleaned_args
                        changed = True

        if changed:
            tree.write(filename, encoding="UTF-8", xml_declaration=True)


for project_path in breathe_projects.values():
    clean_doxygen_xml(project_path)

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
    "public_docs_features": os.environ.get("CI") == "true",
    "external_links": [],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/rapidsai/rapidsmpf",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
    "navbar_align": "right",
    "navbar_center": "navbar-nav, version-switcher, navbar-external-links",
    "navigation_with_keys": True,
    "show_toc_level": 2,
    "switcher": {
        "json_url": "https://docs.nvidia.com/rapidsmpf/versions.json",
        "version_match": version,
    },
}


# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = "nvidia_sphinx_theme"

numpydoc_class_members_toctree = False


# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-default_role
default_role = "any"


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
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


def on_missing_reference(app, env, node, contnode):
    if (refid := node.get("refid")) is not None and "hpp" in refid:
        return contnode

    if node["refdomain"] in ("std", "cpp") and (
        reftarget := node.get("reftarget")
    ) is not None:
        if match := re.search("(.*)<.*>", reftarget):
            reftarget = match.group(1)

        prefixes = [
            "rapidsmpf::",
            "rapidsmpf::bootstrap::",
            "rapidsmpf::coll::",
            "rapidsmpf::communicator::",
            "rapidsmpf::config::",
            "rapidsmpf::mpi::",
            "rapidsmpf::rrun::",
            "rapidsmpf::shuffler::",
            "rapidsmpf::streaming::",
            "rapidsmpf::streaming::actor::",
            "",
        ]
        for name, _, _, _, _, _ in env.domains["cpp"].get_objects():
            for prefix in prefixes:
                if name == f"{prefix}{reftarget}" or f"{prefix}{name}" == reftarget:
                    if (
                        ref := env.domains["cpp"].resolve_xref(
                            env,
                            node.get("refdoc"),
                            app.builder,
                            node["reftype"],
                            name,
                            node,
                            contnode,
                        )
                    ) is not None:
                        return ref
        return contnode

    return None


def setup(app):
    app.connect("missing-reference", on_missing_reference)
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
    # We're subclassing this from RMM, and sphinx can't find these methods.
    ("py:obj", "rapidsmpf.rmm_resource_adaptor.RmmResourceAdaptor.allocate"),
    ("py:obj", "rapidsmpf.rmm_resource_adaptor.RmmResourceAdaptor.deallocate"),
    ("py:obj", "rapidsmpf.memory.buffer_resource.OwningDeviceMemoryResource.allocate"),
    ("py:obj", "rapidsmpf.memory.buffer_resource.OwningDeviceMemoryResource.deallocate"),
]
