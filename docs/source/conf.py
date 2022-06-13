#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PyDFCS documentation build configuration file, created by
# sphinx-quickstart on Tue Mar  6 13:11:15 2018.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# General information about the project.
project = 'dfcs_vipa'
copyright = '2021, Grzegorz Kowzan'
author = 'Grzegorz Kowzan'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.extlinks',
    'sphinx.ext.mathjax',
    # 'sphinx.ext.autosummary',
    # 'numpydoc'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
exclude_patterns = []

# -- Formatting options
add_module_names = False
python_use_unqualified_type_names = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
# html_sidebars = {
#     '**': [
#         'about.html',
#         'navigation.html',
#         'searchbox.html',
#     ]
# }
# html_show_sourcelink = True

# -- Intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'xarray': ('http://xarray.pydata.org/en/stable/', None),
}

# -- pygments
pygments_style = 'sphinx'

# -- numpydoc
# numpydoc_class_members_toctree = False
# numpydoc_show_class_members = False
# numpydoc_show_inherited_class_members = False

# -- autodoc
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True
}
autoclass_content = 'both'
autodoc_member_order = 'bysource'
autodoc_preserve_defaults = True
