import sys
import os
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SCOTCH'
copyright = '2025, Spencer Halberg-Spencer, Harmon Bhasin, Sushmita Roy'
author = 'Spencer Halberg-Spencer, Harmon Bhasin, Sushmita Roy'
release = "'1.1.0'"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode',
              'sphinx_autodoc_typehints',
              'myst_parser',
              'nbsphinx']

templates_path = ['_templates']
exclude_patterns = []

language = 'Python'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

sys.path.insert(0, os.path.abspath('../../../SCOTCH'))

source_suffix = ['.rst',
                 '.md']

ignore_patterns = ['notebooks/**']




