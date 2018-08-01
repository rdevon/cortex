Develop
===============

Documentation
~~~~~~~~~~~~~
Make sure that the cortex package is installed and configured. For development purpose, if you are
making changes to documentation, for example modifications inside docstrings or changes in some .rst
files

Building Documentation
^^^^^^^^^^^^^^^^^^^^^^
To build the documentation, the docs.py script under the root of the project is facilitating the process.
Before making a Pull Request to the remote repository, you should run the script.
::

    $ python docs.py

Serving Documentation Locally
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you want to have a look at your changes before making a Pull Request on GitHub, it is possible to
serve locally the generated html files.
::

    $ cd docs/build/html
    $ python -m http.server 8000 --bind 127.0.0.1


