Develop
===============

Documentation
~~~~~~~~~~~~~
Make sure that the cortex package is installed and configured.

Generate API documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^
::

    $ cd docs
    $ sphinx-apidoc -f -o source ../cortex

Verify Documentation Formatting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

    $ cd docs/source
    $ doc8 -v

Building Documentation
^^^^^^^^^^^^^^^^^^^^^^

::

    $ cd docs
    $ make html

Serving Documentation Locally
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    $ cd docs/build/html
    $ python -m http.server 8000 --bind 127.0.0.1


