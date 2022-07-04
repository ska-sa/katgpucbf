.. _dev-environment:

Development Environment
=======================

Setting up a development environment
------------------------------------

First, clone the repo from Github.

.. code-block:: bash

  git clone git@github.com:ska-sa/katgpucbf.git

A setup script (:program:`dev-setup.sh`) is included for your convenience to
get going.

.. code-block:: bash

  cd katgpucbf
  source dev-setup.sh

The script will perform the following actions:

  - Create a fresh Python virtual environment.
  - Install all the requirements for running, developing and building this
    documentation.
  - Install the :mod:`katgpucbf` package itself, in editable mode.
  - Build this documentation.
  - Install :program:`pre-commit` to help with keeping things tidy.

Sourcing the script instead of executing it directly will keep your virtual
environment active, so you can get going straight away. Next time you want to
work, you can just source the virtual environment directly:

.. code-block:: bash

  source .venv/bin/activate

And you are ready to start developing with :mod:`katgpucbf`!

.. tip::

  I don't recommend using the  ``dev-setup.sh`` for anything other than initial
  setup. If you run it again, the requirements will be re-installed, and the
  module will be re-installed in editable mode. It's unlikely that any of this
  will be harmful in any way, but it will use up a few minutes. You probably
  won't want to do that every time.


Pre-commit
----------

As alluded to in the previous section, :mod:`katgpucbf` contains a pre-commit
workflow for auto-formatting Python code. This workflow checks for
inconsistencies in style, design and complexity according to `PEP8`_ guidelines.
It also checks for compliance with Python docstring conventions according to
`PEP257`_ and supports `mypy`_ type-checking if type hints are used in the code.

The workflow runs whenever Python code is committed to :mod:`katgpucbf`.

This section discusses a high-level view of the pre-commit flow and its
constituent parts. For more detailed information, please consult its
`documentation`_. The inclusion of the ``pre-commit`` library requirement,
subsequent installation and initialisation of the pre-commit flow is carried
out for you in the execution of ``dev-setup.sh``. The following steps are
discussed should you be interested, or even required, to update various
aspects of the pre-commit flow.

Initial Setup
^^^^^^^^^^^^^

This repo contains the following configuration files for the pre-commit flow
to monitor Python development.

- ``.pre-commit-config.yaml`` for `pre-commit`_ specifies which git hooks will
  be run before committing to the repo.
- ``pyproject.toml`` contains configuration for `black`_, The Uncompromising
  Python Code Formatter. This can also contain configuration if more advanced
  build systems such as C++ compilation (e.g. `pybind11`_) are used.
- ``.flake8`` for `flake8`_, a tool for enforcing PEP8-based style guide for
  Python.
- ``.pydocstyle.ini`` for `pydocstyle`_, a tool for enforcing PEP257-based
  doc-string style guides for Python.
- ``mypy.ini`` file for `mypy`_, a static type checker (or lint-like tool)
  for type annotations in the Python code - according to `PEP484`_ and
  `PEP526`_ notation.

Install Prerequisites
^^^^^^^^^^^^^^^^^^^^^

Although ``black``, ``flake8``, ``mypy`` and ``pydocstyle`` are used, the only
prerequisite is the **pre-commit**  Python library. That is, the YAML
configuration file is set up so that when the pre-commit hooks are installed,
all dependencies are automatically installed. (Note, they won't be available to
you in Python, they will be used only by pre-commit. If you want to use them
separately, you will need to install them individually with pip.)

**pre-commit** is essentially a framework for managing git hooks and can be
installed by running:

  .. code-block:: bash

    $ pip install pre-commit


A reminder that the **pre-commit** library is already listed as a development
requirement for :mod:`katgpucbf`.

Generate pre-commit Git Hooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To generate the git hooks themselves, the following command is run in the
project directory. Note that the ``.pre-commit-config.yaml`` file must be in
the project directory **before** generating the git hooks.

  .. code-block:: bash

    $ pre-commit install

After this, ``pre-commit`` will run automatically on the execution of the
``git commit`` command with the installed hooks.

.. _documentation: https://pre-commit.com/
.. _PEP8: https://www.python.org/dev/peps/pep-0008/
.. _PEP257: https://www.python.org/dev/peps/pep-0257/
.. _PEP484: https://www.python.org/dev/peps/pep-0484/
.. _PEP526: https://www.python.org/dev/peps/pep-0526/
.. _pre-commit: https://pre-commit.com/
.. _black: https://github.com/psf/black
.. _flake8: https://flake8.pycqa.org/en/latest/
.. _pydocstyle: http://www.pydocstyle.org/
.. _mypy: https://mypy.readthedocs.io/en/stable/index.html
.. _pybind11: https://pybind11.readthedocs.io/

Light-weight installation
-------------------------

There are a few cases where it is unnecessary (and inconvenient) to install
CUDA, such as for building the documentation or launching a correlator on a
remote system. If one does not use :program:`dev-setup.sh` but installs
manually (in a virtual environment) using ``pip install -e .``, then only a
subset of dependencies are installed. There are also some optional extras that
can be installed, such as ``pip install -e ".[doc]"`` to install necessary
dependencies for building the documentation. Refer to ``setup.cfg`` to see what
extras are available.

This is not recommended for day-to-day development, because it will install
whatever is the latest version at the time, rather than the known-good versions
pinned in requirements.txt.
