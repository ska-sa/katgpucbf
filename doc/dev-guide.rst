.. _dev-environment:

Development Environment
=======================

Setting up a development environment
------------------------------------

First, clone the repo from Github.

.. code-block:: bash

  git clone git@github.com:ska-sa/katgpucbf.git

A setup script (``dev-setup.sh``) is included for your convenience to
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

:mod:`katgpucbf` is configured with pre-commit for auto-formatting Python code.
Pre-commit runs whenever Python code is committed to :mod:`katgpucbf`.

For more detailed information, please consult the `pre-commit`_ documentation. The
installation and initialisation of the pre-commit flow is handled in ``dev-setup.sh``.

.. _pre-commit: https://pre-commit.com/

Initial Setup
^^^^^^^^^^^^^

This repo contains the following configuration files for the pre-commit flow
to monitor Python development.

- ``.pre-commit-config.yaml`` for `pre-commit`_ specifies which git hooks will
  be run before committing to the repo.
- ``pyproject.toml`` contains build system requirements and information, which
  are used by pip to build and use the package - e.g. `black`_, :mod:`pytest`.
  The `pyproject-toml`_ documentation outlines the potential for configuration very well.
- ``.flake8`` for `flake8`_, a tool for enforcing :pep:`8`-based style guide
  for Python.
- ``.pydocstyle.ini`` for `pydocstyle`_, a tool for enforcing :pep:`257`-based
  doc-string style guides for Python.
- ``mypy.ini`` file for `mypy`_, a static type checker (or lint-like tool)
  for type annotations in the Python code - according to :pep:`484` and
  :pep:`526` notation.

.. _black: https://github.com/psf/black
.. _pyproject-toml: https://pip.pypa.io/en/stable/reference/build-system/pyproject-toml/
.. _flake8: https://flake8.pycqa.org/en/latest/
.. _pydocstyle: http://www.pydocstyle.org/
.. _mypy: https://mypy.readthedocs.io/en/stable/index.html

Install Prerequisites
^^^^^^^^^^^^^^^^^^^^^

Although `black`_, `flake8`_, `mypy`_ and `pydocstyle`_ are used,
the only prerequisite is the **pre-commit**  Python library. That is, the YAML
configuration file is set up so that when the pre-commit hooks are installed,
all dependencies are automatically installed. (Note, they won't be available to
you in your Python environment, they will be used only by pre-commit. If you want
to use them separately, you will need to install them individually with pip.)

.. todo:: ``NGC-693``
    Explain why :mod:`katgpucbf` has a ``requirements-dev.txt`` (in addition to ``requirements.txt``).

Should you wish to develop for :mod:`katgpucbf` without the pre-commit checks enabled,
you can do so by executing the installation commands in ``dev-setup.sh`` separately
and bypassing the ``pre-commit install``.

.. note::
    Contributions (i.e. pull-requests) will not be accepted/merged until all the checks pass.

Light-weight installation
-------------------------

There are a few cases where it is unnecessary (and inconvenient) to install
CUDA, such as for building the documentation or launching a correlator on a
remote system. If one does not use ``dev-setup.sh`` but installs
manually (in a virtual environment) using ``pip install -e .``, then only a
subset of dependencies are installed. There are also some optional extras that
can be installed, such as ``pip install -e ".[doc]"`` to install necessary
dependencies for building the documentation. Refer to ``setup.cfg`` to see what
extras are available.

This is not recommended for day-to-day development, because it will install
whatever is the latest version at the time, rather than the known-good versions
pinned in requirements.txt.
