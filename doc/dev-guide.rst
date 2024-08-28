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
work, navigate into the :mod:`katgpucbf` directory and source the virtual
environment directly:

.. code-block:: bash

  source .venv/bin/activate

And you are ready to start developing with :mod:`katgpucbf`!

.. tip::

  I don't recommend using the  :program:`dev-setup.sh` for anything other than
  initial setup. If you run it again, the requirements will be re-installed, and
  the module will be re-installed in editable mode. It's unlikely that any of
  this will be harmful in any way, but it will use up a few minutes. You
  probably won't want to do that every time.


Pre-commit
----------

:mod:`katgpucbf` is configured with pre-commit for auto-formatting Python code.
Pre-commit runs whenever anything is committed to the repository.

For more detailed information, please consult the `pre-commit`_ documentation.
The installation and initialisation of the pre-commit flow is handled in
:program:`dev-setup.sh`.

.. _pre-commit: https://pre-commit.com/

Configuration Files
^^^^^^^^^^^^^^^^^^^

This repo contains the following configuration files for the pre-commit flow
to monitor Python development.

- ``.pre-commit-config.yaml`` for `pre-commit`_ specifies which git hooks will
  be run before committing to the repo.
- ``pyproject.toml`` dictates the configuration of utilities such as
  :external+black:doc:`black <getting_started>` and `isort`_.
- ``.flake8`` for :external+flake8:doc:`flake8 <user/index>`, a tool for enforcing
  :pep:`8`-based style guide for Python.
- ``.pydocstyle.ini`` for :external+pydocstyle:doc:`pydocstyle <usage>`, a tool
  for enforcing :pep:`257`-based docstring style guides for Python.
- ``mypy.ini`` file for :external+mypy:doc:`mypy <getting_started>`, a static type checker
  (or lint-like tool) for type annotations in the Python code - according to
  :pep:`484` and :pep:`526` notation.

.. _isort: https://pycqa.github.io/isort/

Installation Prerequisites
^^^^^^^^^^^^^^^^^^^^^^^^^^

Although :external+black:doc:`black <getting_started>`, :external+flake8:doc:`flake8 <user/index>`,
:external+pydocstyle:doc:`pydocstyle <usage>` and :external+mypy:doc:`mypy <getting_started>`
are used, the only prerequisite is the **pre-commit**  Python library. That is,
the YAML configuration file is set up so that when the pre-commit hooks are
installed, all dependencies are automatically installed. (Note, they won't be
available to you in your Python environment, they will be used only by pre-commit.
If you want to use them separately, you will need to install them separately with pip.)

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

Boiler-plate files
------------------

The module contains the following boiler-plate files:

- ``Dockerfile`` for generating repeatable container images which are capable of
  running this package.
- ``Jenkinsfile`` for a Jenkins Continuous Integration (CI) server to run unit
  tests automatically. Comments in the file document hardware requirements.
- ``requirements.in`` and ``requirements-dev.in`` specify the Python
  prerequisites for running and developing with this package respectively.
  They are used as inputs to `pip-compile`_.
- ``requirements.txt`` and ``requirements-dev.txt`` list complete pinned
  requirements, to ensure repeatable operation. These are the output of the
  ``pip-compile`` process mentioned above. These should be passed to ``pip
  install`` with the ``-r`` flag to install the requirements either to run or
  develop. Development requires an additional set of packages which are not
  required for users to run the software (such as pytest). Note that developers
  should install both sets of requirements, not just the development ones.
- ``setup.cfg`` allows :external+setuptools:doc:`setuptools  <setuptools>`
  to install this package.
- ``pyproject.toml`` is a standard file included with many Python projects. It
  is used to store some configuration for pre-commit (as described above), some
  configuration options for :mod:`pytest`, and other configuration as described
  :external+pip:doc:`here <reference/build-system/pyproject-toml>`.

.. _pip-compile: https://pip-tools.readthedocs.io/en/latest/#without-setup-py

Preparing to raise a Pull Request
---------------------------------

Pre-commit compliance
^^^^^^^^^^^^^^^^^^^^^

Contributors who prefer to develop without pre-commit enabled will be required
to ensure that any submissions pass all the checks described here before they
can be accepted and merged.

No judgement, we know pre-commit can be annoying if you're not used to it.
This is in place in order to keep the code-base consistent so we can focus
on the work at hand - rather than maintaining code readability and appearance.

Module documentation updates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:mod:`katgpucbf` holds documentation within its code-base.
:external+sphinx-apidoc:doc:`sphinx-apidoc <index>` provides a manner to generate
module documentation as reStructuredText. If you, the developer, add or remove
a module or file, execute the full ``sphinx-apidoc`` command below to regenerate
the module documentation with your updates. The incantation below is run from the
root :mod:`katgpucbf` directory.

.. code-block:: bash

  sphinx-apidoc -efo doc/ src/

.. note::

    The above command will likely generate a :file:`modules.rst` file, which is
    not necessary to commit.
