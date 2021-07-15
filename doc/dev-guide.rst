Guideline for Development
=========================


Getting Started
---------------
First, clone the repo from Github. Don't forget to clone :option:`!--recursive`,
so that the 3rd-party :mod:`.spead2` dependency gets pulled in.

.. code-block:: bash

  git clone --recursive git@github.com:ska-sa/katgpucbf.git

If you do forget the :option:`!--recursive`, then you can pull it in afterwards:

.. code-block:: bash

  cd katgpucbf
  git submodule update --init

A setup script (:program:`dev-setup.sh`) is included for your convenience to
get going.

.. code-block:: bash

  cd katgpucbf
  ./dev-setup.sh

The script will perform the following actions:

  - Create a fresh Python 3.8 virtual environment.
  - Install all the requirements for running, developing and building this
    documentation.
  - Install the :mod:`katgpucbf` package itself, in editable mode.
  - Compile the ``.so`` files needed to run the :mod:`.xbgpu` unit tests.
  - Install :program:`pre-commit` to help with keeping things tidy.

You will still need to activate your new virtual environment before you can get
going:

.. code-block:: bash

  source .venv/bin/activate

And you are ready to start developing with :mod:`katgpucbf`!

Pre-commit
----------

`Pre-commit's documentation`_ describes more or less what it does. The
configuration is located in ``.pre-commit-config.yaml``, though the various
other modules which are loaded in this repo will have their configuration in
various places.

.. todo:: Bring in the readme from the pre-commit repo here?

.. _Pre-commit's documentation:: https://pre-commit.com/

Unit Testing
------------
