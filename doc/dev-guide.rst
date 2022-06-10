.. _getting-started:

Getting Started
===============

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

Pre-commit's `documentation`_ describes what it does in sufficient detail. The
configuration is located in ``.pre-commit-config.yaml``, though the various
other modules which are loaded in this repo will have their configuration in
various places.

.. todo:: Merge the readme from the pre-commit repo into this section?

.. _documentation: https://pre-commit.com/




TODOs
-----

This list is assembled from throughout the documentation. If you're looking for
something to keep yourself busy, this is a good place to start.

.. tip::

  This list only includes TODOs formatted in a way that Sphinx understands.
  There are likely others formatted as comments throughout the code which don't
  appear listed here. ``grep`` can help you find them!

  The ``test``  and ``qualification`` folders are not pulled in by Sphinx, and
  so any TODOs there will also not be included in this list.

.. todolist::
