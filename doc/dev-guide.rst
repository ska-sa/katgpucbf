.. _dev-guide:

Development Guide
=================
Setting up a development environment
------------------------------------

First, clone the repo from Github.

.. code-block:: bash

  git clone git@github.com:ska-sa/katgpucbf.git

Next, create and activate a Python `virtual environment`_ for your work. While
you can use Python's built-in virtual environment support, we recommend
`pyenv`_ and `pyenv-virtualenv`_, which make it easy to manage different
virtual environments and even different Python versions for different
projects.

.. _virtual environment: https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-and-use-virtual-environments
.. _pyenv: https://github.com/pyenv/pyenv
.. _pyenv-virtualenv: https://github.com/pyenv/pyenv-virtualenv


A setup script (:program:`dev-setup.sh`) is included for your convenience to
get going.

.. code-block:: bash

  cd katgpucbf
  ./dev-setup.sh

The script will perform the following actions:

  - Install all the requirements for running, developing and building this
    documentation.
  - Install the :mod:`katgpucbf` package itself, in editable mode.
  - Build this documentation.
  - Install :program:`pre-commit` to help with keeping things tidy.

And you are ready to start developing with :mod:`katgpucbf`!

.. tip::

  I don't recommend using the  :program:`dev-setup.sh` for anything other than
  initial setup. If you run it again, the requirements will be re-installed, and
  the package will be re-installed in editable mode. It's unlikely that any of
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
  :external+black:doc:`black <getting_started>`, `isort`_ and `ruff`_.
- ``mypy.ini`` file for :external+mypy:doc:`mypy <getting_started>`, a static type checker
  (or lint-like tool) for type annotations in the Python code - according to
  :pep:`484` and :pep:`526` notation.

.. _isort: https://pycqa.github.io/isort/
.. _ruff: https://docs.astral.sh/ruff/

Pre-commit compliance
^^^^^^^^^^^^^^^^^^^^^

Contributors who prefer to develop without pre-commit enabled will be required
to ensure that any submissions pass all the checks described here before they
can be accepted and merged.

No judgement, we know pre-commit can be annoying if you're not used to it.
This is in place in order to keep the code-base consistent so we can focus
on the work at hand - rather than maintaining code readability and appearance.

Installation Prerequisites
^^^^^^^^^^^^^^^^^^^^^^^^^^

Although :external+black:doc:`black <getting_started>`, `ruff`_,
and :external+mypy:doc:`mypy <getting_started>`
are used, the only prerequisite is the **pre-commit**  Python library. That is,
the YAML configuration file is set up so that when the pre-commit hooks are
installed, all dependencies are automatically installed. (Note, they won't be
available to you in your Python environment; they will be used only by pre-commit.
If you want to use them separately, you will need to install them separately with pip.)

Light-weight installation
-------------------------

There are a few cases where it is unnecessary (and inconvenient) to install
CUDA, such as for building the documentation or launching a correlator on a
remote system. If one does not use :program:`dev-setup.sh` but installs
manually (in a virtual environment) using ``pip install -e .``, then only a
subset of dependencies are installed. There are also some optional extras that
can be installed, such as ``pip install -e ".[doc]"`` to install necessary
dependencies for building the documentation. Refer to ``pyproject.toml`` to see
what extras are available.

This is not recommended for day-to-day development, because it will install
whatever is the latest version at the time, rather than the known-good versions
pinned in requirements.txt.

Boiler-plate files
------------------

The package contains the following boiler-plate files:

- ``Dockerfile`` for generating repeatable container images which are capable of
  running this package.
- ``Jenkinsfile`` for a Jenkins Continuous Integration (CI) server to run unit
  tests automatically. Comments in the file document hardware requirements.
- ``requirements.in`` and ``requirements-dev.in`` specify the Python
  prerequisites for running and developing with this package respectively.
  They are used as inputs to `uv pip compile`_.
- ``requirements.txt`` and ``requirements-dev.txt`` list complete pinned
  requirements, to ensure repeatable operation. These are the output of the
  ``uv pip compile`` process mentioned above. These should be passed to ``pip
  install`` with the ``-r`` flag to install the requirements either to run or
  develop. Development requires an additional set of packages which are not
  required for users to run the software (such as pytest). Note that developers
  should install both sets of requirements, not just the development ones.
- ``pyproject.toml`` is a standard file included with many Python projects. It
  is used to store some configuration for pre-commit (as described above), some
  configuration options for :mod:`pytest`, and other configuration as described
  :external+setuptools:doc:`here <userguide/quickstart>`.

.. _uv pip compile: https://docs.astral.sh/uv/pip/compile/

Making a contribution
---------------------

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

Raising the Pull Request
^^^^^^^^^^^^^^^^^^^^^^^^

For contributions from within the organisation, the steps are below. There should
normally be a Jira ticket associated with the change. For very minor changes (such
as updating a dependency or fixing a typo in documentation) it is acceptable to
make a PR without an associated ticket, in which case Jira-related instructions
can be ignored.

1. Ensure you have the latest ``main`` in your local directory.
2. Develop your contribution on a branch off ``main``.

   a. Please prefix your branch name with the Jira ticket number. e.g. For
      ticket NGC-1234, a suitable branch name would be
      ``NGC-1234-task-description``.
   b. Once you have created your branch, push it to this repository.

3. Once your changes are ready for review, create the `Pull Request`_. Discuss
   with the team who will do the review and add them as a Reviewer. If in doubt,
   request a review from the Team Lead.

   a. Note that we do not use the *Assignee* field for GitHub PRs.
   b. Where your changes are related to a Jira ticket, please assign it to the
      reviewer and move the ticket to **Feedback requested**.
   c. If GitHub's Pull Request page indicates there are merge conflicts ahead
      (*Can't automatically merge*), please resolve any merge conflicts before
      requesting a review from a developer.
   d. A template is provided for the Pull Request description. Please fill in
      the template with as much detail as possible and complete the checklist
      items as indicated. You may prefix a checklist item with `(n/a)` if an
      item is not applicable to your PR.

4. This repo is monitored by the organisation's in-house Jenkins. It does an
   end-to-end build when a Pull Request is created. Please address any issues
   or failures reported by Jenkins and update your Reviewers accordingly.
5. Once you have received a review, respond to any comments and apply any
   requested changes/fixes.

   a. The reviewer should move the Jira ticket associated with this PR back
      to **In progress** and assign it back to you.

6. Re-request a review from the Reviewer(s) of your the PR and repeat steps 4
   and 5 until there is conclusion on Approval. Note that, by default, the
   responsibility for merging is with the submitter rather than the reviewer.
   The submitter may be aware of other reasons to hold off on merging, such as
   PRs on related repositories that need to be merged simultaneously to update
   an interface. However, for trivial changes (such as typos in comments) the
   reviewer may do the merge.
7. Once you have received approval to merge, click **Merge pull request** with
   its default setting of "Create a merge commit". Github will automatically
   delete your branch after you merge.

For contributors outside the organisation, the steps are:

1. `Fork the repository`_ on GitHub to your own account.
2. When you deem your changes ready for review, please ensure you run the unit
   tests locally to ensure they pass, as our Jenkins does not run on forks.
3. Create a `Pull Request`_ from your fork to the main repository - similar to
   the internal process described above, but without the Jira ticket steps.
4. Address any review comments as described in step 5 above.
5. Note that your reviewer will merge your Pull Request upon approval.

GitHub's Pull Request page and manner of reviewing may make it tough to trace
comments (especially after content has changed). Please do not resolve comments
until you (and your reviewer(s)) are absolutely sure the original comment has
been addressed.


.. _Pull Request: https://docs.github.com/en/pull-requests
.. _Fork the repository: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks
