Unit Testing
------------

Unit testing for this module is performed using :mod:`.pytest` with support from
:mod:`!pytest-asyncio`. Unit test files should follow :mod:`.pytest` conventions.
Additionally, :mod:`.coverage` is used to give the developer insight into what
the unit tests are actually testing, and what code remains untested. Both of
these packages are installed if the ``dev-setup.sh`` script is used as described
in :ref:`dev-environment`.

In order to run the tests, use the following command:

.. code-block:: bash

  pytest

:mod:`.pytest` reads its configuration from ``pyproject.toml``. Also installed
as part of this project's ``requirements-dev.txt`` are :mod:`.coverage` and
:mod:`.pytest-cov`. As currently configured, running the unit tests as described
above will execute a subset of the parameterised tests (see the docstring for
``test/conftest.py``). While every combination of parameters won't always be
tested, each individual parameter will be tested at least once.

If you'd like an HTML test-coverage report (at the expense of a slightly longer
time taken to run the test), execute ``pytest`` with the :option:`!--cov` flag.
This report can then be viewed by:

.. code-block:: bash

  xdg-open htmlcov/index.html

Or, if you are developing on a remote server:

.. code-block:: bash

  cd htmlcov && python -m http.server 8089

If you are using VSCode, the editor will prompt you to open the link in a
browser, and automatically forward the port to your ``localhost``. If not, or if
you'd prefer to do it the old-fashioned way, point a browser at port ``8089``
on the machine that you are developing on.

The results will look something like this:

.. image:: images/coverage_screenshot.png

The colour key is at the top of the page, but briefly, lines marked in green
were executed by the tests, red were not. Yellow lines indicate branches which
were only partially covered, i.e. all possible ways to branch were not tested.
In the cases shown, it is because only expected values were passed to the
function in question: the unit tests didn't pass invalid inputs in order to
check that exceptions were raised appropriately.

On the right hand side, a context is shown for the lines that were executed, as
shown in this image:

.. image:: images/coverage_screenshot_contexts.png

On the left side of the ``|`` is the static context - in this case showing
information regarding the git commit that I ran the test on. The right side
shows the dynamic context - in this case, two different tests both executed this
code during the course of their run.

.. note::

  :mod:`.coverage`\'s "dynamic context" output is currently specified by
  :mod:`.pytest-cov` to describe the test function which executed the line of
  code in question. If desired, it can instead be specified in coverage's
  configuration as described in `coverage's documentation`_. This produces a
  slightly different output which conveys more or less similar information.

  .. _coverage's documentation: https://coverage.readthedocs.io/en/latest/contexts.html#dynamic-contexts

  :mod:`.coverage`\'s `static context`_ is more difficult to specify in a way that
  is useful. To generate the report above, I executed the following command:

  .. _static context: https://coverage.readthedocs.io/en/latest/contexts.html#static-contexts

  .. code-block:: bash

    coverage run --context=$(git describe --tags --dirty --always)

  This gives more useful information about exactly what code was run, and whether
  it's committed or dirty. Unfortunately, doing things this way you miss out on
  the features of :mod:`.pytest-cov`. :mod:`.coverage` supports specifying a
  static context using either the command line (as shown) or via its
  configuration file, including reading of environment variables, but support
  doesn't extend to evaluating arbitrary shell expressions as is possible from
  the command line.

  The package author `suggests`_ the use of a Makefile to generate an environment
  variable which the configuration can then use in generating a static context.
  This strikes me as a good solution, but I am reluctant to include yet another
  boiler-plate file in the repository, so I leave this to the discretion of the
  individual developer to make use of as desired.

  .. tip::

    Although having said that, the Makefile could also replace dev-setup.sh,
    allowing the developer to do something like

    .. code-block:: bash

      make develop  # to set up the environment
      make test     # to actually run the tests


  .. _suggests: https://github.com/nedbat/coveragepy/issues/1190
